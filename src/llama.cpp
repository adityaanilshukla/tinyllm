// LLaMA-family decoder (TinyLlama-1.1B and friends): RMSNorm, RoPE, GQA, SwiGLU.
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyllm/model.h"
#include "tinyllm/ops.h"

namespace tinyllm {
namespace {

class Llama final : public Model {
 public:
  Llama(std::shared_ptr<WeightFile> wf, int max_ctx) : wf_(std::move(wf)), arena_(arena_bytes(wf_->config(), max_ctx)) {
    config_ = wf_->config();
    max_ctx_ = max_ctx;
    const Config& c = config_;
    if (c.arch != Arch::LLAMA) throw std::runtime_error("make_llama: config arch is not LLAMA");
    if (max_ctx <= 0) throw std::runtime_error("make_llama: max_ctx must be > 0");
    if (c.n_kv_heads <= 0 || c.n_heads % c.n_kv_heads != 0)
      throw std::runtime_error("make_llama: n_heads must be a multiple of n_kv_heads");
    if (c.head_dim % 2 != 0) throw std::runtime_error("make_llama: head_dim must be even");

    d_ = c.d_model;
    q_dim_ = c.n_heads * c.head_dim;
    kv_dim_ = c.n_kv_heads * c.head_dim;

    tok_emb_ = &get("tok_emb", {c.vocab_size, d_});
    if (tok_emb_->dtype != DType::F32) throw std::runtime_error("make_llama: tok_emb must be F32");
    norm_f_ = get("norm_f.w", {d_}).f32();
    lm_head_ = wf_->has("lm_head.w") ? &get("lm_head.w", {c.vocab_size, d_}) : tok_emb_;

    layers_.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; ++i) {
      const std::string p = "l" + std::to_string(i) + ".";
      Layer& L = layers_[i];
      L.attn_norm = get(p + "attn_norm.w", {d_}).f32();
      L.wq = &get(p + "attn.q.w", {q_dim_, d_});
      L.wk = &get(p + "attn.k.w", {kv_dim_, d_});
      L.wv = &get(p + "attn.v.w", {kv_dim_, d_});
      L.wo = &get(p + "attn.o.w", {d_, q_dim_});
      L.mlp_norm = get(p + "mlp_norm.w", {d_}).f32();
      L.wgate = &get(p + "mlp.gate.w", {c.d_ff, d_});
      L.wup = &get(p + "mlp.up.w", {c.d_ff, d_});
      L.wdown = &get(p + "mlp.down.w", {d_, c.d_ff});
    }

    x_ = arena_.alloc_n<float>(d_);
    h_ = arena_.alloc_n<float>(d_);
    q_ = arena_.alloc_n<float>(q_dim_);
    att_ = arena_.alloc_n<float>(q_dim_);
    proj_ = arena_.alloc_n<float>(d_);
    gate_ = arena_.alloc_n<float>(c.d_ff);
    up_ = arena_.alloc_n<float>(c.d_ff);
    scores_ = arena_.alloc_n<float>(static_cast<size_t>(c.n_heads) * max_ctx);
    logits_ = arena_.alloc_n<float>(c.vocab_size);
    cache_.init(arena_, c.n_layers, max_ctx, kv_dim_);
  }

  const float* forward(int token, int pos) override {
    const Config& c = config_;
    if (pos < 0 || pos >= max_ctx_)
      throw std::out_of_range("llama forward: pos " + std::to_string(pos) + " outside context of " +
                              std::to_string(max_ctx_));
    if (token < 0 || token >= c.vocab_size)
      throw std::out_of_range("llama forward: token id " + std::to_string(token) + " out of range");

    std::memcpy(x_, tok_emb_->f32() + static_cast<size_t>(token) * d_, sizeof(float) * d_);

    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = layers_[l];
      float* kc = cache_.k_at(l, pos);
      float* vc = cache_.v_at(l, pos);

      // Attention block. K/V are projected straight into the cache row.
      rmsnorm(h_, x_, L.attn_norm, d_, c.norm_eps);
      linear(q_, h_, *L.wq, nullptr, d_, q_dim_);
      linear(kc, h_, *L.wk, nullptr, d_, kv_dim_);
      linear(vc, h_, *L.wv, nullptr, d_, kv_dim_);
      rope(q_, c.n_heads, c.head_dim, pos, c.rope_theta);
      rope(kc, c.n_kv_heads, c.head_dim, pos, c.rope_theta);
      attention(att_, q_, cache_.k_at(l, 0), cache_.v_at(l, 0), pos + 1, c.n_heads, c.n_kv_heads,
                c.head_dim, scores_);
      linear(proj_, att_, *L.wo, nullptr, q_dim_, d_);
      add_inplace(x_, proj_, d_);

      // SwiGLU MLP block.
      rmsnorm(h_, x_, L.mlp_norm, d_, c.norm_eps);
      linear(gate_, h_, *L.wgate, nullptr, d_, c.d_ff);
      linear(up_, h_, *L.wup, nullptr, d_, c.d_ff);
      silu_mul(gate_, up_, c.d_ff);
      linear(proj_, gate_, *L.wdown, nullptr, c.d_ff, d_);
      add_inplace(x_, proj_, d_);
    }

    rmsnorm(h_, x_, norm_f_, d_, c.norm_eps);
    linear(logits_, h_, *lm_head_, nullptr, d_, c.vocab_size);
    return logits_;
  }

 private:
  struct Layer {
    const float* attn_norm = nullptr;
    const float* mlp_norm = nullptr;
    const Tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const Tensor *wgate = nullptr, *wup = nullptr, *wdown = nullptr;
  };

  static size_t arena_bytes(const Config& c, int max_ctx) {
    const size_t q_dim = static_cast<size_t>(c.n_heads) * c.head_dim;
    const size_t kv_dim = static_cast<size_t>(c.n_kv_heads) * c.head_dim;
    const size_t ctx = static_cast<size_t>(std::max(max_ctx, 1));
    size_t floats = 3 * static_cast<size_t>(c.d_model)          // x, h, proj
                    + 2 * q_dim                                 // q, att
                    + 2 * static_cast<size_t>(c.d_ff)           // gate, up
                    + static_cast<size_t>(c.n_heads) * ctx      // scores
                    + static_cast<size_t>(c.vocab_size)         // logits
                    + 2 * static_cast<size_t>(c.n_layers) * ctx * kv_dim;  // KV cache
    return floats * sizeof(float) + 16 * 64;  // alignment slack
  }

  const Tensor& get(const std::string& name, std::vector<int64_t> shape) const {
    const Tensor& t = wf_->get(name);
    bool ok = t.ndim == static_cast<int32_t>(shape.size());
    for (size_t i = 0; ok && i < shape.size(); ++i) ok = t.shape[i] == shape[i];
    if (!ok) throw std::runtime_error("make_llama: tensor " + name + " has unexpected shape");
    if (shape.size() == 1 && t.dtype != DType::F32)
      throw std::runtime_error("make_llama: tensor " + name + " must be F32");
    return t;
  }

  std::shared_ptr<WeightFile> wf_;
  Arena arena_;
  KVCache cache_;
  int d_ = 0, q_dim_ = 0, kv_dim_ = 0;
  const Tensor* tok_emb_ = nullptr;
  const Tensor* lm_head_ = nullptr;
  const float* norm_f_ = nullptr;
  std::vector<Layer> layers_;
  float *x_ = nullptr, *h_ = nullptr, *q_ = nullptr, *att_ = nullptr, *proj_ = nullptr;
  float *gate_ = nullptr, *up_ = nullptr, *scores_ = nullptr, *logits_ = nullptr;
};

}  // namespace

std::unique_ptr<Model> make_llama(std::shared_ptr<WeightFile> wf, int max_ctx) {
  if (!wf) throw std::invalid_argument("make_llama: null WeightFile");
  return std::make_unique<Llama>(std::move(wf), max_ctx);
}

}  // namespace tinyllm
