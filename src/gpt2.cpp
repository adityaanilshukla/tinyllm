// GPT-2 forward pass (single-token decoding with KV cache).
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyllm/arena.h"
#include "tinyllm/model.h"
#include "tinyllm/ops.h"

namespace tinyllm {
namespace {

class GPT2 final : public Model {
 public:
  GPT2(std::shared_ptr<WeightFile> wf, int max_ctx) : wf_(std::move(wf)) {
    config_ = wf_->config();
    const Config& c = config_;
    if (c.arch != Arch::GPT2) throw std::runtime_error("make_gpt2: file is not a GPT-2 model");
    if (c.d_model <= 0 || c.n_layers <= 0 || c.n_heads <= 0 || c.vocab_size <= 0)
      throw std::runtime_error("make_gpt2: invalid config");
    max_ctx_ = (max_ctx <= 0 || max_ctx > c.max_ctx) ? c.max_ctx : max_ctx;
    n_kv_heads_ = c.n_kv_heads > 0 ? c.n_kv_heads : c.n_heads;
    head_dim_ = c.head_dim > 0 ? c.head_dim : c.d_model / c.n_heads;
    d_ff_ = c.d_ff > 0 ? c.d_ff : 4 * c.d_model;
    if (n_kv_heads_ * head_dim_ != c.d_model || c.n_heads * head_dim_ != c.d_model)
      throw std::runtime_error("make_gpt2: head dims do not match d_model");

    const int d = c.d_model;
    tok_emb_ = &wf_->get("tok_emb");
    pos_emb_ = &wf_->get("pos_emb");
    if (tok_emb_->dtype != DType::F32 || pos_emb_->dtype != DType::F32)
      throw std::runtime_error("make_gpt2: embeddings must be F32");
    if (tok_emb_->shape[0] != c.vocab_size || tok_emb_->shape[1] != d)
      throw std::runtime_error("make_gpt2: tok_emb shape mismatch");
    if (pos_emb_->shape[0] < max_ctx_ || pos_emb_->shape[1] != d)
      throw std::runtime_error("make_gpt2: pos_emb shape mismatch");
    ln_f_w_ = wf_->get("ln_f.w").f32();
    ln_f_b_ = wf_->get("ln_f.b").f32();

    layers_.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; ++i) {
      const std::string p = "l" + std::to_string(i) + ".";
      Layer& L = layers_[i];
      L.ln1_w = wf_->get(p + "ln1.w").f32();
      L.ln1_b = wf_->get(p + "ln1.b").f32();
      L.qkv_w = &wf_->get(p + "attn.qkv.w");
      L.qkv_b = wf_->get(p + "attn.qkv.b").f32();
      L.o_w = &wf_->get(p + "attn.o.w");
      L.o_b = wf_->get(p + "attn.o.b").f32();
      L.ln2_w = wf_->get(p + "ln2.w").f32();
      L.ln2_b = wf_->get(p + "ln2.b").f32();
      L.up_w = &wf_->get(p + "mlp.up.w");
      L.up_b = wf_->get(p + "mlp.up.b").f32();
      L.down_w = &wf_->get(p + "mlp.down.w");
      L.down_b = wf_->get(p + "mlp.down.b").f32();
    }

    // One arena for activations, scratch and the KV cache.
    auto padded = [](size_t n) { return ((n * sizeof(float) + 63) / 64) * 64; };
    const size_t kv_n = static_cast<size_t>(c.n_layers) * max_ctx_ * d;
    size_t bytes = 2 * padded(kv_n);
    bytes += padded(d) * 4;                                        // x, h, attn, tmp
    bytes += padded(3 * static_cast<size_t>(d));                   // qkv
    bytes += padded(d_ff_);                                        // ff
    bytes += padded(static_cast<size_t>(c.n_heads) * max_ctx_);    // scores
    bytes += padded(c.vocab_size);                                 // logits
    bytes += 64 * 16;                                              // alignment slack
    arena_ = std::make_unique<Arena>(bytes);

    kv_.init(*arena_, c.n_layers, max_ctx_, n_kv_heads_ * head_dim_);
    x_ = arena_->alloc_n<float>(d);
    h_ = arena_->alloc_n<float>(d);
    attn_ = arena_->alloc_n<float>(d);
    tmp_ = arena_->alloc_n<float>(d);
    qkv_ = arena_->alloc_n<float>(3 * static_cast<size_t>(d));
    ff_ = arena_->alloc_n<float>(d_ff_);
    scores_ = arena_->alloc_n<float>(static_cast<size_t>(c.n_heads) * max_ctx_);
    logits_ = arena_->alloc_n<float>(c.vocab_size);
  }

  const float* forward(int token, int pos) override {
    const Config& c = config_;
    const int d = c.d_model;
    if (pos < 0 || pos >= max_ctx_)
      throw std::out_of_range("gpt2: position " + std::to_string(pos) + " out of range (max_ctx " +
                              std::to_string(max_ctx_) + ")");
    if (token < 0 || token >= c.vocab_size) throw std::out_of_range("gpt2: token id out of range");

    const float* te = tok_emb_->f32() + static_cast<size_t>(token) * d;
    const float* pe = pos_emb_->f32() + static_cast<size_t>(pos) * d;
    for (int i = 0; i < d; ++i) x_[i] = te[i] + pe[i];

    const int kv_dim = n_kv_heads_ * head_dim_;
    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = layers_[l];
      // Attention block.
      layernorm(h_, x_, L.ln1_w, L.ln1_b, d, c.norm_eps);
      linear(qkv_, h_, *L.qkv_w, L.qkv_b, d, 3 * d);
      const float* q = qkv_;
      std::memcpy(kv_.k_at(l, pos), qkv_ + d, sizeof(float) * kv_dim);
      std::memcpy(kv_.v_at(l, pos), qkv_ + 2 * d, sizeof(float) * kv_dim);
      attention(attn_, q, kv_.k_at(l, 0), kv_.v_at(l, 0), pos + 1, c.n_heads, n_kv_heads_, head_dim_,
                scores_);
      linear(tmp_, attn_, *L.o_w, L.o_b, d, d);
      add_inplace(x_, tmp_, d);

      // MLP block.
      layernorm(h_, x_, L.ln2_w, L.ln2_b, d, c.norm_eps);
      linear(ff_, h_, *L.up_w, L.up_b, d, d_ff_);
      gelu(ff_, d_ff_);
      linear(tmp_, ff_, *L.down_w, L.down_b, d_ff_, d);
      add_inplace(x_, tmp_, d);
    }

    layernorm(h_, x_, ln_f_w_, ln_f_b_, d, c.norm_eps);
    linear(logits_, h_, *tok_emb_, nullptr, d, c.vocab_size);  // tied LM head
    return logits_;
  }

 private:
  struct Layer {
    const float *ln1_w, *ln1_b, *qkv_b, *o_b, *ln2_w, *ln2_b, *up_b, *down_b;
    const Tensor *qkv_w, *o_w, *up_w, *down_w;
  };

  std::shared_ptr<WeightFile> wf_;
  int n_kv_heads_ = 0, head_dim_ = 0, d_ff_ = 0;
  const Tensor* tok_emb_ = nullptr;
  const Tensor* pos_emb_ = nullptr;
  const float* ln_f_w_ = nullptr;
  const float* ln_f_b_ = nullptr;
  std::vector<Layer> layers_;

  std::unique_ptr<Arena> arena_;
  KVCache kv_;
  float *x_ = nullptr, *h_ = nullptr, *attn_ = nullptr, *tmp_ = nullptr, *qkv_ = nullptr, *ff_ = nullptr,
        *scores_ = nullptr, *logits_ = nullptr;
};

}  // namespace

std::unique_ptr<Model> make_gpt2(std::shared_ptr<WeightFile> wf, int max_ctx) {
  if (!wf) throw std::invalid_argument("make_gpt2: null WeightFile");
  return std::make_unique<GPT2>(std::move(wf), max_ctx);
}

}  // namespace tinyllm
