// Element-wise verification against real PyTorch/HuggingFace reference dumps, generated
// by scripts/gen_reference.py (see scripts/README.md for exact commands). Two independent
// things are checked:
//
//   1. Gpt2Reference.ElementWiseMatchesHuggingFace: reproduces the GPT-2 small forward
//      pass for a fixed 5-token prompt using ONLY the public ops in tinyllm/ops.h and
//      weights loaded via WeightFile, one token at a time (mirroring src/gpt2.cpp, which
//      this file deliberately does not call into -- the point is an independent
//      reproduction through the public surface, not a call to Model::forward). Every
//      intermediate -- embeddings, each block's ln1/qkv/attention/residual/ln2/mlp/
//      residual, the final ln_f, and the logits -- is compared element-wise against HF's
//      real GPT-2 weights run through transformers (attn_implementation="eager").
//
//   2. OpLevelReference.MatchesHuggingFaceAndTorch: fixed-seed random inputs run through
//      tinyllm's layernorm, gelu, softmax, attention (MHA and GQA), rmsnorm, rope, and
//      silu_mul, compared against torch/transformers's own reference implementations of
//      each (torch.nn.functional.layer_norm/softmax/silu, transformers' NewGELUActivation,
//      LlamaRMSNorm, and apply_rotary_pos_emb). This does not need any model weights,
//      only the reference dump under models/ref/ops.
//
// Both halves skip cleanly (GTEST_SKIP) when their reference dumps are absent, following
// the TINYLLM_MODELS_DIR skip pattern used elsewhere in this suite (see gpt2_e2e_test.cpp).
// Neither dump is checked into the repo (models/ is gitignored); regenerate with:
//
//   uv venv --python 3.12 models/.venv-ref
//   uv pip install --python models/.venv-ref torch --extra-index-url https://download.pytorch.org/whl/cpu
//   uv pip install --python models/.venv-ref transformers numpy safetensors huggingface_hub
//   hf download openai-community/gpt2 --local-dir models/gpt2
//   ./build/tinyllm-convert models/gpt2 models/gpt2.tllm
//   models/.venv-ref/bin/python scripts/gen_reference.py
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "test_util.h"
#include "tinyllm/format.h"
#include "tinyllm/model.h"
#include "tinyllm/ops.h"

namespace tinyllm {
namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Reference dump reader: "<name> <dim0> [dim1 ...]" manifest + flat float32 "<name>.f32"
// binaries, written by scripts/gen_reference.py. See that script's module docstring for
// the exact layout.

class RefSet {
 public:
  bool load(const fs::path& dir) {
    dir_ = dir;
    std::ifstream f(dir / "manifest.txt");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      std::string name;
      ss >> name;
      std::vector<int64_t> shape;
      int64_t d;
      while (ss >> d) shape.push_back(d);
      shapes_[name] = shape;
    }
    return !shapes_.empty();
  }

  // Numel from the manifest (sanity: matches the caller's expectation).
  int64_t numel(const std::string& name) const {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) throw std::runtime_error("RefSet: no such tensor '" + name + "'");
    int64_t n = 1;
    for (int64_t d : it->second) n *= d;
    return n;
  }

  // Reads the whole tensor.
  std::vector<float> get(const std::string& name) const {
    const int64_t n = numel(name);
    std::ifstream f(dir_ / (name + ".f32"), std::ios::binary);
    if (!f) throw std::runtime_error("RefSet: cannot open " + name + ".f32");
    std::vector<float> v(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!f || f.gcount() != static_cast<std::streamsize>(v.size() * sizeof(float)))
      throw std::runtime_error("RefSet: short read on " + name + ".f32");
    return v;
  }

  // Reads one row of a [rows, cols] tensor (0-based).
  std::vector<float> row(const std::string& name, int64_t r, int64_t cols) const {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) throw std::runtime_error("RefSet: no such tensor '" + name + "'");
    if (it->second.size() != 2 || it->second[1] != cols)
      throw std::runtime_error("RefSet: '" + name + "' is not a [rows, " + std::to_string(cols) + "] tensor");
    std::ifstream f(dir_ / (name + ".f32"), std::ios::binary);
    if (!f) throw std::runtime_error("RefSet: cannot open " + name + ".f32");
    f.seekg(r * cols * static_cast<int64_t>(sizeof(float)));
    std::vector<float> v(static_cast<size_t>(cols));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!f || f.gcount() != static_cast<std::streamsize>(v.size() * sizeof(float)))
      throw std::runtime_error("RefSet: short read on row " + std::to_string(r) + " of " + name);
    return v;
  }

 private:
  fs::path dir_;
  std::map<std::string, std::vector<int64_t>> shapes_;
};

// Tracks the worst-case |error| seen per named category, and asserts it against a
// tolerance at each call site (so a failure points at the exact category/position).
class ErrorTracker {
 public:
  void check(const std::string& category, const std::vector<float>& got, const std::vector<float>& ref,
             double tol, const std::string& where) {
    ASSERT_EQ(got.size(), ref.size()) << category << " " << where;
    double m = 0.0;
    size_t argmax = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double e = std::fabs(double(got[i]) - double(ref[i]));
      if (e > m) { m = e; argmax = i; }
    }
    double& w = worst_[category];
    w = std::max(w, m);
    EXPECT_LE(m, tol) << category << " " << where << ": max abs err " << m << " at index " << argmax
                       << " (tol " << tol << ", got " << got[argmax] << " vs ref " << ref[argmax] << ")";
  }

  void print_summary() const {
    std::printf("\n--- reference_pytorch_test worst-case element-wise error per category ---\n");
    for (const auto& [name, err] : worst_) std::printf("  %-14s max abs err = %.6g\n", name.c_str(), err);
    std::printf("---------------------------------------------------------------------------\n");
  }

 private:
  std::map<std::string, double> worst_;
};

// ---------------------------------------------------------------------------
// 1. GPT-2 end-to-end, element-wise against real HF weights.

const std::vector<int> kPrompt = {464, 3139, 286, 4881, 318};  // "The capital of France is"

TEST(Gpt2Reference, ElementWiseMatchesHuggingFace) {
  const fs::path models = test::models_dir();
  const fs::path weights_path = models / "gpt2.tllm";
  const fs::path ref_dir = models / "ref" / "gpt2";
  if (!fs::exists(weights_path)) GTEST_SKIP() << "models/gpt2.tllm not found; see scripts/README.md";
  RefSet ref;
  if (!ref.load(ref_dir)) GTEST_SKIP() << "models/ref/gpt2/manifest.txt not found; run scripts/gen_reference.py";

  auto wf = WeightFile::open(weights_path.string());
  const Config& c = wf->config();
  ASSERT_EQ(c.arch, Arch::GPT2);
  const int d = c.d_model, n_heads = c.n_heads, head_dim = c.head_dim, d_ff = c.d_ff;
  const int T = static_cast<int>(kPrompt.size());
  ASSERT_EQ(d, 768);
  ASSERT_EQ(c.n_layers, 12);

  const Tensor& tok_emb = wf->get("tok_emb");
  const Tensor& pos_emb = wf->get("pos_emb");
  const float* ln_f_w = wf->get("ln_f.w").f32();
  const float* ln_f_b = wf->get("ln_f.b").f32();

  struct Layer {
    const float *ln1_w, *ln1_b, *qkv_b, *o_b, *ln2_w, *ln2_b, *up_b, *down_b;
    const Tensor *qkv_w, *o_w, *up_w, *down_w;
  };
  std::vector<Layer> layers(c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    const std::string p = "l" + std::to_string(i) + ".";
    Layer& L = layers[i];
    L.ln1_w = wf->get(p + "ln1.w").f32();
    L.ln1_b = wf->get(p + "ln1.b").f32();
    L.qkv_w = &wf->get(p + "attn.qkv.w");
    L.qkv_b = wf->get(p + "attn.qkv.b").f32();
    L.o_w = &wf->get(p + "attn.o.w");
    L.o_b = wf->get(p + "attn.o.b").f32();
    L.ln2_w = wf->get(p + "ln2.w").f32();
    L.ln2_b = wf->get(p + "ln2.b").f32();
    L.up_w = &wf->get(p + "mlp.up.w");
    L.up_b = wf->get(p + "mlp.up.b").f32();
    L.down_w = &wf->get(p + "mlp.down.w");
    L.down_b = wf->get(p + "mlp.down.b").f32();
  }

  // Manual KV cache: [layer][pos][d] (GPT-2 is MHA, so kv_dim == d).
  std::vector<std::vector<float>> k_cache(c.n_layers, std::vector<float>(size_t(T) * d));
  std::vector<std::vector<float>> v_cache(c.n_layers, std::vector<float>(size_t(T) * d));

  std::vector<float> x(d), h(d), qkv(3 * size_t(d)), attn_merged(d), tmp(d), ff(d_ff);
  std::vector<float> scores(size_t(n_heads) * T);
  std::vector<float> logits(c.vocab_size);

  ErrorTracker err;
  // Tolerances: f32 end-to-end against an independently-computed f32 HF reference. Early,
  // low-dimensional ops (embeddings, a single layernorm) match to within float rounding;
  // later ops accumulate more summation-order divergence through wider matmuls and more
  // residual additions, so tolerances widen with depth. See the report for the actual
  // worst case measured per category (printed below, and surfaced as EXPECT_LE messages).
  constexpr double kTolTiny = 1e-4;    // embed, ln1, ln2: small sums, O(1)-scale values
  constexpr double kTolMed = 5e-4;     // qkv, attn_merged, resid1, mlp_fc, mlp_gelu
  constexpr double kTolResid = 1e-3;   // resid2, ln_f: accumulated over the whole block
  constexpr double kTolLogits = 1e-3;  // logits: a 768-wide matmul on top of 12 blocks' drift

  for (int pos = 0; pos < T; ++pos) {
    const int token = kPrompt[pos];
    const float* te = tok_emb.f32() + size_t(token) * d;
    const float* pe = pos_emb.f32() + size_t(pos) * d;
    for (int i = 0; i < d; ++i) x[i] = te[i] + pe[i];
    err.check("embed", x, ref.row("embed", pos, d), kTolTiny, "pos=" + std::to_string(pos));

    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = layers[l];
      const std::string where = "layer=" + std::to_string(l) + " pos=" + std::to_string(pos);

      layernorm(h.data(), x.data(), L.ln1_w, L.ln1_b, d, c.norm_eps);
      err.check("ln1", h, ref.row("l" + std::to_string(l) + ".ln1", pos, d), kTolTiny, where);

      linear(qkv.data(), h.data(), *L.qkv_w, L.qkv_b, d, 3 * d);
      err.check("qkv", qkv, ref.row("l" + std::to_string(l) + ".qkv", pos, 3 * d), kTolMed, where);

      std::copy(qkv.begin() + d, qkv.begin() + 2 * d, k_cache[l].begin() + size_t(pos) * d);
      std::copy(qkv.begin() + 2 * d, qkv.begin() + 3 * d, v_cache[l].begin() + size_t(pos) * d);
      attention(attn_merged.data(), qkv.data(), k_cache[l].data(), v_cache[l].data(), pos + 1, n_heads,
                n_heads, head_dim, scores.data());
      err.check("attn_merged", attn_merged, ref.row("l" + std::to_string(l) + ".attn_merged", pos, d),
                kTolMed, where);

      linear(tmp.data(), attn_merged.data(), *L.o_w, L.o_b, d, d);
      add_inplace(x.data(), tmp.data(), d);
      err.check("resid1", x, ref.row("l" + std::to_string(l) + ".resid1", pos, d), kTolResid, where);

      layernorm(h.data(), x.data(), L.ln2_w, L.ln2_b, d, c.norm_eps);
      err.check("ln2", h, ref.row("l" + std::to_string(l) + ".ln2", pos, d), kTolTiny, where);

      linear(ff.data(), h.data(), *L.up_w, L.up_b, d, d_ff);
      err.check("mlp_fc", ff, ref.row("l" + std::to_string(l) + ".mlp_fc", pos, d_ff), kTolMed, where);

      gelu(ff.data(), d_ff);
      err.check("mlp_gelu", ff, ref.row("l" + std::to_string(l) + ".mlp_gelu", pos, d_ff), kTolMed, where);

      linear(tmp.data(), ff.data(), *L.down_w, L.down_b, d_ff, d);
      add_inplace(x.data(), tmp.data(), d);
      err.check("resid2", x, ref.row("l" + std::to_string(l) + ".resid2", pos, d), kTolResid, where);
    }

    layernorm(h.data(), x.data(), ln_f_w, ln_f_b, d, c.norm_eps);
    err.check("ln_f", h, ref.row("ln_f", pos, d), kTolResid, "pos=" + std::to_string(pos));

    linear(logits.data(), h.data(), tok_emb, nullptr, d, c.vocab_size);
    err.check("logits", logits, ref.row("logits", pos, c.vocab_size), kTolLogits, "pos=" + std::to_string(pos));
  }

  err.print_summary();

  // Also exercise the actual shipped path (load_model() + Model::forward(), src/gpt2.cpp)
  // against the same HF logits, so this test covers not just "these ops match HF" but
  // "the real engine's wiring of those ops matches HF" -- closing the gap the existing
  // gpt2_e2e_test.cpp leaves (it only pins tokens generated by this engine itself).
  {
    auto model = load_model(weights_path.string(), 64);
    const float* model_logits = nullptr;
    for (int pos = 0; pos < T; ++pos) model_logits = model->forward(kPrompt[pos], pos);
    std::vector<float> got(model_logits, model_logits + c.vocab_size);
    err.check("shipped_logits", got, ref.row("logits", T - 1, c.vocab_size), kTolLogits, "pos=" + std::to_string(T - 1));
  }
}

// ---------------------------------------------------------------------------
// 2. Op-level references: fixed-seed random inputs, HF/torch reference implementations.

TEST(OpLevelReference, MatchesHuggingFaceAndTorch) {
  const fs::path ref_dir = fs::path(test::models_dir()) / "ref" / "ops";
  RefSet ref;
  if (!ref.load(ref_dir)) GTEST_SKIP() << "models/ref/ops/manifest.txt not found; run scripts/gen_reference.py";

  ErrorTracker err;

  // LayerNorm: torch.nn.functional.layer_norm, eps = 1e-5 (see gen_reference.py).
  {
    auto x = ref.get("layernorm.x"), g = ref.get("layernorm.g"), b = ref.get("layernorm.b");
    const int D = int(x.size());
    std::vector<float> y(D);
    layernorm(y.data(), x.data(), g.data(), b.data(), D, 1e-5f);
    err.check("layernorm", y, ref.get("layernorm.y"), 2e-5, "D=" + std::to_string(D));
  }

  // GELU ("gelu_new"): transformers.activations.NewGELUActivation.
  {
    auto x = ref.get("gelu.x");
    auto y = x;
    gelu(y.data(), int64_t(y.size()));
    err.check("gelu", y, ref.get("gelu.y"), 1e-5, "n=" + std::to_string(y.size()));
  }

  // Softmax: torch.nn.functional.softmax, two widths. Wider sums accumulate more
  // summation-order divergence between the two independent float32 implementations.
  for (int n : {50, 2000}) {
    const std::string tag = "softmax_" + std::to_string(n);
    auto x = ref.get(tag + ".x");
    softmax(x.data(), n);
    err.check("softmax", x, ref.get(tag + ".y"), 1e-5, "n=" + std::to_string(n));
  }

  // Attention: manual single-query-over-causal-cache torch reference. MHA (GPT-2 dims)
  // and GQA (TinyLlama dims); see ref_attention() in gen_reference.py.
  {
    auto q = ref.get("attn_mha.q"), k = ref.get("attn_mha.k"), v = ref.get("attn_mha.v");
    const int H = 12, KVH = 12, hd = 64, n_pos = 5;
    std::vector<float> out(size_t(H) * hd), scores(size_t(H) * n_pos);
    attention(out.data(), q.data(), k.data(), v.data(), n_pos, H, KVH, hd, scores.data());
    err.check("attn_mha", out, ref.get("attn_mha.out"), 1e-5, "H=12 hd=64 n_pos=5");
  }
  {
    auto q = ref.get("attn_gqa.q"), k = ref.get("attn_gqa.k"), v = ref.get("attn_gqa.v");
    const int H = 32, KVH = 4, hd = 64, n_pos = 7;
    std::vector<float> out(size_t(H) * hd), scores(size_t(H) * n_pos);
    attention(out.data(), q.data(), k.data(), v.data(), n_pos, H, KVH, hd, scores.data());
    err.check("attn_gqa", out, ref.get("attn_gqa.out"), 1e-5, "H=32 KVH=4 hd=64 n_pos=7");
  }

  // RMSNorm: transformers.models.llama.modeling_llama.LlamaRMSNorm, eps = 1e-5.
  {
    auto x = ref.get("rmsnorm.x"), g = ref.get("rmsnorm.g");
    const int D = int(x.size());
    std::vector<float> y(D);
    rmsnorm(y.data(), x.data(), g.data(), D, 1e-5f);
    err.check("rmsnorm", y, ref.get("rmsnorm.y"), 1e-4, "D=" + std::to_string(D));
  }

  // RoPE: transformers' apply_rotary_pos_emb (HF rotate_half pairing), theta = 10000.
  // Both tinyllm and HF compute the rotation angle pos * theta^(-2i/hd) in float32, so
  // error grows with pos (same falloff documented for Rope.MatchesHfRotateHalf in
  // ops_test.cpp, which uses the identical tolerance formula against its own reference).
  for (int pos : {5, 2047}) {
    const std::string tag = "rope_p" + std::to_string(pos);
    auto x = ref.get(tag + ".x");
    const int H = 32, hd = 64;
    rope(x.data(), H, hd, pos, 10000.0f);
    const double tol = 1e-5 + 4e-7 * pos;
    err.check("rope", x, ref.get(tag + ".y"), tol, "pos=" + std::to_string(pos));
  }

  // SwiGLU gate: torch.nn.functional.silu(gate) * up.
  {
    auto gate = ref.get("siluMul.gate"), up = ref.get("siluMul.up");
    silu_mul(gate.data(), up.data(), int64_t(gate.size()));
    err.check("silu_mul", gate, ref.get("siluMul.y"), 1e-4, "n=" + std::to_string(gate.size()));
  }

  err.print_summary();
}

}  // namespace
}  // namespace tinyllm
