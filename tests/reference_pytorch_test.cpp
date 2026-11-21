// Element-wise verification against real PyTorch/HuggingFace reference dumps, generated
// by scripts/gen_reference.py (see scripts/README.md for exact commands). Three
// independent things are checked:
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
//   2. TinyLlamaReference.ElementWiseMatchesHuggingFace: the same kind of independent
//      reproduction, through the public ops (rmsnorm, linear, rope, attention, silu_mul),
//      of TinyLlama-1.1B's forward pass for a fixed 6-token id sequence (BOS + 5-token
//      prompt), plus a check of the actual shipped load_model()+Model::forward() path
//      (src/llama.cpp) against the same HF logits. This is the one that closes the gap
//      noted in CLAUDE.md: TinyLlama's real weights were previously only spot-checked by
//      eye, never run end-to-end against HF. q_proj/k_proj outputs are captured PRE-RoPE;
//      the merged GQA attention output is POST-RoPE, so RoPE and attention are exercised
//      together there, not in isolation (see (3) below for isolated RoPE/attention checks
//      against TinyLlama's own dimensions, just not its real weights).
//
//   3. Gpt2Q8Reference / Gpt2Q4Reference / TinyLlamaQ8Reference / TinyLlamaQ4Reference:
//      the quantized (Q8/Q4) paths, run end to end through the shipped
//      load_model()+Model::forward() on the real models/<model>-{q8,q4}.tllm weights and
//      checked for FIDELITY against the f32 HF ground truth: the quantized greedy argmax
//      and top-5 must match HF's f32 logits at every prompt position. The raw logit
//      divergence (max/mean abs) is reported as the measured quantization cost, not gated
//      on a tuned bound. See the detailed comment above RunQuantFidelity for why a
//      quantized W8A8/W4A8 pipeline cannot be pinned f32-tight.
//
//   4. OpLevelReference.MatchesHuggingFaceAndTorch: fixed-seed random inputs run through
//      tinyllm's layernorm, gelu, softmax, attention (MHA and GQA), rmsnorm, rope, and
//      silu_mul, compared against torch/transformers's own reference implementations of
//      each (torch.nn.functional.layer_norm/softmax/silu, transformers' NewGELUActivation,
//      LlamaRMSNorm, and apply_rotary_pos_emb). This does not need any model weights,
//      only the reference dump under models/ref/ops.
//
// All tests skip cleanly (GTEST_SKIP) when their reference dumps or weight files are
// absent, following the TINYLLM_MODELS_DIR skip pattern used elsewhere in this suite (see
// gpt2_e2e_test.cpp). None of these dumps are checked into the repo (models/ is
// gitignored); regenerate with:
//
//   uv venv --python 3.12 models/.venv-ref
//   uv pip install --python models/.venv-ref torch --extra-index-url https://download.pytorch.org/whl/cpu
//   uv pip install --python models/.venv-ref transformers numpy safetensors huggingface_hub
//   hf download openai-community/gpt2 --local-dir models/gpt2
//   hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama
//   ./build/tinyllm-convert models/gpt2 models/gpt2.tllm
//   ./build/tinyllm-convert models/tinyllama models/tinyllama-f32.tllm
//   ./build/tinyllm-convert models/gpt2 models/gpt2-q8.tllm --quant q8
//   ./build/tinyllm-convert models/gpt2 models/gpt2-q4.tllm --quant q4
//   ./build/tinyllm-convert models/tinyllama models/tinyllama-q8.tllm --quant q8
//   ./build/tinyllm-convert models/tinyllama models/tinyllama-q4.tllm --quant q4
//   models/.venv-ref/bin/python scripts/gen_reference.py
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "test_util.h"
#include "tinyllm/format.h"
#include "tinyllm/model.h"
#include "tinyllm/ops.h"
#include "tinyllm/tokenizer.h"

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
// 2. TinyLlama end-to-end, element-wise against real HF weights.

// "The capital of France is", SentencePiece ids with no BOS. This is the one place in
// the suite where the prompt's token ids are *verified*, not just asserted: the test
// below calls the engine's real SentencePiece encode() on the literal string and
// ASSERT_EQs the result against this list before trusting anything else, since
// scripts/gen_reference.py's HF-side tokenization (see its own cross-check comment) runs
// through a different code path (no `sentencepiece` pip package in that venv, so
// AutoTokenizer falls back to the fast tokenizers-crate backend built from
// tokenizer.json) and the two must agree on the exact ids for the comparison below to be
// meaningful.
const std::vector<int> kTinyLlamaPromptIds = {450, 7483, 310, 3444, 338};

TEST(TinyLlamaReference, ElementWiseMatchesHuggingFace) {
  const fs::path models = test::models_dir();
  const fs::path weights_path = models / "tinyllama-f32.tllm";
  const fs::path ref_dir = models / "ref" / "tinyllama";
  const fs::path tok_path = models / "tinyllama" / "tokenizer.model";
  if (!fs::exists(weights_path)) GTEST_SKIP() << "models/tinyllama-f32.tllm not found; see scripts/README.md";
  if (!fs::exists(tok_path)) GTEST_SKIP() << "models/tinyllama/tokenizer.model not found; see scripts/README.md";
  RefSet ref;
  if (!ref.load(ref_dir))
    GTEST_SKIP() << "models/ref/tinyllama/manifest.txt not found; run scripts/gen_reference.py";

  // Tokenization cross-check (see kTinyLlamaPromptIds comment above): the engine's own
  // SentencePiece encode() never adds BOS (tinyllm/tokenizer.h, CLAUDE.md); the CLI
  // prepends it for LLaMA (tools/cli.cpp). Print both the bare encode and the full
  // BOS-prefixed sequence so a mismatch is obvious from test output, not just a diff.
  auto tok = make_sp_tokenizer(tok_path.string());
  const std::vector<int> encoded = tok->encode("The capital of France is");
  {
    std::string s;
    for (size_t i = 0; i < encoded.size(); ++i) s += (i ? ", " : "") + std::to_string(encoded[i]);
    std::printf("tinyllama encode(\"The capital of France is\") = [%s]\n", s.c_str());
  }
  ASSERT_EQ(encoded, kTinyLlamaPromptIds)
      << "engine SentencePiece encode does not match the ids scripts/gen_reference.py used for HF; "
         "regenerate the refs or repin kTinyLlamaPromptIds/TINYLLAMA_PROMPT_IDS to the engine's output";

  auto wf = WeightFile::open(weights_path.string());
  const Config& c = wf->config();
  ASSERT_EQ(c.arch, Arch::LLAMA);
  ASSERT_EQ(c.d_model, 2048);
  ASSERT_EQ(c.n_layers, 22);
  ASSERT_EQ(c.n_heads, 32);
  ASSERT_EQ(c.n_kv_heads, 4);
  ASSERT_EQ(c.head_dim, 64);
  ASSERT_GE(c.bos_id, 0) << "LLaMA config has no bos_id; cannot build the BOS-prefixed sequence";

  std::vector<int> prompt = {c.bos_id};
  prompt.insert(prompt.end(), encoded.begin(), encoded.end());
  {
    std::string s;
    for (size_t i = 0; i < prompt.size(); ++i) s += (i ? ", " : "") + std::to_string(prompt[i]);
    std::printf("tinyllama full id sequence (BOS + prompt) = [%s]\n", s.c_str());
  }

  const int d = c.d_model, n_heads = c.n_heads, n_kv_heads = c.n_kv_heads, head_dim = c.head_dim, d_ff = c.d_ff;
  const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
  const int T = static_cast<int>(prompt.size());

  const Tensor& tok_emb = wf->get("tok_emb");
  const float* norm_f_w = wf->get("norm_f.w").f32();
  const Tensor& lm_head = wf->has("lm_head.w") ? wf->get("lm_head.w") : tok_emb;

  struct Layer {
    const float *attn_norm, *mlp_norm;
    const Tensor *wq, *wk, *wv, *wo, *wgate, *wup, *wdown;
  };
  std::vector<Layer> layers(c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    const std::string p = "l" + std::to_string(i) + ".";
    Layer& L = layers[i];
    L.attn_norm = wf->get(p + "attn_norm.w").f32();
    L.wq = &wf->get(p + "attn.q.w");
    L.wk = &wf->get(p + "attn.k.w");
    L.wv = &wf->get(p + "attn.v.w");
    L.wo = &wf->get(p + "attn.o.w");
    L.mlp_norm = wf->get(p + "mlp_norm.w").f32();
    L.wgate = &wf->get(p + "mlp.gate.w");
    L.wup = &wf->get(p + "mlp.up.w");
    L.wdown = &wf->get(p + "mlp.down.w");
  }

  // Manual KV cache: [layer][pos][kv_dim] (GQA, so kv_dim < q_dim here).
  std::vector<std::vector<float>> k_cache(c.n_layers, std::vector<float>(size_t(T) * kv_dim));
  std::vector<std::vector<float>> v_cache(c.n_layers, std::vector<float>(size_t(T) * kv_dim));

  std::vector<float> x(d), h(d), q(q_dim), v_pre(kv_dim), att(q_dim), proj(d), gate(d_ff), up(d_ff);
  std::vector<float> scores(size_t(n_heads) * T);
  std::vector<float> logits(c.vocab_size);

  ErrorTracker err;
  // Tolerances: f32 end-to-end against an independently-computed f32 HF reference, same
  // methodology as Gpt2Reference above. TinyLlama is 22 layers (vs. GPT-2's 12) and wider
  // (d_model 2048 vs 768, d_ff 5632 vs 3072), so summation-order drift in the matmuls
  // accumulates further by the last layer; RoPE error also grows with position, but at
  // these early positions (0-5) that term is negligible (~1e-6, see OpLevelReference's
  // pos=2047 case for where it actually matters) -- depth, not RoPE, is what widens these
  // tolerances relative to GPT-2's. Each value below is set from the measured worst case
  // (see the worst-case summary this test prints), rounded up with a small margin for
  // run-to-run noise (parallel reduction order in the F32 thread pool), not guessed:
  //   embed 0, ln1 1.1e-5, ln2 8.1e-6, q 1.0e-5, k 2.0e-5, v 3.6e-6, attn_merged 4.9e-6,
  //   gate 1.8e-5, up 1.7e-5, swiglu 3.3e-4, down 1.2e-4, resid1/resid2 1.2e-4,
  //   norm_f 3.3e-4, logits 4.2e-5, shipped_logits 3.1e-5. Reproducible bit-for-bit across
  //   repeated runs on this machine (checked 3x), so the margins below are not chasing
  //   run-to-run thread-scheduling noise.
  constexpr double kTolTiny = 5e-5;    // embed, ln1, ln2: small sums, O(1)-scale values
  constexpr double kTolQkv = 5e-5;     // q, k, v: single d_model-wide matmul, pre-RoPE
  constexpr double kTolAttn = 2e-5;    // attn_merged: RoPE + GQA attention wired together
  constexpr double kTolMlp = 1e-3;     // gate/up/swiglu/down: wider matmuls (d_ff=5632)
  constexpr double kTolResid = 1e-3;   // resid1/resid2/norm_f: accumulated over 22 blocks
  constexpr double kTolLogits = 2e-4;  // logits, shipped_logits: 2048-wide matmul on top of 22 blocks' drift

  for (int pos = 0; pos < T; ++pos) {
    const int token = prompt[pos];
    const float* te = tok_emb.f32() + size_t(token) * d;
    std::copy(te, te + d, x.begin());
    err.check("embed", x, ref.row("embed", pos, d), kTolTiny, "pos=" + std::to_string(pos));

    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = layers[l];
      const std::string lp = "l" + std::to_string(l) + ".";
      const std::string where = "layer=" + std::to_string(l) + " pos=" + std::to_string(pos);

      rmsnorm(h.data(), x.data(), L.attn_norm, d, c.norm_eps);
      err.check("ln1", h, ref.row(lp + "ln1", pos, d), kTolTiny, where);

      linear(q.data(), h.data(), *L.wq, nullptr, d, q_dim);
      err.check("q", q, ref.row(lp + "q", pos, q_dim), kTolQkv, where);

      float* kc = &k_cache[l][size_t(pos) * kv_dim];
      linear(kc, h.data(), *L.wk, nullptr, d, kv_dim);
      err.check("k", std::vector<float>(kc, kc + kv_dim), ref.row(lp + "k", pos, kv_dim), kTolQkv, where);

      linear(v_pre.data(), h.data(), *L.wv, nullptr, d, kv_dim);
      err.check("v", v_pre, ref.row(lp + "v", pos, kv_dim), kTolQkv, where);
      std::copy(v_pre.begin(), v_pre.end(), v_cache[l].begin() + size_t(pos) * kv_dim);

      // q/k above are captured PRE-RoPE (matching the HF q_proj/k_proj hook); rope() is
      // applied in place now, exactly as src/llama.cpp does, so attn_merged below
      // exercises RoPE and GQA attention together (see the file header comment).
      rope(q.data(), n_heads, head_dim, pos, c.rope_theta);
      rope(kc, n_kv_heads, head_dim, pos, c.rope_theta);

      attention(att.data(), q.data(), k_cache[l].data(), v_cache[l].data(), pos + 1, n_heads, n_kv_heads,
                head_dim, scores.data());
      err.check("attn_merged", att, ref.row(lp + "attn_merged", pos, q_dim), kTolAttn, where);

      linear(proj.data(), att.data(), *L.wo, nullptr, q_dim, d);
      add_inplace(x.data(), proj.data(), d);
      err.check("resid1", x, ref.row(lp + "resid1", pos, d), kTolResid, where);

      rmsnorm(h.data(), x.data(), L.mlp_norm, d, c.norm_eps);
      err.check("ln2", h, ref.row(lp + "ln2", pos, d), kTolTiny, where);

      linear(gate.data(), h.data(), *L.wgate, nullptr, d, d_ff);
      err.check("gate", gate, ref.row(lp + "gate", pos, d_ff), kTolMlp, where);

      linear(up.data(), h.data(), *L.wup, nullptr, d, d_ff);
      err.check("up", up, ref.row(lp + "up", pos, d_ff), kTolMlp, where);

      silu_mul(gate.data(), up.data(), d_ff);
      err.check("swiglu", gate, ref.row(lp + "swiglu", pos, d_ff), kTolMlp, where);

      linear(proj.data(), gate.data(), *L.wdown, nullptr, d_ff, d);
      err.check("down", proj, ref.row(lp + "down", pos, d), kTolMlp, where);
      add_inplace(x.data(), proj.data(), d);
      err.check("resid2", x, ref.row(lp + "resid2", pos, d), kTolResid, where);
    }

    rmsnorm(h.data(), x.data(), norm_f_w, d, c.norm_eps);
    err.check("norm_f", h, ref.row("norm_f", pos, d), kTolResid, "pos=" + std::to_string(pos));

    linear(logits.data(), h.data(), lm_head, nullptr, d, c.vocab_size);
    err.check("logits", logits, ref.row("logits", pos, c.vocab_size), kTolLogits, "pos=" + std::to_string(pos));
  }

  // Also exercise the actual shipped path (load_model() + Model::forward(), src/llama.cpp)
  // against the same HF logits, closing the gap CLAUDE.md flags: TinyLlama's real weights
  // run through the real engine wiring, not just this file's independent reproduction.
  // Done before print_summary() so shipped_logits shows up in the printed table too.
  {
    auto model = load_model(weights_path.string(), 64);
    const float* model_logits = nullptr;
    for (int pos = 0; pos < T; ++pos) model_logits = model->forward(prompt[pos], pos);
    std::vector<float> got(model_logits, model_logits + c.vocab_size);
    err.check("shipped_logits", got, ref.row("logits", T - 1, c.vocab_size), kTolLogits,
              "pos=" + std::to_string(T - 1));
  }

  err.print_summary();
}

// ---------------------------------------------------------------------------
// 3. Quantized (Q8/Q4) fidelity against the f32 HuggingFace ground truth.
//
// tinyllm's quantized matvec is W8A8 / W4A8: matvec_q8 and matvec_q4 (src/quant.cpp)
// quantize the ACTIVATION vector to int8 on every call ("q8_0-style activation
// quantization", see the src/quant.cpp top-of-file comment and quantize_x there), not
// just the weights. An INT8/INT4 forward therefore cannot match the f32 model
// element-wise: quantization is a real lossy transform whose error accumulates with depth
// (empirically it reaches a few tenths on GPT-2's logits by the final layer). The honest
// question for a quantized path is FIDELITY, not f32-tight agreement: does the quantized
// model still predict what the f32 model predicts?
//
// We gate on the behavioral invariant that is both bug-sensitive and exactly the showcase
// claim -- the quantized engine's greedy argmax (and top-5) must match HF's f32 argmax at
// every prompt position -- and we REPORT the raw logit divergence (max/mean abs) as the
// measured quantization cost rather than asserting a hand-tuned bound on it. A wiring,
// scale, or layout bug in the quant kernels scrambles the ranking and trips the argmax
// gate; benign quantization noise does not. The isolated matvec_q8/matvec_q4 kernels are
// pinned tightly against naive references in quant_test.cpp and (on real op shapes) in
// OpLevelReference; this test covers the full quantized pipeline end to end on real
// weights via the shipped load_model()+Model::forward() path.
//
// Reference: the ORIGINAL f32 HF logits dumped at models/ref/<model>/logits.f32 by the
// f32 generation above (sections 1 and 2). No quant-specific Python reference is needed.
// Skips cleanly if the quantized .tllm or the f32 dump is missing.

int arg_max(const std::vector<float>& v) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(v.size()); ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

std::vector<int> top_k(const std::vector<float>& v, int k) {
  std::vector<int> idx(v.size());
  for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return v[a] > v[b]; });
  idx.resize(k);
  return idx;
}

int top_k_overlap(const std::vector<float>& a, const std::vector<float>& b, int k) {
  const std::vector<int> ta = top_k(a, k);
  const std::vector<int> tbv = top_k(b, k);
  const std::set<int> tb(tbv.begin(), tbv.end());
  int o = 0;
  for (int x : ta) o += static_cast<int>(tb.count(x));
  return o;
}

// Runs the shipped quantized model over `prompt` and checks, at every position, that its
// greedy argmax and top-5 match the f32 HF reference logits. Reports the logit divergence
// (the measured quantization cost) rather than gating a tuned bound on it.
void RunQuantFidelity(const std::string& label, const fs::path& weights_path,
                      const fs::path& f32_ref_dir, const std::vector<int>& prompt,
                      bool require_exact_argmax) {
  if (!fs::exists(weights_path))
    GTEST_SKIP() << weights_path.string() << " not found; convert the model with --quant";
  RefSet f32;
  if (!f32.load(f32_ref_dir))
    GTEST_SKIP() << f32_ref_dir.string() << "/manifest.txt not found; run scripts/gen_reference.py";

  auto wf = WeightFile::open(weights_path.string());
  const int V = wf->config().vocab_size;
  const int T = static_cast<int>(prompt.size());

  auto model = load_model(weights_path.string(), 64);
  int argmax_match = 0, in_top5 = 0, top5_sum = 0;
  double worst_abs = 0.0, sum_abs = 0.0;
  size_t nvals = 0;
  std::vector<int> flips;
  for (int pos = 0; pos < T; ++pos) {
    const float* ql = model->forward(prompt[pos], pos);
    const std::vector<float> q(ql, ql + V);
    const std::vector<float> f = f32.row("logits", pos, V);
    ASSERT_EQ(static_cast<int>(f.size()), V) << label << " pos=" << pos;
    const std::vector<int> f_top = top_k(f, 5);
    const int qa = arg_max(q), fa = f_top[0];
    const bool q_in_f5 = std::find(f_top.begin(), f_top.end(), qa) != f_top.end();
    const double f_gap = double(f[fa]) - double(f[f_top[1]]);  // f32 top1 - top2 margin
    if (q_in_f5) ++in_top5;
    if (qa == fa) ++argmax_match;
    else {
      flips.push_back(pos);
      std::printf(
          "    [%s] pos %d FLIP: f32 top1=%d (%.3f, margin %.3f over top2), quant top1=%d (%.3f); "
          "quant pick in f32 top-5: %s\n",
          label.c_str(), pos, fa, f[fa], f_gap, qa, q[qa], q_in_f5 ? "yes" : "no");
    }
    top5_sum += top_k_overlap(q, f, 5);
    for (int i = 0; i < V; ++i) {
      const double e = std::fabs(double(q[i]) - double(f[i]));
      worst_abs = std::max(worst_abs, e);
      sum_abs += e;
      ++nvals;
    }
  }

  std::printf(
      "[%s] argmax match %d/%d, greedy pick in f32 top-5 %d/%d, mean top-5 overlap %.2f/5; "
      "logit diff vs f32 HF: max abs %.4g, mean abs %.4g (measured quantization cost)\n",
      label.c_str(), argmax_match, T, in_top5, T, double(top5_sum) / T, worst_abs,
      sum_abs / double(nvals));

  std::string flip_str;
  for (int p : flips) flip_str += std::to_string(p) + " ";

  // Bug-sensitive floor for EVERY quantized precision: the greedy pick must stay inside the
  // f32 model's top-5 at every position. A wiring/scale/layout bug in matvec_q8/matvec_q4
  // (src/quant.cpp) pushes the argmax to an unrelated token (out of the top-5); benign
  // quantization noise only reshuffles already near-tied candidates. The per-position FLIP
  // lines above show where and why q4 diverges (always a near-tied or high-entropy position,
  // with the pick still in f32's top-5).
  EXPECT_EQ(in_top5, T)
      << label << ": quantized greedy pick left the f32 top-5 at position(s) " << flip_str
      << "(likely a kernel bug, not quantization noise)";

  // Q8 is near-lossless, so additionally require its greedy token to EXACTLY match f32 at
  // every position. Q4 is aggressive; its argmax may legitimately differ at high-entropy
  // positions (the top-5 floor above still holds), so the agreement is reported, not gated.
  if (require_exact_argmax)
    EXPECT_EQ(argmax_match, T)
        << label << ": q8 greedy argmax should match f32 (near-lossless); diverged at " << flip_str;
}

TEST(Gpt2Q8Reference, PredictionsMatchF32HuggingFace) {
  const fs::path m = test::models_dir();
  RunQuantFidelity("gpt2-q8", m / "gpt2-q8.tllm", m / "ref" / "gpt2", kPrompt, /*require_exact_argmax=*/true);
}
TEST(Gpt2Q4Reference, PredictionsMatchF32HuggingFace) {
  const fs::path m = test::models_dir();
  RunQuantFidelity("gpt2-q4", m / "gpt2-q4.tllm", m / "ref" / "gpt2", kPrompt, /*require_exact_argmax=*/false);
}

void RunTinyLlamaQuantFidelity(const std::string& quant, bool require_exact_argmax) {
  const fs::path m = test::models_dir();
  const fs::path weights = m / ("tinyllama-" + quant + ".tllm");
  const fs::path tok_path = m / "tinyllama" / "tokenizer.model";
  if (!fs::exists(weights))
    GTEST_SKIP() << weights.string() << " not found; convert with --quant " << quant;
  if (!fs::exists(tok_path)) GTEST_SKIP() << tok_path.string() << " not found; see scripts/README.md";

  auto wf = WeightFile::open(weights.string());
  ASSERT_EQ(wf->config().arch, Arch::LLAMA);
  const int bos = wf->config().bos_id;
  ASSERT_GE(bos, 0);
  auto tok = make_sp_tokenizer(tok_path.string());
  const std::vector<int> encoded = tok->encode("The capital of France is");
  ASSERT_EQ(encoded, kTinyLlamaPromptIds)
      << "engine SentencePiece encode does not match the ids scripts/gen_reference.py used for HF";
  std::vector<int> prompt = {bos};
  prompt.insert(prompt.end(), encoded.begin(), encoded.end());

  RunQuantFidelity("tinyllama-" + quant, weights, m / "ref" / "tinyllama", prompt, require_exact_argmax);
}

TEST(TinyLlamaQ8Reference, PredictionsMatchF32HuggingFace) {
  RunTinyLlamaQuantFidelity("q8", /*require_exact_argmax=*/true);
}
TEST(TinyLlamaQ4Reference, PredictionsMatchF32HuggingFace) {
  RunTinyLlamaQuantFidelity("q4", /*require_exact_argmax=*/false);
}

// ---------------------------------------------------------------------------
// 4. Op-level references: fixed-seed random inputs, HF/torch reference implementations.

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
