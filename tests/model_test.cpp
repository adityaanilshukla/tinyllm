// Full forward passes of tiny random GPT-2 and LLaMA models, written to .tllm and run
// through load_model(), against naive double-precision reference implementations that
// follow the HF modeling code. Needs no model files.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.h"
#include "tinyllm/model.h"

namespace tinyllm {
namespace {

using test::OutTensor;
using Vec = std::vector<double>;
using Weights = std::map<std::string, std::vector<float>>;  // dequantized view of what was written

Vec matvec(const std::vector<float>& W, const Vec& x, size_t N) {
  const size_t K = x.size();
  Vec y(N, 0.0);
  for (size_t r = 0; r < N; ++r)
    for (size_t k = 0; k < K; ++k) y[r] += double(W[r * K + k]) * x[k];
  return y;
}

void add_to(Vec& y, const std::vector<float>& b) {
  for (size_t i = 0; i < y.size(); ++i) y[i] += b[i];
}

Vec layernorm(const Vec& x, const std::vector<float>& g, const std::vector<float>& b, double eps) {
  double mean = 0.0, var = 0.0;
  for (double v : x) mean += v;
  mean /= x.size();
  for (double v : x) var += (v - mean) * (v - mean);
  var /= x.size();
  Vec y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = (x[i] - mean) / std::sqrt(var + eps) * g[i] + b[i];
  return y;
}

Vec rmsnorm(const Vec& x, const std::vector<float>& g, double eps) {
  double ss = 0.0;
  for (double v : x) ss += v * v;
  const double inv = 1.0 / std::sqrt(ss / x.size() + eps);
  Vec y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * g[i];
  return y;
}

// Causal attention of one query against cached rows [0, n_pos).
Vec attend(const Vec& q, const std::vector<Vec>& K, const std::vector<Vec>& V, int H, int KVH, int hd) {
  Vec out(size_t(H) * hd, 0.0);
  for (int h = 0; h < H; ++h) {
    const int kvh = h / (H / KVH);
    Vec s(K.size());
    double mx = -1e300, z = 0.0;
    for (size_t t = 0; t < K.size(); ++t) {
      double d = 0.0;
      for (int i = 0; i < hd; ++i) d += q[h * hd + i] * K[t][kvh * hd + i];
      s[t] = d / std::sqrt(double(hd));
      mx = std::max(mx, s[t]);
    }
    for (double& v : s) z += (v = std::exp(v - mx));
    for (size_t t = 0; t < K.size(); ++t)
      for (int i = 0; i < hd; ++i) out[h * hd + i] += s[t] / z * V[t][kvh * hd + i];
  }
  return out;
}

// HF apply_rotary_pos_emb with rotate_half.
void rope(Vec& x, int H, int hd, int pos, double theta) {
  const int half = hd / 2;
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < half; ++i) {
      const double ang = pos * std::pow(theta, -2.0 * i / hd);
      double& a = x[h * hd + i];
      double& b = x[h * hd + i + half];
      const double a0 = a, b0 = b;
      a = a0 * std::cos(ang) - b0 * std::sin(ang);
      b = b0 * std::cos(ang) + a0 * std::sin(ang);
    }
}

// Writes `t` to the list and records its dequantized values for the reference.
void put(std::vector<OutTensor>& out, Weights& w, OutTensor t) {
  w[t.name] = test::dequant(t);
  out.push_back(std::move(t));
}

// Linear weight with entries uniform in [-scale, scale] / sqrt(cols), so activations stay O(1).
void put_linear(std::vector<OutTensor>& out, Weights& w, const std::string& name, DType dt, int rows, int cols,
                uint32_t seed, float scale = 1.0f) {
  auto v = test::rand_vec(size_t(rows) * cols, seed, -scale, scale);
  for (float& x : v) x /= std::sqrt(float(cols));
  put(out, w, test::quant_tensor(name, dt, rows, cols, v));
}

void put_vec(std::vector<OutTensor>& out, Weights& w, const std::string& name, int n, uint32_t seed, float lo,
             float hi) {
  put(out, w, test::f32_tensor(name, {n}, test::rand_vec(n, seed, lo, hi)));
}

std::string model_path(const std::string& name) { return (test::tmp_dir() / (name + ".tllm")).string(); }

double max_abs(const float* a, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(double(a[i])));
  return m;
}

// ---------------------------------------------------------------------------
// GPT-2: d = 64, 4 heads, 2 layers, vocab 97 (odd on purpose), ctx 16.

Config gpt2_config() {
  Config c;
  c.arch = Arch::GPT2;
  c.vocab_size = 97;
  c.d_model = 64;
  c.n_layers = 2;
  c.n_heads = 4;
  c.n_kv_heads = 4;
  c.head_dim = 16;
  c.d_ff = 256;
  c.max_ctx = 16;
  c.norm_eps = 1e-5f;
  c.bos_id = c.eos_id = 96;
  return c;
}

Weights write_gpt2(const std::string& path, DType dt) {
  const Config c = gpt2_config();
  const int d = c.d_model, ff = c.d_ff;
  std::vector<OutTensor> out;
  Weights w;
  uint32_t seed = 100;
  put(out, w, test::f32_tensor("tok_emb", {c.vocab_size, d}, test::rand_vec(size_t(c.vocab_size) * d, seed++)));
  put(out, w, test::f32_tensor("pos_emb", {c.max_ctx, d}, test::rand_vec(size_t(c.max_ctx) * d, seed++, -0.3f, 0.3f)));
  put_vec(out, w, "ln_f.w", d, seed++, 0.8f, 1.2f);
  put_vec(out, w, "ln_f.b", d, seed++, -0.1f, 0.1f);
  for (int l = 0; l < c.n_layers; ++l) {
    const std::string p = "l" + std::to_string(l) + ".";
    put_vec(out, w, p + "ln1.w", d, seed++, 0.8f, 1.2f);
    put_vec(out, w, p + "ln1.b", d, seed++, -0.1f, 0.1f);
    put_linear(out, w, p + "attn.qkv.w", dt, 3 * d, d, seed++);
    put_vec(out, w, p + "attn.qkv.b", 3 * d, seed++, -0.1f, 0.1f);
    put_linear(out, w, p + "attn.o.w", dt, d, d, seed++);
    put_vec(out, w, p + "attn.o.b", d, seed++, -0.1f, 0.1f);
    put_vec(out, w, p + "ln2.w", d, seed++, 0.8f, 1.2f);
    put_vec(out, w, p + "ln2.b", d, seed++, -0.1f, 0.1f);
    put_linear(out, w, p + "mlp.up.w", dt, ff, d, seed++);
    put_vec(out, w, p + "mlp.up.b", ff, seed++, -0.1f, 0.1f);
    put_linear(out, w, p + "mlp.down.w", dt, d, ff, seed++);
    put_vec(out, w, p + "mlp.down.b", d, seed++, -0.1f, 0.1f);
  }
  test::write_tllm(path, c, out);
  return w;
}

class RefGpt2 {
 public:
  RefGpt2(const Config& c, const Weights& w) : c_(c), w_(w), K_(c.n_layers), V_(c.n_layers) {}

  Vec forward(int token, int pos) {
    const int d = c_.d_model;
    Vec x(d);
    for (int i = 0; i < d; ++i) x[i] = double(at("tok_emb")[token * d + i]) + at("pos_emb")[pos * d + i];
    for (int l = 0; l < c_.n_layers; ++l) {
      const std::string p = "l" + std::to_string(l) + ".";
      Vec h = layernorm(x, at(p + "ln1.w"), at(p + "ln1.b"), c_.norm_eps);
      Vec qkv = matvec(at(p + "attn.qkv.w"), h, 3 * d);
      add_to(qkv, at(p + "attn.qkv.b"));
      K_[l].emplace_back(qkv.begin() + d, qkv.begin() + 2 * d);
      V_[l].emplace_back(qkv.begin() + 2 * d, qkv.end());
      Vec att = attend(Vec(qkv.begin(), qkv.begin() + d), K_[l], V_[l], c_.n_heads, c_.n_heads, c_.head_dim);
      Vec o = matvec(at(p + "attn.o.w"), att, d);
      add_to(o, at(p + "attn.o.b"));
      for (int i = 0; i < d; ++i) x[i] += o[i];
      h = layernorm(x, at(p + "ln2.w"), at(p + "ln2.b"), c_.norm_eps);
      Vec f = matvec(at(p + "mlp.up.w"), h, c_.d_ff);
      add_to(f, at(p + "mlp.up.b"));
      const double k = std::sqrt(2.0 / M_PI);
      for (double& v : f) v = 0.5 * v * (1.0 + std::tanh(k * (v + 0.044715 * v * v * v)));
      Vec down = matvec(at(p + "mlp.down.w"), f, d);
      add_to(down, at(p + "mlp.down.b"));
      for (int i = 0; i < d; ++i) x[i] += down[i];
    }
    Vec h = layernorm(x, at("ln_f.w"), at("ln_f.b"), c_.norm_eps);
    return matvec(at("tok_emb"), h, c_.vocab_size);
  }

 private:
  const std::vector<float>& at(const std::string& n) const { return w_.at(n); }
  Config c_;
  const Weights& w_;
  std::vector<std::vector<Vec>> K_, V_;
};

// ---------------------------------------------------------------------------
// LLaMA: d = 64, 4 query heads sharing 2 KV heads, head_dim 16, d_ff 96, untied head.

Config llama_config() {
  Config c;
  c.arch = Arch::LLAMA;
  c.vocab_size = 83;
  c.d_model = 64;
  c.n_layers = 2;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 16;
  c.d_ff = 96;
  c.max_ctx = 32;
  c.norm_eps = 1e-5f;
  c.rope_theta = 10000.0f;
  c.bos_id = 1;
  c.eos_id = 2;
  return c;
}

Weights write_llama(const std::string& path, DType dt) {
  const Config c = llama_config();
  const int d = c.d_model, qd = c.n_heads * c.head_dim, kvd = c.n_kv_heads * c.head_dim;
  std::vector<OutTensor> out;
  Weights w;
  uint32_t seed = 500;
  put(out, w, test::f32_tensor("tok_emb", {c.vocab_size, d}, test::rand_vec(size_t(c.vocab_size) * d, seed++)));
  put_vec(out, w, "norm_f.w", d, seed++, 0.8f, 1.2f);
  put_linear(out, w, "lm_head.w", dt, c.vocab_size, d, seed++);
  for (int l = 0; l < c.n_layers; ++l) {
    const std::string p = "l" + std::to_string(l) + ".";
    put_vec(out, w, p + "attn_norm.w", d, seed++, 0.8f, 1.2f);
    // Larger q/k weights so attention isn't nearly uniform and RoPE matters.
    put_linear(out, w, p + "attn.q.w", dt, qd, d, seed++, 3.0f);
    put_linear(out, w, p + "attn.k.w", dt, kvd, d, seed++, 3.0f);
    put_linear(out, w, p + "attn.v.w", dt, kvd, d, seed++);
    put_linear(out, w, p + "attn.o.w", dt, d, qd, seed++);
    put_vec(out, w, p + "mlp_norm.w", d, seed++, 0.8f, 1.2f);
    put_linear(out, w, p + "mlp.gate.w", dt, c.d_ff, d, seed++);
    put_linear(out, w, p + "mlp.up.w", dt, c.d_ff, d, seed++);
    put_linear(out, w, p + "mlp.down.w", dt, d, c.d_ff, seed++);
  }
  test::write_tllm(path, c, out);
  return w;
}

class RefLlama {
 public:
  RefLlama(const Config& c, const Weights& w) : c_(c), w_(w), K_(c.n_layers), V_(c.n_layers) {}

  Vec forward(int token, int pos) {
    const int d = c_.d_model, H = c_.n_heads, KVH = c_.n_kv_heads, hd = c_.head_dim;
    Vec x(at("tok_emb").begin() + token * d, at("tok_emb").begin() + (token + 1) * d);
    for (int l = 0; l < c_.n_layers; ++l) {
      const std::string p = "l" + std::to_string(l) + ".";
      Vec h = rmsnorm(x, at(p + "attn_norm.w"), c_.norm_eps);
      Vec q = matvec(at(p + "attn.q.w"), h, H * hd);
      Vec k = matvec(at(p + "attn.k.w"), h, KVH * hd);
      Vec v = matvec(at(p + "attn.v.w"), h, KVH * hd);
      rope(q, H, hd, pos, c_.rope_theta);
      rope(k, KVH, hd, pos, c_.rope_theta);
      K_[l].push_back(k);
      V_[l].push_back(v);
      Vec o = matvec(at(p + "attn.o.w"), attend(q, K_[l], V_[l], H, KVH, hd), d);
      for (int i = 0; i < d; ++i) x[i] += o[i];
      h = rmsnorm(x, at(p + "mlp_norm.w"), c_.norm_eps);
      Vec g = matvec(at(p + "mlp.gate.w"), h, c_.d_ff);
      Vec u = matvec(at(p + "mlp.up.w"), h, c_.d_ff);
      for (int i = 0; i < c_.d_ff; ++i) g[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
      Vec down = matvec(at(p + "mlp.down.w"), g, d);
      for (int i = 0; i < d; ++i) x[i] += down[i];
    }
    return matvec(at("lm_head.w"), rmsnorm(x, at("norm_f.w"), c_.norm_eps), c_.vocab_size);
  }

 private:
  const std::vector<float>& at(const std::string& n) const { return w_.at(n); }
  Config c_;
  const Weights& w_;
  std::vector<std::vector<Vec>> K_, V_;
};

// ---------------------------------------------------------------------------

// Runs `tokens` through both and returns the worst |engine - ref| / max|ref| over all
// positions. For quantized files the reference uses the dequantized weights, so the gap
// is only the kernel's per-call Q8 quantization of activations (observed ~1e-2 for
// LLaMA Q8, ~4e-3 for GPT-2; a layout or pairing bug gives O(1)).
template <typename Ref>
double run_and_compare(Model& m, Ref& ref, const std::vector<int>& tokens) {
  double worst = 0.0;
  const int V = m.config().vocab_size;
  for (size_t pos = 0; pos < tokens.size(); ++pos) {
    const float* got = m.forward(tokens[pos], int(pos));
    Vec want = ref.forward(tokens[pos], int(pos));
    double scale = 0.0, err = 0.0;
    for (int i = 0; i < V; ++i) {
      scale = std::max(scale, std::fabs(want[i]));
      err = std::max(err, std::fabs(got[i] - want[i]));
    }
    EXPECT_GT(max_abs(got, V), 0.0);
    worst = std::max(worst, err / scale);
  }
  return worst;
}

const std::vector<int> kTokens = {5, 17, 3, 3, 60, 0, 42, 11, 80, 7, 29, 64};

class TinyModel : public ::testing::TestWithParam<DType> {};

TEST_P(TinyModel, Gpt2MatchesReference) {
  const DType dt = GetParam();
  const std::string path = model_path("gpt2_" + std::to_string(int(dt)));
  Weights w = write_gpt2(path, dt);
  auto m = load_model(path);
  ASSERT_EQ(m->config().arch, Arch::GPT2);
  EXPECT_EQ(m->max_ctx(), 16);
  RefGpt2 ref(gpt2_config(), w);
  const double rel = run_and_compare(*m, ref, kTokens);
  EXPECT_LT(rel, dt == DType::F32 ? 1e-5 : 3e-2) << "dtype " << int(dt);
  std::filesystem::remove(path);
}

TEST_P(TinyModel, LlamaMatchesReference) {
  const DType dt = GetParam();
  const std::string path = model_path("llama_" + std::to_string(int(dt)));
  Weights w = write_llama(path, dt);
  auto m = load_model(path);
  ASSERT_EQ(m->config().arch, Arch::LLAMA);
  RefLlama ref(llama_config(), w);
  const double rel = run_and_compare(*m, ref, kTokens);
  EXPECT_LT(rel, dt == DType::F32 ? 1e-5 : 3e-2) << "dtype " << int(dt);
  std::filesystem::remove(path);
}

INSTANTIATE_TEST_SUITE_P(DTypes, TinyModel, ::testing::Values(DType::F32, DType::Q8, DType::Q4),
                         [](const auto& info) {
                           return std::string(info.param == DType::F32 ? "F32" : info.param == DType::Q8 ? "Q8" : "Q4");
                         });

// Two models fed the same tokens in lockstep produce identical logits (no state leaks
// between instances, e.g. via the thread-local RoPE cache).
TEST(TinyModelBasics, InterleavedInstancesAgree) {
  const std::string path = model_path("llama_det");
  write_llama(path, DType::F32);
  auto a = load_model(path);
  auto b = load_model(path);
  const int V = a->config().vocab_size;
  for (int pos = 0; pos < 8; ++pos) {
    const float* la = a->forward(kTokens[pos], pos);
    std::vector<float> got_a(la, la + V);
    const float* lb = b->forward(kTokens[pos], pos);
    ASSERT_EQ(got_a, std::vector<float>(lb, lb + V)) << "pos " << pos;
  }
  std::filesystem::remove(path);
}

TEST(TinyModelBasics, RejectsOutOfRangeInputs) {
  const std::string path = model_path("gpt2_range");
  write_gpt2(path, DType::F32);
  auto m = load_model(path, 4);
  EXPECT_EQ(m->max_ctx(), 4);
  EXPECT_THROW(m->forward(97, 0), std::out_of_range);
  EXPECT_THROW(m->forward(-1, 0), std::out_of_range);
  EXPECT_THROW(m->forward(0, 4), std::out_of_range);
  EXPECT_NO_THROW(m->forward(0, 0));
  std::filesystem::remove(path);

  const std::string lpath = model_path("llama_range");
  write_llama(lpath, DType::F32);
  auto l = load_model(lpath);
  EXPECT_EQ(l->max_ctx(), 32);
  EXPECT_THROW(l->forward(83, 0), std::out_of_range);
  EXPECT_THROW(l->forward(0, 32), std::out_of_range);
  std::filesystem::remove(lpath);
}

}  // namespace
}  // namespace tinyllm
