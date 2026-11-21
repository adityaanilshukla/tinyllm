// softmax, layernorm, GELU, add, attention (MHA + GQA), rmsnorm, RoPE, SwiGLU against
// naive double-precision references.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "test_util.h"
#include "tinyllm/ops.h"

namespace tinyllm {
namespace {

using test::rand_vec;

std::vector<double> ref_softmax(const std::vector<double>& x) {
  double mx = -std::numeric_limits<double>::infinity();
  for (double v : x) mx = std::max(mx, v);
  std::vector<double> y(x.size());
  double s = 0.0;
  for (size_t i = 0; i < x.size(); ++i) s += (y[i] = std::exp(x[i] - mx));
  for (double& v : y) v /= s;
  return y;
}

// ---------------------------------------------------------------------------

TEST(Softmax, MatchesReference) {
  for (int n : {1, 2, 7, 100, 50257}) {
    auto x = rand_vec(n, n, -10.0f, 10.0f);
    std::vector<double> xd(x.begin(), x.end());
    auto ref = ref_softmax(xd);
    softmax(x.data(), n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(x[i], ref[i], 1e-6 + 1e-5 * ref[i]) << "n=" << n << " i=" << i;
      sum += x[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-4) << "n=" << n;
  }
}

TEST(Softmax, StableWithLargeInputs) {
  // exp(1000) overflows float; a naive softmax returns NaN/inf here.
  std::vector<float> x = {1000.0f, 1001.0f, 1002.0f, -1000.0f, 88.0f};
  std::vector<double> xd(x.begin(), x.end());
  auto ref = ref_softmax(xd);
  softmax(x.data(), int(x.size()));
  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_TRUE(std::isfinite(x[i]));
    EXPECT_NEAR(x[i], ref[i], 1e-6);
  }
  std::vector<float> big(16, 3.0e38f);
  softmax(big.data(), 16);
  for (float v : big) EXPECT_FLOAT_EQ(v, 1.0f / 16);
  std::vector<float> neg = {-1e30f, -1e30f + 1e24f};
  softmax(neg.data(), 2);
  EXPECT_TRUE(std::isfinite(neg[0]) && std::isfinite(neg[1]));
  EXPECT_NEAR(neg[0] + neg[1], 1.0f, 1e-6);
}

TEST(Softmax, IgnoresNonPositiveLength) {
  float x[1] = {5.0f};
  softmax(x, 0);
  EXPECT_EQ(x[0], 5.0f);
}

// ---------------------------------------------------------------------------

std::vector<double> ref_layernorm(const std::vector<float>& x, const float* g, const float* b, double eps) {
  const size_t D = x.size();
  double mean = 0.0, var = 0.0;
  for (float v : x) mean += v;
  mean /= D;
  for (float v : x) var += (v - mean) * (v - mean);
  var /= D;
  std::vector<double> y(D);
  for (size_t i = 0; i < D; ++i) y[i] = (x[i] - mean) / std::sqrt(var + eps) * (g ? g[i] : 1.0) + (b ? b[i] : 0.0);
  return y;
}

TEST(LayerNorm, MatchesReference) {
  for (int D : {1, 5, 64, 768, 1001}) {
    // Offset mean so a single-pass E[x^2] - E[x]^2 would lose precision.
    auto x = rand_vec(D, D, 99.0f, 101.0f);
    auto g = rand_vec(D, D + 1, 0.5f, 1.5f);
    auto b = rand_vec(D, D + 2);
    std::vector<float> y(D);
    layernorm(y.data(), x.data(), g.data(), b.data(), D, 1e-5f);
    auto ref = ref_layernorm(x, g.data(), b.data(), 1e-5);
    for (int i = 0; i < D; ++i) ASSERT_NEAR(y[i], ref[i], 2e-4) << "D=" << D << " i=" << i;
  }
}

TEST(LayerNorm, NullGammaBetaIsIdentityAffine) {
  auto x = rand_vec(33, 4, -3.0f, 3.0f);
  std::vector<float> y(33);
  layernorm(y.data(), x.data(), nullptr, nullptr, 33, 1e-5f);
  auto ref = ref_layernorm(x, nullptr, nullptr, 1e-5);
  for (int i = 0; i < 33; ++i) EXPECT_NEAR(y[i], ref[i], 1e-5);
}

TEST(LayerNorm, InPlace) {
  auto x = rand_vec(64, 5);
  auto ref = ref_layernorm(x, nullptr, nullptr, 1e-5);
  layernorm(x.data(), x.data(), nullptr, nullptr, 64, 1e-5f);
  for (int i = 0; i < 64; ++i) EXPECT_NEAR(x[i], ref[i], 1e-5);
}

// ---------------------------------------------------------------------------

TEST(Gelu, MatchesTanhApproximation) {
  auto x = rand_vec(1000, 6, -8.0f, 8.0f);
  x.insert(x.end(), {0.0f, -0.0f, 30.0f, -30.0f, 1e-8f});
  auto y = x;
  gelu(y.data(), int64_t(y.size()));
  const double c = std::sqrt(2.0 / M_PI);
  for (size_t i = 0; i < x.size(); ++i) {
    const double v = x[i];
    const double ref = 0.5 * v * (1.0 + std::tanh(c * (v + 0.044715 * v * v * v)));
    ASSERT_NEAR(y[i], ref, 1e-6 + 1e-6 * std::fabs(ref)) << "x=" << v;
  }
  // Sanity against exact erf GELU: the tanh form is within ~1e-3 of it.
  for (size_t i = 0; i < x.size(); ++i) {
    const double v = x[i];
    EXPECT_NEAR(y[i], 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))), 2e-3);
  }
}

TEST(AddInplace, Adds) {
  auto a = rand_vec(17, 1), b = rand_vec(17, 2);
  auto y = a;
  add_inplace(y.data(), b.data(), 17);
  for (int i = 0; i < 17; ++i) EXPECT_EQ(y[i], a[i] + b[i]);
}

// ---------------------------------------------------------------------------
// Attention: q [H * hd], k/v [n_pos][KVH * hd]; head h reads KV head h / (H / KVH).

std::vector<double> ref_attention(const std::vector<float>& q, const std::vector<float>& k,
                                  const std::vector<float>& v, int n_pos, int H, int KVH, int hd) {
  std::vector<double> out(size_t(H) * hd, 0.0);
  const int group = H / KVH;
  const size_t stride = size_t(KVH) * hd;
  for (int h = 0; h < H; ++h) {
    const int kvh = h / group;
    std::vector<double> s(n_pos);
    for (int t = 0; t < n_pos; ++t) {
      double d = 0.0;
      for (int i = 0; i < hd; ++i) d += double(q[h * hd + i]) * k[t * stride + kvh * hd + i];
      s[t] = d / std::sqrt(double(hd));
    }
    s = ref_softmax(s);
    for (int t = 0; t < n_pos; ++t)
      for (int i = 0; i < hd; ++i) out[h * hd + i] += s[t] * v[t * stride + kvh * hd + i];
  }
  return out;
}

// (n_heads, n_kv_heads, head_dim, n_pos). Large cases cross the parallel threshold.
class Attention : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(Attention, MatchesReference) {
  const auto [H, KVH, hd, n_pos] = GetParam();
  auto q = rand_vec(size_t(H) * hd, 41, -2.0f, 2.0f);
  auto k = rand_vec(size_t(n_pos) * KVH * hd, 42, -2.0f, 2.0f);
  auto v = rand_vec(size_t(n_pos) * KVH * hd, 43);
  std::vector<float> out(size_t(H) * hd, -999.0f), scores(size_t(H) * n_pos);
  attention(out.data(), q.data(), k.data(), v.data(), n_pos, H, KVH, hd, scores.data());
  auto ref = ref_attention(q, k, v, n_pos, H, KVH, hd);
  for (size_t i = 0; i < out.size(); ++i) ASSERT_NEAR(out[i], ref[i], 1e-5) << "i=" << i;
}

INSTANTIATE_TEST_SUITE_P(MHA, Attention,
                         ::testing::Values(std::make_tuple(1, 1, 8, 1), std::make_tuple(4, 4, 16, 1),
                                           std::make_tuple(4, 4, 16, 9), std::make_tuple(12, 12, 64, 77),
                                           std::make_tuple(12, 12, 64, 300), std::make_tuple(3, 3, 7, 5)));
INSTANTIATE_TEST_SUITE_P(GQA, Attention,
                         ::testing::Values(std::make_tuple(8, 2, 16, 1), std::make_tuple(8, 2, 16, 13),
                                           std::make_tuple(32, 4, 64, 200), std::make_tuple(6, 1, 12, 31),
                                           std::make_tuple(32, 4, 64, 3)));

// With one position the output is exactly that position's V row (softmax of one score).
TEST(AttentionBasics, SinglePositionReturnsV) {
  const int H = 4, KVH = 2, hd = 8;
  auto q = rand_vec(H * hd, 1), k = rand_vec(KVH * hd, 2), v = rand_vec(KVH * hd, 3);
  std::vector<float> out(H * hd), scores(H);
  attention(out.data(), q.data(), k.data(), v.data(), 1, H, KVH, hd, scores.data());
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < hd; ++i) EXPECT_FLOAT_EQ(out[h * hd + i], v[(h / 2) * hd + i]);
}

TEST(AttentionBasics, RejectsBadKvHeads) {
  std::vector<float> buf(1024);
  EXPECT_THROW(attention(buf.data(), buf.data(), buf.data(), buf.data(), 1, 6, 4, 8, buf.data()),
               std::runtime_error);
  EXPECT_THROW(attention(buf.data(), buf.data(), buf.data(), buf.data(), 1, 6, 0, 8, buf.data()),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// LLaMA ops.

TEST(RmsNorm, MatchesReference) {
  for (int D : {1, 15, 16, 37, 2048}) {
    auto x = rand_vec(D, D, -3.0f, 3.0f);
    auto g = rand_vec(D, D + 1, 0.5f, 1.5f);
    std::vector<float> y(D);
    rmsnorm(y.data(), x.data(), g.data(), D, 1e-5f);
    double ss = 0.0;
    for (float v : x) ss += double(v) * v;
    const double inv = 1.0 / std::sqrt(ss / D + 1e-5);
    for (int i = 0; i < D; ++i) ASSERT_NEAR(y[i], x[i] * inv * g[i], 2e-5 * std::fabs(x[i] * inv * g[i]) + 1e-6);
  }
}

// HF LLaMA: q_embed = q * cos + rotate_half(q) * sin, rotate_half(x) = cat(-x2, x1),
// cos/sin = cos/sin(pos * theta^(-2i/hd)) repeated over both halves.
std::vector<double> ref_rope(const std::vector<float>& x, int H, int hd, int pos, double theta) {
  std::vector<double> y(x.size());
  const int half = hd / 2;
  for (int h = 0; h < H; ++h) {
    const float* xh = x.data() + h * hd;
    for (int j = 0; j < hd; ++j) {
      const int i = j % half;
      const double ang = pos * std::pow(theta, -2.0 * i / hd);
      const double rot = j < half ? -double(xh[j + half]) : double(xh[j - half]);
      y[h * hd + j] = xh[j] * std::cos(ang) + rot * std::sin(ang);
    }
  }
  return y;
}

TEST(Rope, MatchesHfRotateHalf) {
  // (n_heads, head_dim, pos, theta); positions out of order exercise the cos/sin cache.
  const std::vector<std::tuple<int, int, int, float>> cases = {
      {1, 2, 1, 10000.f},    {4, 64, 0, 10000.f},  {4, 64, 5, 10000.f},   {4, 64, 3, 10000.f},
      {4, 64, 3, 500000.f},  {2, 16, 3, 10000.f},  {32, 64, 1000, 10000.f}, {8, 128, 2047, 10000.f},
      {4, 64, 5, 10000.f}};
  for (const auto& [H, hd, pos, theta] : cases) {
    auto x = rand_vec(size_t(H) * hd, H * 1000 + hd + pos, -2.0f, 2.0f);
    auto ref = ref_rope(x, H, hd, pos, theta);
    rope(x.data(), H, hd, pos, theta);
    // HF (and rope()) compute the angle in float32, so the error grows with pos: two
    // roundings of ~6e-8 relative each, times a pair norm of at most 2 * sqrt(2).
    const double tol = 1e-5 + 4e-7 * pos;
    for (size_t i = 0; i < x.size(); ++i)
      ASSERT_NEAR(x[i], ref[i], tol) << "H=" << H << " hd=" << hd << " pos=" << pos << " theta=" << theta
                                     << " i=" << i;
  }
}

TEST(Rope, PreservesPairNorms) {
  const int H = 2, hd = 32;
  auto x = rand_vec(H * hd, 9);
  auto y = x;
  rope(y.data(), H, hd, 123, 10000.f);
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < hd / 2; ++i) {
      const float* a = x.data() + h * hd;
      const float* b = y.data() + h * hd;
      EXPECT_NEAR(a[i] * a[i] + a[i + hd / 2] * a[i + hd / 2], b[i] * b[i] + b[i + hd / 2] * b[i + hd / 2], 1e-5);
    }
}

TEST(SiluMul, MatchesReference) {
  auto g = rand_vec(500, 10, -10.0f, 10.0f);
  auto u = rand_vec(500, 11, -3.0f, 3.0f);
  g.insert(g.end(), {100.0f, -100.0f, -1000.0f, 0.0f});
  u.insert(u.end(), {2.0f, 2.0f, 2.0f, 2.0f});
  auto y = g;
  silu_mul(y.data(), u.data(), int64_t(y.size()));
  for (size_t i = 0; i < g.size(); ++i) {
    const double ref = g[i] / (1.0 + std::exp(-double(g[i]))) * u[i];
    ASSERT_TRUE(std::isfinite(y[i])) << "g=" << g[i];
    ASSERT_NEAR(y[i], ref, 1e-6 + 1e-6 * std::fabs(ref)) << "g=" << g[i];
  }
}

}  // namespace
}  // namespace tinyllm
