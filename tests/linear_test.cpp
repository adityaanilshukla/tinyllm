// linear() for F32 / Q8 / Q4 weights against a naive double-precision matvec.
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include "test_util.h"
#include "tinyllm/ops.h"
#include "tinyllm/quant.h"

namespace tinyllm {
namespace {

using test::rand_vec;

Tensor view(const test::OutTensor& t) {
  Tensor v;
  v.dtype = t.dtype;
  v.ndim = static_cast<int32_t>(t.shape.size());
  for (size_t i = 0; i < t.shape.size(); ++i) v.shape[i] = t.shape[i];
  v.data = t.bytes.data();
  v.nbytes = t.bytes.size();
  return v;
}

std::vector<double> ref_matvec(const std::vector<float>& W, const std::vector<float>& x, const float* bias, int K,
                               int N) {
  std::vector<double> y(N);
  for (int r = 0; r < N; ++r) {
    double s = bias ? bias[r] : 0.0;
    for (int k = 0; k < K; ++k) s += double(W[size_t(r) * K + k]) * x[k];
    y[r] = s;
  }
  return y;
}

// Sizes cover: tiny (single-threaded), N not a multiple of 4 or of the thread grain,
// K not a multiple of the 8/16-wide SIMD loops, and N*K above the 1 << 16 threading cutoff.
class LinearF32 : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(LinearF32, MatchesReference) {
  const auto [N, K] = GetParam();
  auto W = rand_vec(size_t(N) * K, 11);
  auto x = rand_vec(K, 12);
  auto bias = rand_vec(N, 13);
  auto t = test::f32_tensor("w", {N, K}, W);
  for (const float* b : {static_cast<const float*>(nullptr), static_cast<const float*>(bias.data())}) {
    std::vector<float> y(N, -999.0f);
    linear(y.data(), x.data(), view(t), b, K, N);
    auto ref = ref_matvec(W, x, b, K, N);
    for (int r = 0; r < N; ++r) ASSERT_NEAR(y[r], ref[r], 1e-5 * std::sqrt(double(K)) + 1e-6) << "row " << r;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, LinearF32,
                         ::testing::Values(std::make_tuple(1, 1), std::make_tuple(3, 5), std::make_tuple(4, 8),
                                           std::make_tuple(7, 33), std::make_tuple(1027, 67),
                                           std::make_tuple(2053, 768), std::make_tuple(50, 3001)));

// Quantized: two checks per size.
//  - Kernel exactness: equal to the dot of dequantized W with dequantized Q8(x), since
//    the kernel quantizes the activation once per call.
//  - End-to-end: close to the true f32 product, with a tolerance from the quantization step.
class LinearQuant : public ::testing::TestWithParam<std::tuple<DType, int, int>> {};

TEST_P(LinearQuant, MatchesReference) {
  const auto [dtype, N, K] = GetParam();
  auto W = rand_vec(size_t(N) * K, 21);
  auto x = rand_vec(K, 22);
  auto bias = rand_vec(N, 23);
  auto t = test::quant_tensor("w", dtype, N, K, W);
  auto Wd = test::dequant(t);

  std::vector<BlockQ8> xq(K / kQK);
  quantize_row_q8(x.data(), xq.data(), K);
  std::vector<float> xd(K);
  dequantize_row(DType::Q8, xq.data(), xd.data(), K);

  std::vector<float> y(N, -999.0f);
  linear(y.data(), x.data(), view(t), bias.data(), K, N);
  auto exact = ref_matvec(Wd, xd, bias.data(), K, N);
  auto f32 = ref_matvec(W, x, bias.data(), K, N);
  // Per-element weight error is at most one quant step (amax/127 or amax/8), activation
  // error half a step; random-sign errors grow like sqrt(K).
  const double wstep = dtype == DType::Q8 ? 1.0 / 127 : 1.0 / 8;
  const double e2e_tol = 4.0 * (wstep + 0.5 / 127) * std::sqrt(double(K));
  for (int r = 0; r < N; ++r) {
    ASSERT_NEAR(y[r], exact[r], 2e-5 * std::sqrt(double(K)) + 1e-6) << "row " << r;
    ASSERT_NEAR(y[r], f32[r], e2e_tol) << "row " << r;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, LinearQuant,
                         ::testing::Combine(::testing::Values(DType::Q8, DType::Q4),
                                            ::testing::Values(1, 3, 37, 1001, 2051),
                                            ::testing::Values(32, 96, 160, 2048)));

TEST(Linear, ShapeMismatchThrows) {
  auto t = test::f32_tensor("w", {4, 8}, rand_vec(32, 1));
  std::vector<float> x(8), y(8);
  EXPECT_THROW(linear(y.data(), x.data(), view(t), nullptr, 8, 3), std::runtime_error);
  EXPECT_THROW(linear(y.data(), x.data(), view(t), nullptr, 4, 8), std::runtime_error);
  auto v = test::f32_tensor("v", {32}, rand_vec(32, 1));
  EXPECT_THROW(linear(y.data(), x.data(), view(v), nullptr, 32, 1), std::runtime_error);
}

// The same matvec called concurrently from several threads must still be correct
// (each pool falls back to inline execution or serializes callers).
TEST(Linear, ConcurrentCallersGetCorrectResults) {
  const int N = 1024, K = 256;
  auto W = rand_vec(size_t(N) * K, 31);
  auto x = rand_vec(K, 32);
  auto tf = test::f32_tensor("w", {N, K}, W);
  auto tq = test::quant_tensor("w", DType::Q8, N, K, W);
  std::vector<float> want_f(N), want_q(N);
  linear(want_f.data(), x.data(), view(tf), nullptr, K, N);
  linear(want_q.data(), x.data(), view(tq), nullptr, K, N);

  std::vector<std::thread> threads;
  std::vector<int> bad(4, 0);
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      std::vector<float> y(N);
      for (int it = 0; it < 50; ++it) {
        linear(y.data(), x.data(), view(t % 2 ? tq : tf), nullptr, K, N);
        if (y != (t % 2 ? want_q : want_f)) ++bad[t];
      }
    });
  }
  for (auto& th : threads) th.join();
  for (int t = 0; t < 4; ++t) EXPECT_EQ(bad[t], 0) << "thread " << t;
}

}  // namespace
}  // namespace tinyllm
