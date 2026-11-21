// Q8/Q4 block quantization: round-trip error bounds and the exact on-disk layout.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "test_util.h"
#include "tinyllm/quant.h"

namespace tinyllm {
namespace {

using test::rand_vec;

TEST(Quant, RowBytes) {
  EXPECT_EQ(quant_row_bytes(DType::F32, 7), 28u);
  EXPECT_EQ(quant_row_bytes(DType::Q8, 64), 2 * 36u);
  EXPECT_EQ(quant_row_bytes(DType::Q4, 96), 3 * 20u);
  EXPECT_THROW(quant_row_bytes(DType::Q8, 33), std::runtime_error);
  EXPECT_THROW(quant_row_bytes(DType::Q4, 16), std::runtime_error);
}

TEST(Quant, RejectsRowsNotMultipleOf32) {
  std::vector<float> x(48, 1.0f);
  std::vector<uint8_t> buf(1024);
  EXPECT_THROW(quantize_row_q8(x.data(), buf.data(), 48), std::runtime_error);
  EXPECT_THROW(quantize_row_q4(x.data(), buf.data(), 48), std::runtime_error);
}

TEST(Quant, Q8RoundTripWithinHalfStep) {
  const int n = 32 * 8;
  // Blocks with very different magnitudes, so a shared scale would fail.
  std::vector<float> x = rand_vec(n, 1);
  for (int b = 0; b < n / kQK; ++b)
    for (int j = 0; j < kQK; ++j) x[b * kQK + j] *= std::pow(10.0f, float(b) - 4.0f);
  std::vector<BlockQ8> q(n / kQK);
  quantize_row_q8(x.data(), q.data(), n);
  std::vector<float> y(n);
  dequantize_row(DType::Q8, q.data(), y.data(), n);
  for (int b = 0; b < n / kQK; ++b) {
    float amax = 0.f;
    for (int j = 0; j < kQK; ++j) amax = std::max(amax, std::fabs(x[b * kQK + j]));
    EXPECT_FLOAT_EQ(q[b].d, amax / 127.f);
    const float bound = 0.5f * q[b].d * (1.0f + 1e-5f);
    for (int j = 0; j < kQK; ++j) {
      const int i = b * kQK + j;
      EXPECT_LE(std::fabs(x[i] - y[i]), bound) << "block " << b << " elem " << j;
      EXPECT_LE(std::abs(int(q[b].qs[j])), 127);
    }
  }
}

TEST(Quant, Q4RoundTripWithinBound) {
  const int n = 32 * 16;
  std::vector<float> x = rand_vec(n, 2, -3.0f, 3.0f);
  std::vector<BlockQ4> q(n / kQK);
  quantize_row_q4(x.data(), q.data(), n);
  std::vector<float> y(n);
  dequantize_row(DType::Q4, q.data(), y.data(), n);
  for (int b = 0; b < n / kQK; ++b) {
    // Scale is chosen so the largest-magnitude element maps exactly to q = 0 (-8 * d).
    float amax = 0.f, m = 0.f;
    for (int j = 0; j < kQK; ++j)
      if (std::fabs(x[b * kQK + j]) > amax) amax = std::fabs(m = x[b * kQK + j]);
    EXPECT_FLOAT_EQ(q[b].d, m / -8.f);
    const float d = std::fabs(q[b].d);
    for (int j = 0; j < kQK; ++j) {
      const int i = b * kQK + j;
      // Codes cover [-8, 7] * d, so values on the opposite side of m can clip by up to one step.
      const float t = x[i] / q[b].d;
      const float bound = (t > 7.5f ? 1.0f : 0.5f) * d * (1.0f + 1e-5f);
      EXPECT_LE(std::fabs(x[i] - y[i]), bound) << "block " << b << " elem " << j;
    }
  }
}

TEST(Quant, ZeroBlocksRoundTripToZero) {
  std::vector<float> x(64, 0.0f);
  BlockQ8 q8[2];
  BlockQ4 q4[2];
  quantize_row_q8(x.data(), q8, 64);
  quantize_row_q4(x.data(), q4, 64);
  std::vector<float> y(64, 1.0f);
  dequantize_row(DType::Q8, q8, y.data(), 64);
  for (float v : y) EXPECT_EQ(v, 0.0f);
  y.assign(64, 1.0f);
  dequantize_row(DType::Q4, q4, y.data(), 64);
  for (float v : y) EXPECT_EQ(v, 0.0f);
}

// Exact layout: fp32 scale, then 16 bytes where the low nibble is element j and the
// high nibble is element j + 16. Values -8..7 with max |x| = 8 give d = 1 and q = x + 8.
TEST(Quant, Q4NibbleLayoutIsExact) {
  float x[kQK];
  for (int j = 0; j < 16; ++j) {
    x[j] = float(j - 8);       // -8 .. 7
    x[j + 16] = float(7 - j);  //  7 .. -8
  }
  BlockQ4 q;
  quantize_row_q4(x, &q, kQK);
  EXPECT_EQ(q.d, 1.0f);
  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(q.qs[j] & 0x0F, j) << "low nibble " << j;
    EXPECT_EQ(q.qs[j] >> 4, 15 - j) << "high nibble " << j;
  }
  // Raw bytes: scale first, nibbles right after it.
  uint8_t raw[sizeof(BlockQ4)];
  std::memcpy(raw, &q, sizeof(raw));
  float d;
  std::memcpy(&d, raw, 4);
  EXPECT_EQ(d, 1.0f);
  EXPECT_EQ(raw[4], 0x0F << 4 | 0x00);
  EXPECT_EQ(raw[4 + 15], 0x00 << 4 | 0x0F);
}

TEST(Quant, Q4HandBuiltBlockDequantizes) {
  BlockQ4 b;
  b.d = 0.5f;
  for (int j = 0; j < 16; ++j) b.qs[j] = uint8_t(((j * 7 + 3) & 0xF) << 4 | (j & 0xF));
  float y[kQK];
  dequantize_row(DType::Q4, &b, y, kQK);
  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(y[j], float((j & 0xF) - 8) * 0.5f);
    EXPECT_EQ(y[j + 16], float(((j * 7 + 3) & 0xF) - 8) * 0.5f);
  }
}

TEST(Quant, Q8HandBuiltBlockDequantizes) {
  BlockQ8 b[2];
  b[0].d = 0.25f;
  b[1].d = -2.0f;
  for (int j = 0; j < kQK; ++j) {
    b[0].qs[j] = int8_t(j * 8 - 127);
    b[1].qs[j] = int8_t(-j);
  }
  float y[2 * kQK];
  dequantize_row(DType::Q8, b, y, 2 * kQK);
  for (int j = 0; j < kQK; ++j) {
    EXPECT_EQ(y[j], float(j * 8 - 127) * 0.25f);
    EXPECT_EQ(y[kQK + j], float(-j) * -2.0f);
  }
}

// matvec on hand-built blocks: the result must equal the dot of the dequantized weights
// with the dequantized Q8 activations (the kernel quantizes x the same way).
template <typename Block>
void check_matvec_hand_built(DType dtype) {
  const int N = 5, K = 96, nb = K / kQK;
  std::vector<Block> W(N * nb);
  std::mt19937 rng(7);
  for (auto& b : W) {
    b.d = std::uniform_real_distribution<float>(-0.1f, 0.1f)(rng);
    for (auto& q : b.qs) q = static_cast<std::remove_reference_t<decltype(q)>>(rng() & 0xFF);
  }
  std::vector<float> x = rand_vec(K, 8);
  std::vector<BlockQ8> xq(nb);
  quantize_row_q8(x.data(), xq.data(), K);
  std::vector<float> xd(K), wd(K), y(N);
  dequantize_row(DType::Q8, xq.data(), xd.data(), K);
  if (dtype == DType::Q8) matvec_q8(y.data(), W.data(), x.data(), N, K);
  else matvec_q4(y.data(), W.data(), x.data(), N, K);
  for (int r = 0; r < N; ++r) {
    dequantize_row(dtype, W.data() + r * nb, wd.data(), K);
    double ref = 0.0, mag = 0.0;
    for (int k = 0; k < K; ++k) {
      ref += double(wd[k]) * xd[k];
      mag += std::fabs(double(wd[k]) * xd[k]);
    }
    EXPECT_NEAR(y[r], ref, 1e-5 * mag + 1e-7) << "row " << r;
  }
}

TEST(Quant, MatvecQ8HandBuiltBlocks) { check_matvec_hand_built<BlockQ8>(DType::Q8); }
TEST(Quant, MatvecQ4HandBuiltBlocks) { check_matvec_hand_built<BlockQ4>(DType::Q4); }

}  // namespace
}  // namespace tinyllm
