#pragma once
#include <cstddef>
#include <cstdint>

#include "tinyllm/tensor.h"

namespace tinyllm {

// Block quantization along the input dim, 32 values per block. Rows must have
// in % kQK == 0. A quantized row of n values is n / kQK blocks laid out back to back.
constexpr int kQK = 32;

// x[j] = qs[j] * d
struct BlockQ8 {
  float d;
  int8_t qs[kQK];
};

// q in [0, 15], x[j] = (q - 8) * d. Low nibble of qs[j] is element j,
// high nibble is element j + 16.
struct BlockQ4 {
  float d;
  uint8_t qs[kQK / 2];
};

static_assert(sizeof(BlockQ8) == 36, "BlockQ8 layout");
static_assert(sizeof(BlockQ4) == 20, "BlockQ4 layout");

size_t quant_row_bytes(DType dtype, int64_t n);  // F32 -> n * 4

void quantize_row_q8(const float* x, void* out, int64_t n);
void quantize_row_q4(const float* x, void* out, int64_t n);
void dequantize_row(DType dtype, const void* in, float* out, int64_t n);

// y[N] = W[N, K] x[K], W quantized row-wise. Implemented in src/quant.cpp.
void matvec_q8(float* y, const void* W, const float* x, int64_t N, int64_t K);
void matvec_q4(float* y, const void* W, const float* x, int64_t N, int64_t K);

}  // namespace tinyllm
