// Fused layer norm: one block per row, single pass over x for (shifted) sum and sum of squares,
// block-wide warp-shuffle reductions, then a second pass to normalize and write y.
#include "common.cuh"
#include "ops_cuda.h"

namespace tinyllm {
namespace cuda {

namespace {

constexpr int LN_THREADS = 256;

__global__ void __launch_bounds__(LN_THREADS)
    layernorm_kernel(float* y, const float* x, const float* __restrict__ gamma,
                     const float* __restrict__ beta, int D, float eps) {
  __shared__ float scratch[32];
  const size_t row = blockIdx.x;
  const float* xr = x + row * (size_t)D;
  float* yr = y + row * (size_t)D;

  // Shifted sums (shift = x[0]) avoid catastrophic cancellation in E[x^2] - E[x]^2
  // when the row mean is large relative to its spread.
  const float shift = xr[0];
  float s = 0.0f;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = xr[i] - shift;
    s += v;
    ss = fmaf(v, v, ss);
  }
  s = block_reduce_sum(s, scratch);
  ss = block_reduce_sum(ss, scratch);  // block_reduce_sum syncs before reusing scratch

  const float inv_d = 1.0f / (float)D;
  const float mean_shifted = s * inv_d;
  const float var = fmaxf(ss * inv_d - mean_shifted * mean_shifted, 0.0f);
  const float mean = shift + mean_shifted;
  const float rstd = rsqrtf(var + eps);

  // Each thread re-reads only the elements it read above, so y aliasing x is safe.
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = (xr[i] - mean) * rstd;
    const float g = gamma != nullptr ? gamma[i] : 1.0f;
    const float b = beta != nullptr ? beta[i] : 0.0f;
    yr[i] = fmaf(v, g, b);
  }
}

}  // namespace

void layernorm(float* y, const float* x, const float* gamma, const float* beta, int rows, int D,
               float eps, cudaStream_t stream) {
  if (rows <= 0 || D <= 0) return;
  layernorm_kernel<<<rows, LN_THREADS, 0, stream>>>(y, x, gamma, beta, D, eps);
  CUDA_CHECK_LAUNCH();
}

}  // namespace cuda
}  // namespace tinyllm
