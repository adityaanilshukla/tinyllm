// Numerically stable row softmax: one block per row. Pass 1: block max-reduce.
// Pass 2: y = exp(x - max), block sum-reduce. Pass 3: y *= 1/sum.
#include "common.cuh"
#include "ops_cuda.h"

namespace tinyllm {
namespace cuda {

namespace {

constexpr int SM_THREADS = 256;

__global__ void __launch_bounds__(SM_THREADS)
    softmax_kernel(float* y, const float* x, int n) {
  __shared__ float scratch[32];
  const size_t row = blockIdx.x;
  const float* xr = x + row * (size_t)n;
  float* yr = y + row * (size_t)n;

  float m = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, xr[i]);
  m = block_reduce_max(m, scratch);

  // Each thread touches only its own indices in every pass, so y may alias x.
  float s = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float e = expf(xr[i] - m);
    yr[i] = e;
    s += e;
  }
  s = block_reduce_sum(s, scratch);

  const float inv = 1.0f / s;
  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] *= inv;
}

}  // namespace

void softmax(float* y, const float* x, int rows, int n, cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return;
  softmax_kernel<<<rows, SM_THREADS, 0, stream>>>(y, x, n);
  CUDA_CHECK_LAUNCH();
}

}  // namespace cuda
}  // namespace tinyllm
