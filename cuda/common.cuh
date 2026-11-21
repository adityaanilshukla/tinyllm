#pragma once
// Shared CUDA helpers: error-checking macros and warp/block reductions.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                                    \
  do {                                                                                      \
    cudaError_t err__ = (expr);                                                             \
    if (err__ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__),         \
                   __FILE__, __LINE__, cudaGetErrorString(err__));                          \
      std::abort();                                                                         \
    }                                                                                       \
  } while (0)

// Check for launch errors right after a <<<>>> launch (does not synchronize).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace tinyllm {
namespace cuda {

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(kFullMask, v, off);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, off));
  return v;
}

// Block-wide reductions. blockDim.x must be a multiple of 32 and <= 1024.
// `scratch` must hold at least 32 floats. Every thread gets the result.
// Contains __syncthreads(), so all threads of the block must call it.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_sum(v);
  __syncthreads();  // protect scratch from a previous use
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (lane < nwarps) ? scratch[lane] : 0.0f;
  v = warp_reduce_sum(v);
  return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_max(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (lane < nwarps) ? scratch[lane] : -INFINITY;
  v = warp_reduce_max(v);
  return v;
}

}  // namespace cuda
}  // namespace tinyllm
