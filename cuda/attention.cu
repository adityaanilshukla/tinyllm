// Fused single-query attention over a KV cache (decode step), GQA-aware.
// One block per query head:
//   1. stage q (pre-scaled by 1/sqrt(head_dim)) in shared memory
//   2. scores[t] = q . k[t] for t < n_pos: one warp per position, lanes split head_dim,
//      warp-shuffle reduce; scores live in shared memory
//   3. stable softmax over scores (block max-reduce, exp, block sum-reduce)
//   4. out[d] = sum_t p[t] * v[t][d]: threads split head_dim, so V reads are coalesced
// Dynamic shared memory: (head_dim + n_pos) floats.
#include <cmath>

#include "common.cuh"
#include "ops_cuda.h"

namespace tinyllm {
namespace cuda {

namespace {

constexpr int ATT_THREADS = 128;

__global__ void __launch_bounds__(ATT_THREADS)
    attention_kernel(float* __restrict__ out, const float* __restrict__ q,
                     const float* __restrict__ k, const float* __restrict__ v, int n_pos,
                     int n_heads, int n_kv_heads, int head_dim, float scale) {
  extern __shared__ float smem[];
  float* sq = smem;                 // [head_dim]
  float* scores = smem + head_dim;  // [n_pos]
  __shared__ float scratch[32];

  const int h = blockIdx.x;
  const int group = n_heads / n_kv_heads;
  const int kvh = h / group;
  const size_t kv_stride = (size_t)n_kv_heads * head_dim;
  const float* qh = q + (size_t)h * head_dim;
  const float* kbase = k + (size_t)kvh * head_dim;
  const float* vbase = v + (size_t)kvh * head_dim;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) sq[d] = qh[d] * scale;
  __syncthreads();

  // Scores: warp per position.
  for (int t = warp; t < n_pos; t += nwarps) {
    const float* kt = kbase + (size_t)t * kv_stride;
    float s = 0.0f;
    for (int d = lane; d < head_dim; d += 32) s = fmaf(sq[d], kt[d], s);
    s = warp_reduce_sum(s);
    if (lane == 0) scores[t] = s;
  }
  __syncthreads();

  // Stable softmax over scores[0..n_pos).
  float m = -INFINITY;
  for (int t = threadIdx.x; t < n_pos; t += blockDim.x) m = fmaxf(m, scores[t]);
  m = block_reduce_max(m, scratch);

  float sum = 0.0f;
  for (int t = threadIdx.x; t < n_pos; t += blockDim.x) {
    const float e = expf(scores[t] - m);
    scores[t] = e;
    sum += e;
  }
  sum = block_reduce_sum(sum, scratch);  // its internal __syncthreads also publishes scores
  const float inv = 1.0f / sum;

  // Weighted V sum: threads split head_dim.
  float* oh = out + (size_t)h * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int t = 0; t < n_pos; ++t) acc = fmaf(scores[t], vbase[(size_t)t * kv_stride + d], acc);
    oh[d] = acc * inv;
  }
}

}  // namespace

void attention(float* out, const float* q, const float* k, const float* v, int n_pos, int n_heads,
               int n_kv_heads, int head_dim, cudaStream_t stream) {
  if (n_heads <= 0 || head_dim <= 0 || n_pos <= 0) return;
  if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
    std::fprintf(stderr, "tinyllm::cuda::attention: n_heads (%d) must be a multiple of "
                         "n_kv_heads (%d)\n", n_heads, n_kv_heads);
    std::abort();
  }
  const size_t smem = ((size_t)head_dim + (size_t)n_pos) * sizeof(float);
  if (smem > 48 * 1024) {
    // Opt in to more than the default 48 KB of dynamic shared memory (sm_70+).
    int dev = 0;
    int max_optin = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    // Leave room for the static scratch[32].
    if (smem + 32 * sizeof(float) > (size_t)max_optin) {
      std::fprintf(stderr, "tinyllm::cuda::attention: n_pos=%d needs %zu B shared memory, "
                           "device allows %d B\n", n_pos, smem, max_optin);
      std::abort();
    }
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem));
  }
  const float scale = 1.0f / std::sqrt((float)head_dim);
  attention_kernel<<<n_heads, ATT_THREADS, smem, stream>>>(out, q, k, v, n_pos, n_heads,
                                                           n_kv_heads, head_dim, scale);
  CUDA_CHECK_LAUNCH();
}

}  // namespace cuda
}  // namespace tinyllm
