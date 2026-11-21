// Tiled shared-memory GEMM and warp-per-row matvec.
//   C[M, N] = A[M, K] * W[N, K]^T (+ bias[N]), all row-major fp32.
#include <cstdint>

#include "common.cuh"
#include "ops_cuda.h"

namespace tinyllm {
namespace cuda {

namespace {

// ---- tiled GEMM -----------------------------------------------------------------------
// Block tile BM x BN, K step BK. 256 threads arranged 16 x 16; each thread owns a
// TM x TN = 4 x 4 register tile of C at rows ty*TM.., cols tx*TN...
// Both A and W are K-contiguous, so both tiles are loaded the same way and stored
// transposed in shared memory (As[k][m], Ws[k][n]) so the inner product loop reads
// contiguous rows. +4 padding on the inner dim reduces bank conflicts on the stores.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int GEMM_THREADS = (BM / TM) * (BN / TN);  // 256
constexpr int SPAD = 4;

static_assert(GEMM_THREADS == 256, "tile config");
static_assert((BM * BK) % GEMM_THREADS == 0, "A tile load must divide evenly");
static_assert((BN * BK) % GEMM_THREADS == 0, "W tile load must divide evenly");

__global__ void __launch_bounds__(GEMM_THREADS)
    gemm_tiled_kernel(float* __restrict__ C, const float* __restrict__ A,
                      const float* __restrict__ W, const float* __restrict__ bias, int M, int N,
                      int K) {
  __shared__ float As[BK][BM + SPAD];
  __shared__ float Ws[BK][BN + SPAD];

  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN);  // 0..15, column group
  const int ty = tid / (BN / TN);  // 0..15, row group
  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile [BM x BK]: element idx -> (row = idx / BK, col = idx % BK).
    // 16 consecutive threads read 16 consecutive floats of one row (coalesced 64B).
#pragma unroll
    for (int i = 0; i < (BM * BK) / GEMM_THREADS; ++i) {
      const int idx = tid + i * GEMM_THREADS;
      const int r = idx / BK;
      const int c = idx % BK;
      const int gm = block_m + r;
      const int gk = k0 + c;
      As[c][r] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
    }
    // Load W tile [BN x BK] the same way.
#pragma unroll
    for (int i = 0; i < (BN * BK) / GEMM_THREADS; ++i) {
      const int idx = tid + i * GEMM_THREADS;
      const int r = idx / BK;
      const int c = idx % BK;
      const int gn = block_n + r;
      const int gk = k0 + c;
      Ws[c][r] = (gn < N && gk < K) ? W[(size_t)gn * K + gk] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM];
      float b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Ws[kk][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();  // before the next iteration overwrites the tiles
  }

  // Epilogue: bounds-checked store with optional bias.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gm = block_m + ty * TM + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = block_n + tx * TN + j;
      if (gn >= N) continue;
      float v = acc[i][j];
      if (bias != nullptr) v += bias[gn];
      C[(size_t)gm * N + gn] = v;
    }
  }
}

// ---- matvec (M == 1) ------------------------------------------------------------------
// One warp per output row; each lane strides over K, then a shuffle reduce.
// VEC4 path requires K % 4 == 0 and 16-byte aligned x and W (checked on the host).
constexpr int MV_THREADS = 256;
constexpr int MV_ROWS_PER_BLOCK = MV_THREADS / 32;

template <bool VEC4>
__global__ void __launch_bounds__(MV_THREADS)
    matvec_kernel(float* __restrict__ y, const float* __restrict__ x,
                  const float* __restrict__ W, const float* __restrict__ bias, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * MV_ROWS_PER_BLOCK + warp;
  if (row >= N) return;  // whole warp exits together; no block-level sync below

  const float* w = W + (size_t)row * K;
  float sum = 0.0f;
  if (VEC4) {
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int K4 = K >> 2;
    for (int i = lane; i < K4; i += 32) {
      const float4 a = w4[i];
      const float4 b = x4[i];
      sum = fmaf(a.x, b.x, sum);
      sum = fmaf(a.y, b.y, sum);
      sum = fmaf(a.z, b.z, sum);
      sum = fmaf(a.w, b.w, sum);
    }
  } else {
    for (int i = lane; i < K; i += 32) sum = fmaf(w[i], x[i], sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) y[row] = (bias != nullptr) ? sum + bias[row] : sum;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

void matmul_tiled(float* C, const float* A, const float* W, const float* bias, int M, int N,
                  int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0) return;
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(GEMM_THREADS);
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(C, A, W, bias, M, N, K);
  CUDA_CHECK_LAUNCH();
}

void matvec(float* y, const float* x, const float* W, const float* bias, int N, int K,
            cudaStream_t stream) {
  if (N <= 0) return;
  dim3 grid((N + MV_ROWS_PER_BLOCK - 1) / MV_ROWS_PER_BLOCK);
  dim3 block(MV_THREADS);
  // Row starts are 16B-aligned iff W is aligned and K % 4 == 0.
  const bool vec4 = (K % 4 == 0) && aligned16(x) && aligned16(W);
  if (vec4) {
    matvec_kernel<true><<<grid, block, 0, stream>>>(y, x, W, bias, N, K);
  } else {
    matvec_kernel<false><<<grid, block, 0, stream>>>(y, x, W, bias, N, K);
  }
  CUDA_CHECK_LAUNCH();
}

void matmul(float* C, const float* A, const float* W, const float* bias, int M, int N, int K,
            cudaStream_t stream) {
  if (M == 1) {
    matvec(C, A, W, bias, N, K, stream);
  } else {
    matmul_tiled(C, A, W, bias, M, N, K, stream);
  }
}

// ---- buffer helpers -------------------------------------------------------------------
float* device_alloc(size_t n_floats) {
  if (n_floats == 0) return nullptr;
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n_floats * sizeof(float)));
  return static_cast<float*>(p);
}

void device_free(float* p) {
  if (p != nullptr) CUDA_CHECK(cudaFree(p));
}

void copy_to_device(float* dst, const float* src, size_t n_floats, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, n_floats * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void copy_to_host(float* dst, const float* src, size_t n_floats, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, n_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));  // pageable host memory: make the result visible
}

void synchronize(cudaStream_t stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }

}  // namespace cuda
}  // namespace tinyllm
