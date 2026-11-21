#pragma once
// Host API for tinyllm's CUDA kernels. Mirrors include/tinyllm/ops.h, but every pointer
// is a DEVICE pointer (fp32) and every call is asynchronous on the given stream.
// Plain C++ header: includable from host-only .cpp files compiled by the C++ compiler.
#include <cuda_runtime_api.h>

#include <cstddef>

namespace tinyllm {
namespace cuda {

// ---- GEMM / linear -------------------------------------------------------------------
// C[M, N] = A[M, K] * W[N, K]^T (+ bias[N] broadcast over rows if bias != nullptr).
// All row-major. W uses the same [out, in] layout as CPU weights, so this is a batched
// linear layer. Dispatches to matvec when M == 1, otherwise to the tiled GEMM.
void matmul(float* C, const float* A, const float* W, const float* bias, int M, int N, int K,
            cudaStream_t stream = 0);

// Force a specific kernel (used by the benchmark).
void matmul_tiled(float* C, const float* A, const float* W, const float* bias, int M, int N,
                  int K, cudaStream_t stream = 0);
// y[N] = W[N, K] x[K] (+ bias). Warp per output row.
void matvec(float* y, const float* x, const float* W, const float* bias, int N, int K,
            cudaStream_t stream = 0);

// ---- normalization / softmax ---------------------------------------------------------
// Row-wise layer norm over `rows` rows of length D. y may alias x.
void layernorm(float* y, const float* x, const float* gamma, const float* beta, int rows, int D,
               float eps, cudaStream_t stream = 0);

// Row-wise numerically stable softmax over `rows` rows of length n. y may alias x
// (pass the same pointer for in-place, like the CPU op).
void softmax(float* y, const float* x, int rows, int n, cudaStream_t stream = 0);

// ---- attention -----------------------------------------------------------------------
// Causal attention for one query token at position n_pos - 1. Same semantics as
// tinyllm::attention in ops.h (no scores scratch needed):
//   q:   [n_heads * head_dim]
//   k,v: KV cache for this layer, n_pos rows, row stride n_kv_heads * head_dim
//   out: [n_heads * head_dim]
// Query head h reads KV head h / (n_heads / n_kv_heads). Scale 1/sqrt(head_dim).
void attention(float* out, const float* q, const float* k, const float* v, int n_pos, int n_heads,
               int n_kv_heads, int head_dim, cudaStream_t stream = 0);

// ---- device buffer helpers -----------------------------------------------------------
float* device_alloc(size_t n_floats);
void device_free(float* p);
void copy_to_device(float* dst, const float* src, size_t n_floats, cudaStream_t stream = 0);
void copy_to_host(float* dst, const float* src, size_t n_floats, cudaStream_t stream = 0);
void synchronize(cudaStream_t stream = 0);

// RAII owner of a device float array. Move-only.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) : ptr_(device_alloc(n)), n_(n) {}
  ~DeviceBuffer() { device_free(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
    o.ptr_ = nullptr;
    o.n_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      device_free(ptr_);
      ptr_ = o.ptr_;
      n_ = o.n_;
      o.ptr_ = nullptr;
      o.n_ = 0;
    }
    return *this;
  }
  float* get() const { return ptr_; }
  size_t size() const { return n_; }

 private:
  float* ptr_ = nullptr;
  size_t n_ = 0;
};

}  // namespace cuda
}  // namespace tinyllm
