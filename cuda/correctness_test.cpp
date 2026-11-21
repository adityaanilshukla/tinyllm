// Correctness regression test for the hand-written CUDA kernels.
// Compares each kernel against the CPU reference ops in include/tinyllm/ops.h within a
// documented tolerance. No GoogleTest dependency: a plain main() that prints PASS/FAIL per
// case and returns nonzero if any case fails, so ctest can register it directly.
// Build with -DTINYLLM_CUDA=ON; run ./build-cuda/cuda/tinyllm-correctness-cuda
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ops_cuda.h"
#include "tinyllm/ops.h"
#include "tinyllm/tensor.h"

#define CUDA_CHECK(expr)                                                                  \
  do {                                                                                     \
    cudaError_t err__ = (expr);                                                            \
    if (err__ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__),        \
                   __FILE__, __LINE__, cudaGetErrorString(err__));                         \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

namespace tc = tinyllm::cuda;
using tc::DeviceBuffer;

namespace {

int g_failures = 0;

std::mt19937& rng() {
  static std::mt19937 g(42);
  return g;
}

std::vector<float> random_vec(size_t n, float lo = -1.0f, float hi = 1.0f) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng());
  return v;
}

DeviceBuffer upload(const std::vector<float>& h) {
  DeviceBuffer d(h.size());
  tc::copy_to_device(d.get(), h.data(), h.size());
  return d;
}

std::vector<float> download(const DeviceBuffer& d) {
  std::vector<float> h(d.size());
  tc::copy_to_host(h.data(), d.get(), d.size());
  return h;
}

float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float e = std::fabs(a[i] - b[i]);
    if (!(e <= m)) m = e;  // also propagates NaN
  }
  return m;
}

// Naive CPU GEMM, same reference used by bench_cuda.cpp: C[M,N] = A[M,K] W[N,K]^T + bias.
void cpu_gemm(float* C, const float* A, const float* W, const float* bias, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const float* a = A + (size_t)m * K;
    for (int n = 0; n < N; ++n) {
      const float* w = W + (size_t)n * K;
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      int k = 0;
      for (; k + 4 <= K; k += 4) {
        s0 += a[k] * w[k];
        s1 += a[k + 1] * w[k + 1];
        s2 += a[k + 2] * w[k + 2];
        s3 += a[k + 3] * w[k + 3];
      }
      for (; k < K; ++k) s0 += a[k] * w[k];
      C[(size_t)m * N + n] = (s0 + s1) + (s2 + s3) + (bias ? bias[n] : 0.0f);
    }
  }
}

void check(const char* name, float err, float tol) {
  const bool ok = std::isfinite(err) && err <= tol;
  std::printf("%-40s max_abs_err=%.3e tol=%.3e %s\n", name, err, tol, ok ? "PASS" : "FAIL");
  if (!ok) ++g_failures;
}

// ---- matmul: tiled GEMM path (M > 1) --------------------------------------------------

void test_matmul_tiled() {
  const int M = 37, N = 200, K = 300;  // deliberately not multiples of the tile size
  const auto hA = random_vec((size_t)M * K);
  const auto hW = random_vec((size_t)N * K);
  const auto hb = random_vec((size_t)N);
  DeviceBuffer dA = upload(hA), dW = upload(hW), db = upload(hb), dC((size_t)M * N);

  std::vector<float> hRef((size_t)M * N);
  cpu_gemm(hRef.data(), hA.data(), hW.data(), hb.data(), M, N, K);

  tc::matmul_tiled(dC.get(), dA.get(), dW.get(), db.get(), M, N, K);
  const float err = max_abs_err(download(dC), hRef);
  // fp32 accumulation order differs (register-tiled GEMM vs. naive 4-way unrolled CPU loop),
  // so this is a looser tolerance than the double-precision checks in tests/ops_test.cpp.
  check("matmul tiled (M=37,N=200,K=300)", err, 5e-4f);
}

// ---- matmul: matvec path (M == 1) -----------------------------------------------------

void test_matvec() {
  const int N = 768, K = 768;
  const auto hx = random_vec((size_t)K);
  const auto hW = random_vec((size_t)N * K);
  const auto hb = random_vec((size_t)N);
  DeviceBuffer dx = upload(hx), dW = upload(hW), db = upload(hb), dy((size_t)N);

  tinyllm::Tensor Wt;
  Wt.dtype = tinyllm::DType::F32;
  Wt.ndim = 2;
  Wt.shape[0] = N;
  Wt.shape[1] = K;
  Wt.data = hW.data();
  Wt.nbytes = (size_t)N * K * sizeof(float);

  std::vector<float> hRef((size_t)N);
  tinyllm::linear(hRef.data(), hx.data(), Wt, hb.data(), K, N);

  tc::matmul(dy.get(), dx.get(), dW.get(), db.get(), 1, N, K);
  const float err = max_abs_err(download(dy), hRef);
  check("matvec (N=768,K=768)", err, 2e-4f);
}

// ---- layernorm --------------------------------------------------------------------------

void test_layernorm() {
  const int rows = 9, D = 768;
  const float eps = 1e-5f;
  const auto hx = random_vec((size_t)rows * D, 2.0f, 6.0f);
  const auto hg = random_vec(D, 0.5f, 1.5f);
  const auto hb = random_vec(D);
  DeviceBuffer dx = upload(hx), dg = upload(hg), db = upload(hb), dy((size_t)rows * D);

  std::vector<float> hRef((size_t)rows * D);
  for (int r = 0; r < rows; ++r)
    tinyllm::layernorm(hRef.data() + (size_t)r * D, hx.data() + (size_t)r * D, hg.data(),
                       hb.data(), D, eps);

  tc::layernorm(dy.get(), dx.get(), dg.get(), db.get(), rows, D, eps);
  const float err = max_abs_err(download(dy), hRef);
  check("layernorm (rows=9,D=768)", err, 1e-4f);
}

// ---- softmax ----------------------------------------------------------------------------

void test_softmax() {
  const int rows = 5, n = 1024;
  const auto hx = random_vec((size_t)rows * n, -10.0f, 10.0f);
  DeviceBuffer dx = upload(hx), dy((size_t)rows * n);

  std::vector<float> hRef = hx;  // the CPU op is in place
  for (int r = 0; r < rows; ++r) tinyllm::softmax(hRef.data() + (size_t)r * n, n);

  tc::softmax(dy.get(), dx.get(), rows, n);
  const float err = max_abs_err(download(dy), hRef);
  check("softmax (rows=5,n=1024)", err, 1e-5f);
}

// ---- attention --------------------------------------------------------------------------

void test_attention() {
  // GQA shape (TinyLlama-like): 32 query heads, 4 KV heads, head_dim 64.
  const int n_heads = 32, n_kv_heads = 4, head_dim = 64, n_pos = 97;
  const size_t qn = (size_t)n_heads * head_dim;
  const size_t kvn = (size_t)n_pos * n_kv_heads * head_dim;
  const auto hq = random_vec(qn), hk = random_vec(kvn), hv = random_vec(kvn);
  DeviceBuffer dq = upload(hq), dk = upload(hk), dv = upload(hv), dout(qn);

  std::vector<float> hRef(qn), scratch((size_t)n_heads * n_pos);
  tinyllm::attention(hRef.data(), hq.data(), hk.data(), hv.data(), n_pos, n_heads, n_kv_heads,
                     head_dim, scratch.data());

  tc::attention(dout.get(), dq.get(), dk.get(), dv.get(), n_pos, n_heads, n_kv_heads, head_dim);
  const float err = max_abs_err(download(dout), hRef);
  check("attention GQA (32h/4kv,hd=64,n_pos=97)", err, 1e-4f);
}

}  // namespace

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  test_matmul_tiled();
  test_matvec();
  test_layernorm();
  test_softmax();
  test_attention();

  CUDA_CHECK(cudaDeviceSynchronize());
  if (g_failures > 0) {
    std::fprintf(stderr, "%d case(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all cases PASSED\n");
  return 0;
}
