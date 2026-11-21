// Benchmark tinyllm's CUDA kernels against the CPU reference ops and cuBLAS.
// Build with -DTINYLLM_CUDA=ON; run ./build-cuda/cuda/tinyllm-bench-cuda
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "ops_cuda.h"
#include "tinyllm/ops.h"
#include "tinyllm/tensor.h"

#define CUDA_CHECK(expr)                                                                   \
  do {                                                                                     \
    cudaError_t err__ = (expr);                                                            \
    if (err__ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__),        \
                   __FILE__, __LINE__, cudaGetErrorString(err__));                         \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

#define CUBLAS_CHECK(expr)                                                                 \
  do {                                                                                     \
    cublasStatus_t st__ = (expr);                                                          \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                                   \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__, __FILE__, __LINE__);   \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

namespace tc = tinyllm::cuda;
using tc::DeviceBuffer;

namespace {

constexpr int kWarmup = 3;
constexpr int kIters = 20;

std::mt19937& rng() {
  static std::mt19937 g(1234);
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

// Average milliseconds per call of a GPU launch sequence, measured with cudaEvents.
// warmup/iters default to the module-wide constants; large square matmuls override
// them with smaller counts so a single huge GEMM doesn't dominate the whole bench.
double time_gpu(const std::function<void()>& fn, int warmup = kWarmup, int iters = kIters) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / iters;
}

// Milliseconds per call of a CPU function over `iters` repetitions (plus a warmup if iters > 1).
// Returns a negative value if the CPU op threw (e.g. not implemented yet).
double time_cpu(const std::function<void()>& fn, int iters) {
  try {
    if (iters > 1) fn();  // warmup (skipped for slow single-shot references)
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "  [cpu reference unavailable: %s]\n", e.what());
    return -1.0;
  }
}

void print_header(const char* unit) {
  std::printf("%-34s %-8s %10s %12s %14s %14s\n", "case", "impl", "ms", unit, "err_vs_cublas",
              "err_vs_cpu");
  std::printf("%s\n", std::string(97, '-').c_str());
}

void print_row(const std::string& name, const char* impl, double ms, double rate, float err_blas,
               float err_cpu) {
  char e1[32], e2[32], t[32], r[32];
  if (ms < 0) {
    std::snprintf(t, sizeof t, "n/a");
    std::snprintf(r, sizeof r, "n/a");
  } else {
    std::snprintf(t, sizeof t, "%.4f", ms);
    std::snprintf(r, sizeof r, "%.1f", rate);
  }
  if (err_blas < 0) std::snprintf(e1, sizeof e1, "-");
  else std::snprintf(e1, sizeof e1, "%.3e", err_blas);
  if (err_cpu < 0) std::snprintf(e2, sizeof e2, "-");
  else std::snprintf(e2, sizeof e2, "%.3e", err_cpu);
  std::printf("%-34s %-8s %10s %12s %14s %14s\n", name.c_str(), impl, t, r, e1, e2);
}

// ---- matmul -------------------------------------------------------------------------

// Naive CPU GEMM: C[M,N] = A[M,K] W[N,K]^T + bias. Four partial sums per dot product.
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

// Row-major C[M,N] = A[M,K] W[N,K]^T via column-major cuBLAS:
// col-major C^T [N x M] = (W^T)^T [N x K] * A^T [K x M], where the row-major W and A
// buffers are exactly col-major W^T (K x N, ld K) and A^T (K x M, ld K).
void cublas_gemm(cublasHandle_t h, float* C, const float* A, const float* W, int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, A, K, &beta, C, N));
}

void bench_matmul(cublasHandle_t blas) {
  struct Shape {
    const char* name;
    int N, K;
  };
  const Shape shapes[] = {
      {"gpt2 attn.proj 768x768", 768, 768},
      {"gpt2 mlp.fc 3072x768", 3072, 768},
      {"gpt2 mlp.proj 768x3072", 768, 3072},
      {"llama ffn.up 5632x2048", 5632, 2048},
      {"llama ffn.down 2048x5632", 2048, 5632},
  };
  const int Ms[] = {1, 64, 512};

  std::printf("\n== matmul: C[M,N] = A[M,K] W[N,K]^T  (rate = GFLOP/s) ==\n");
  print_header("GFLOP/s");
  for (const Shape& s : shapes) {
    const int N = s.N, K = s.K;
    const auto hW = random_vec((size_t)N * K);
    const auto hb = random_vec((size_t)N);
    DeviceBuffer dW = upload(hW);
    DeviceBuffer db = upload(hb);
    tinyllm::Tensor Wt;
    Wt.dtype = tinyllm::DType::F32;
    Wt.ndim = 2;
    Wt.shape[0] = N;
    Wt.shape[1] = K;
    Wt.data = hW.data();
    Wt.nbytes = (size_t)N * K * sizeof(float);

    for (int M : Ms) {
      const auto hA = random_vec((size_t)M * K);
      DeviceBuffer dA = upload(hA);
      DeviceBuffer dC((size_t)M * N);
      DeviceBuffer dRef((size_t)M * N);
      const double gflop = 2.0 * M * N * K * 1e-9;
      char nm[96];
      std::snprintf(nm, sizeof nm, "%s M=%d", s.name, M);

      // CPU reference (with bias). linear() for M=1, naive loop otherwise.
      std::vector<float> hCpu((size_t)M * N);
      double cpu_ms;
      if (M == 1) {
        cpu_ms = time_cpu([&] { tinyllm::linear(hCpu.data(), hA.data(), Wt, hb.data(), K, N); },
                          10);
      } else {
        cpu_ms = time_cpu([&] { cpu_gemm(hCpu.data(), hA.data(), hW.data(), hb.data(), M, N, K); },
                          1);
      }
      const bool have_cpu = cpu_ms >= 0;

      // cuBLAS (no bias).
      const double blas_ms = time_gpu([&] { cublas_gemm(blas, dRef.get(), dA.get(), dW.get(), M, N, K); });
      const auto hBlas = download(dRef);

      // Ours: correctness without bias vs cuBLAS, with bias vs CPU.
      tc::matmul(dC.get(), dA.get(), dW.get(), nullptr, M, N, K);
      const float err_blas = max_abs_err(download(dC), hBlas);
      tc::matmul(dC.get(), dA.get(), dW.get(), db.get(), M, N, K);
      const float err_cpu = have_cpu ? max_abs_err(download(dC), hCpu) : -1.0f;
      const double ours_ms = time_gpu([&] { tc::matmul(dC.get(), dA.get(), dW.get(), db.get(), M, N, K); });

      print_row(nm, "cpu", cpu_ms, have_cpu ? gflop / (cpu_ms * 1e-3) : 0, -1, -1);
      print_row(nm, "cublas", blas_ms, gflop / (blas_ms * 1e-3), -1, -1);
      print_row(nm, M == 1 ? "matvec" : "tiled", ours_ms, gflop / (ours_ms * 1e-3), err_blas,
                err_cpu);
      if (M == 1) {
        // Also report the tiled kernel at M=1 for comparison.
        tc::matmul_tiled(dC.get(), dA.get(), dW.get(), nullptr, M, N, K);
        const float e = max_abs_err(download(dC), hBlas);
        const double t = time_gpu([&] { tc::matmul_tiled(dC.get(), dA.get(), dW.get(), nullptr, M, N, K); });
        print_row(nm, "tiled", t, gflop / (t * 1e-3), e, -1);
      }
    }
  }
}

// Square GEMM cases, M = N = K, so the project can state a clean "tiled kernel reaches
// X% of cuBLAS at N x N" number without the model-shaped cases' uneven dimensions.
// The naive CPU GEMM above is ~5 GFLOP/s, so a 2048^3 CPU run is already a few seconds
// and a 4096^3 one would be tens of seconds; the CPU reference is only run at N=1024
// and skipped above that so the whole bench still finishes quickly. cuBLAS and the
// tiled kernel run at every size, with fewer timed iterations at the largest sizes
// (still enough to average out noise) so a single huge GEMM can't dominate the runtime.
void bench_matmul_square(cublasHandle_t blas) {
  struct Case {
    int n;
    bool with_cpu;
    int warmup, iters;
  };
  const Case cases[] = {
      {1024, true, kWarmup, kIters},
      {2048, false, kWarmup, kIters},
      {4096, false, 3, 10},
      {8192, false, 2, 10},
  };

  std::printf("\n== matmul: square C[N,N] = A[N,N] W[N,N]^T, M=N=K  (rate = GFLOP/s) ==\n");
  print_header("GFLOP/s");
  for (const Case& c : cases) {
    const int M = c.n, N = c.n, K = c.n;
    const auto hA = random_vec((size_t)M * K);
    const auto hW = random_vec((size_t)N * K);
    DeviceBuffer dA = upload(hA);
    DeviceBuffer dW = upload(hW);
    DeviceBuffer dC((size_t)M * N);
    DeviceBuffer dRef((size_t)M * N);
    const double gflop = 2.0 * (double)M * N * K * 1e-9;
    char nm[96];
    std::snprintf(nm, sizeof nm, "square %dx%dx%d", M, N, K);

    // CPU reference, only for the smallest size (see comment above).
    std::vector<float> hCpu;
    double cpu_ms = -1.0;
    if (c.with_cpu) {
      hCpu.resize((size_t)M * N);
      cpu_ms = time_cpu([&] { cpu_gemm(hCpu.data(), hA.data(), hW.data(), nullptr, M, N, K); }, 1);
    }
    const bool have_cpu = cpu_ms >= 0;

    // cuBLAS reference.
    const double blas_ms =
        time_gpu([&] { cublas_gemm(blas, dRef.get(), dA.get(), dW.get(), M, N, K); }, c.warmup,
                 c.iters);
    const auto hBlas = download(dRef);

    // Hand-written tiled kernel, forced explicitly (not the M==1 matvec dispatch path).
    tc::matmul_tiled(dC.get(), dA.get(), dW.get(), nullptr, M, N, K);
    const float err_blas = max_abs_err(download(dC), hBlas);
    const float err_cpu = have_cpu ? max_abs_err(download(dC), hCpu) : -1.0f;
    const double ours_ms = time_gpu(
        [&] { tc::matmul_tiled(dC.get(), dA.get(), dW.get(), nullptr, M, N, K); }, c.warmup,
        c.iters);

    if (have_cpu) {
      print_row(nm, "cpu", cpu_ms, gflop / (cpu_ms * 1e-3), -1, -1);
    }
    print_row(nm, "cublas", blas_ms, gflop / (blas_ms * 1e-3), -1, -1);
    print_row(nm, "tiled", ours_ms, gflop / (ours_ms * 1e-3), err_blas, err_cpu);
  }
}

// ---- layernorm ----------------------------------------------------------------------

void bench_layernorm() {
  struct Case {
    int rows, D;
  };
  const Case cases[] = {{1, 768}, {512, 768}, {1, 2048}, {512, 2048}};
  const float eps = 1e-5f;
  std::printf("\n== layernorm (rate = GB/s, read x + write y) ==\n");
  print_header("GB/s");
  for (const Case& c : cases) {
    const size_t n = (size_t)c.rows * c.D;
    // Offset mean to exercise the variance computation.
    const auto hx = random_vec(n, 2.0f, 6.0f);
    const auto hg = random_vec(c.D, 0.5f, 1.5f);
    const auto hb = random_vec(c.D);
    DeviceBuffer dx = upload(hx), dg = upload(hg), db = upload(hb), dy(n);
    const double gb = 2.0 * n * sizeof(float) * 1e-9;
    char nm[96];
    std::snprintf(nm, sizeof nm, "layernorm rows=%d D=%d", c.rows, c.D);

    std::vector<float> hy(n);
    const double cpu_ms = time_cpu(
        [&] {
          for (int r = 0; r < c.rows; ++r)
            tinyllm::layernorm(hy.data() + (size_t)r * c.D, hx.data() + (size_t)r * c.D,
                               hg.data(), hb.data(), c.D, eps);
        },
        5);
    tc::layernorm(dy.get(), dx.get(), dg.get(), db.get(), c.rows, c.D, eps);
    const float err = cpu_ms >= 0 ? max_abs_err(download(dy), hy) : -1.0f;
    const double ms =
        time_gpu([&] { tc::layernorm(dy.get(), dx.get(), dg.get(), db.get(), c.rows, c.D, eps); });
    print_row(nm, "cpu", cpu_ms, cpu_ms >= 0 ? gb / (cpu_ms * 1e-3) : 0, -1, -1);
    print_row(nm, "cuda", ms, gb / (ms * 1e-3), -1, err);
  }
}

// ---- softmax ------------------------------------------------------------------------

void bench_softmax() {
  struct Case {
    int rows, n;
  };
  const Case cases[] = {{12, 1024}, {32, 4096}, {1, 50257}, {512, 1024}};
  std::printf("\n== softmax (rate = GB/s, read x + write y) ==\n");
  print_header("GB/s");
  for (const Case& c : cases) {
    const size_t n = (size_t)c.rows * c.n;
    const auto hx = random_vec(n, -10.0f, 10.0f);
    DeviceBuffer dx = upload(hx), dy(n);
    const double gb = 2.0 * n * sizeof(float) * 1e-9;
    char nm[96];
    std::snprintf(nm, sizeof nm, "softmax rows=%d n=%d", c.rows, c.n);

    std::vector<float> hy(n);
    const double cpu_ms = time_cpu(
        [&] {
          hy = hx;  // CPU op is in place
          for (int r = 0; r < c.rows; ++r) tinyllm::softmax(hy.data() + (size_t)r * c.n, c.n);
        },
        5);
    tc::softmax(dy.get(), dx.get(), c.rows, c.n);
    const float err = cpu_ms >= 0 ? max_abs_err(download(dy), hy) : -1.0f;
    const double ms = time_gpu([&] { tc::softmax(dy.get(), dx.get(), c.rows, c.n); });
    print_row(nm, "cpu", cpu_ms, cpu_ms >= 0 ? gb / (cpu_ms * 1e-3) : 0, -1, -1);
    print_row(nm, "cuda", ms, gb / (ms * 1e-3), -1, err);
  }
}

// ---- attention ----------------------------------------------------------------------

void bench_attention() {
  struct Case {
    const char* name;
    int n_heads, n_kv_heads, head_dim, n_pos;
  };
  const Case cases[] = {
      {"gpt2 12h hd64", 12, 12, 64, 128},
      {"gpt2 12h hd64", 12, 12, 64, 1024},
      {"tinyllama 32h/4kv hd64", 32, 4, 64, 2048},
      {"llama7b 32h hd128", 32, 32, 128, 4096},
  };
  std::printf("\n== attention, one query token (rate = GB/s of K+V cache read) ==\n");
  print_header("GB/s");
  for (const Case& c : cases) {
    const size_t qn = (size_t)c.n_heads * c.head_dim;
    const size_t kvn = (size_t)c.n_pos * c.n_kv_heads * c.head_dim;
    const auto hq = random_vec(qn), hk = random_vec(kvn), hv = random_vec(kvn);
    DeviceBuffer dq = upload(hq), dk = upload(hk), dv = upload(hv), dout(qn);
    const double gb = 2.0 * kvn * sizeof(float) * 1e-9;
    char nm[96];
    std::snprintf(nm, sizeof nm, "%s n_pos=%d", c.name, c.n_pos);

    std::vector<float> hout(qn), scratch((size_t)c.n_heads * c.n_pos);
    const double cpu_ms = time_cpu(
        [&] {
          tinyllm::attention(hout.data(), hq.data(), hk.data(), hv.data(), c.n_pos, c.n_heads,
                             c.n_kv_heads, c.head_dim, scratch.data());
        },
        5);
    tc::attention(dout.get(), dq.get(), dk.get(), dv.get(), c.n_pos, c.n_heads, c.n_kv_heads,
                  c.head_dim);
    const float err = cpu_ms >= 0 ? max_abs_err(download(dout), hout) : -1.0f;
    const double ms = time_gpu([&] {
      tc::attention(dout.get(), dq.get(), dk.get(), dv.get(), c.n_pos, c.n_heads, c.n_kv_heads,
                    c.head_dim);
    });
    print_row(nm, "cpu", cpu_ms, cpu_ms >= 0 ? gb / (cpu_ms * 1e-3) : 0, -1, -1);
    print_row(nm, "cuda", ms, gb / (ms * 1e-3), -1, err);
  }
}

}  // namespace

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("timing: %d warmup + %d timed iterations (cudaEvent); errors are max |diff|\n",
              kWarmup, kIters);

  cublasHandle_t blas;
  CUBLAS_CHECK(cublasCreate(&blas));
  // Plain fp32 (no TF32) so the comparison against our fp32 kernels is apples to apples.
  CUBLAS_CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));

  bench_matmul(blas);
  bench_matmul_square(blas);
  bench_layernorm();
  bench_softmax();
  bench_attention();

  CUBLAS_CHECK(cublasDestroy(blas));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
