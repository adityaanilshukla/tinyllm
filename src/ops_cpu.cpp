// CPU ops for single-token decoding: matvec (F32 NEON / quantized dispatch),
// layernorm, gelu, softmax, add, and GQA-aware attention.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TINYLLM_NEON 1
#endif

#include "tinyllm/ops.h"
#include "tinyllm/quant.h"

namespace tinyllm {
namespace {

// ---------------------------------------------------------------------------
// Persistent thread pool. parallel_for(n, fn) splits [0, n) into chunks that the
// workers and the calling thread pull from an atomic counter. One job at a time;
// calls from multiple threads are serialized by a mutex.
class ThreadPool {
 public:
  ThreadPool() {
    unsigned hc = std::thread::hardware_concurrency();
    n_threads_ = static_cast<int>(hc ? hc : 4);
    if (const char* env = std::getenv("TINYLLM_THREADS")) {
      int v = std::atoi(env);
      if (v > 0) n_threads_ = v;
    }
    for (int i = 1; i < n_threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      generation_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }

  int size() const { return n_threads_; }

  // fn(begin, end) over [0, n) in chunks of `grain` items.
  void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    int64_t n_chunks = (n + grain - 1) / grain;
    if (n_threads_ <= 1 || n_chunks <= 1) {
      fn(0, n);
      return;
    }
    std::lock_guard<std::mutex> job_lk(job_mu_);
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = &fn;
      n_ = n;
      grain_ = grain;
      n_chunks_ = n_chunks;
      next_.store(0, std::memory_order_relaxed);
      done_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    run_chunks();
    // Wait for all chunks to finish (spin briefly, then yield).
    int spins = 0;
    while (done_.load(std::memory_order_acquire) < n_chunks) {
      if (++spins > 4000) std::this_thread::yield();
    }
    // Close the job so no late worker joins, then wait for in-flight workers to leave
    // run_chunks() before the job state can be reused.
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = nullptr;
    }
    while (active_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
  }

 private:
  void run_chunks() {
    for (;;) {
      int64_t c = next_.fetch_add(1, std::memory_order_relaxed);
      if (c >= n_chunks_) break;
      int64_t b = c * grain_;
      int64_t e = std::min(n_, b + grain_);
      (*fn_)(b, e);
      done_.fetch_add(1, std::memory_order_release);
    }
  }

  void worker_loop() {
    uint64_t seen = 0;
    for (;;) {
      // Spin a little before sleeping: matvec calls come back-to-back during decoding.
      for (int i = 0; i < 20000 && generation_.load(std::memory_order_acquire) == seen; ++i) {
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
      }
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return generation_.load(std::memory_order_relaxed) != seen; });
      seen = generation_.load(std::memory_order_relaxed);
      if (stop_) return;
      if (!fn_) continue;
      active_.fetch_add(1, std::memory_order_relaxed);
      lk.unlock();
      run_chunks();
      active_.fetch_sub(1, std::memory_order_release);
    }
  }

  int n_threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex job_mu_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<int> active_{0};
  bool stop_ = false;
  const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t n_ = 0, grain_ = 1, n_chunks_ = 0;
  std::atomic<int64_t> next_{0};
  std::atomic<int64_t> done_{0};
};

ThreadPool& pool() {
  static ThreadPool p;
  return p;
}

// ---------------------------------------------------------------------------
inline float dot_f32(const float* a, const float* b, int n) {
#ifdef TINYLLM_NEON
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#else
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#endif
}

// y += a * x
inline void axpy_f32(float* y, float a, const float* x, int n) {
#ifdef TINYLLM_NEON
  float32x4_t va = vdupq_n_f32(a);
  int i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
  for (; i < n; ++i) y[i] += a * x[i];
#else
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
#endif
}

void matvec_f32_rows(float* y, const float* W, const float* x, int64_t r0, int64_t r1, int K) {
  int64_t r = r0;
  // 4 rows at a time to reuse loads of x.
#ifdef TINYLLM_NEON
  for (; r + 4 <= r1; r += 4) {
    const float* w0 = W + r * K;
    const float* w1 = w0 + K;
    const float* w2 = w1 + K;
    const float* w3 = w2 + K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= K; i += 8) {
      float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
      a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), x0);
      b0 = vfmaq_f32(b0, vld1q_f32(w0 + i + 4), x1);
      a1 = vfmaq_f32(a1, vld1q_f32(w1 + i), x0);
      b1 = vfmaq_f32(b1, vld1q_f32(w1 + i + 4), x1);
      a2 = vfmaq_f32(a2, vld1q_f32(w2 + i), x0);
      b2 = vfmaq_f32(b2, vld1q_f32(w2 + i + 4), x1);
      a3 = vfmaq_f32(a3, vld1q_f32(w3 + i), x0);
      b3 = vfmaq_f32(b3, vld1q_f32(w3 + i + 4), x1);
    }
    float s0 = vaddvq_f32(vaddq_f32(a0, b0));
    float s1 = vaddvq_f32(vaddq_f32(a1, b1));
    float s2 = vaddvq_f32(vaddq_f32(a2, b2));
    float s3 = vaddvq_f32(vaddq_f32(a3, b3));
    for (; i < K; ++i) {
      s0 += w0[i] * x[i];
      s1 += w1[i] * x[i];
      s2 += w2[i] * x[i];
      s3 += w3[i] * x[i];
    }
    y[r] = s0;
    y[r + 1] = s1;
    y[r + 2] = s2;
    y[r + 3] = s3;
  }
#endif
  for (; r < r1; ++r) y[r] = dot_f32(W + r * K, x, K);
}

}  // namespace

// ---------------------------------------------------------------------------
void linear(float* y, const float* x, const Tensor& W, const float* bias, int K, int N) {
  if (W.ndim != 2 || W.shape[0] != N || W.shape[1] != K) {
    throw std::runtime_error("linear: weight shape mismatch (expected [" + std::to_string(N) + ", " +
                             std::to_string(K) + "])");
  }
  const int64_t work = static_cast<int64_t>(N) * K;
  ThreadPool& tp = pool();
  const bool par = work >= (1 << 16) && tp.size() > 1;
  // Grain: aim for several chunks per thread for load balance across P/E cores,
  // rounded to a multiple of 4 rows.
  int64_t grain = N;
  if (par) {
    grain = std::max<int64_t>(4, (N + tp.size() * 4 - 1) / (tp.size() * 4));
    grain = (grain + 3) & ~int64_t(3);
  }

  switch (W.dtype) {
    case DType::F32: {
      const float* w = W.f32();
      if (par) {
        tp.parallel_for(N, grain, [&](int64_t r0, int64_t r1) { matvec_f32_rows(y, w, x, r0, r1, K); });
      } else {
        matvec_f32_rows(y, w, x, 0, N, K);
      }
      break;
    }
    // Quantized kernels (src/quant.cpp) quantize x once and parallelize internally,
    // so call them on the whole matrix rather than nesting our pool.
    case DType::Q8:
      matvec_q8(y, W.data, x, N, K);
      break;
    case DType::Q4:
      matvec_q4(y, W.data, x, N, K);
      break;
    default:
      throw std::runtime_error("linear: unsupported dtype");
  }

  if (bias) add_inplace(y, bias, N);
}

void layernorm(float* y, const float* x, const float* gamma, const float* beta, int D, float eps) {
  double sum = 0.0;
  for (int i = 0; i < D; ++i) sum += x[i];
  const float mean = static_cast<float>(sum / D);
  double var = 0.0;
  for (int i = 0; i < D; ++i) {
    float d = x[i] - mean;
    var += static_cast<double>(d) * d;
  }
  const float inv = 1.0f / std::sqrt(static_cast<float>(var / D) + eps);
  for (int i = 0; i < D; ++i) {
    float n = (x[i] - mean) * inv;
    y[i] = n * (gamma ? gamma[i] : 1.0f) + (beta ? beta[i] : 0.0f);
  }
}

void gelu(float* x, int64_t n) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  for (int64_t i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
  }
}

void softmax(float* x, int n) {
  if (n <= 0) return;
  float mx = x[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

void add_inplace(float* y, const float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += x[i];
}

void attention(float* out, const float* q, const float* k, const float* v, int n_pos, int n_heads,
               int n_kv_heads, int head_dim, float* scores) {
  if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) throw std::runtime_error("attention: bad n_kv_heads");
  const int group = n_heads / n_kv_heads;
  const int64_t kv_stride = static_cast<int64_t>(n_kv_heads) * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto do_heads = [&](int64_t h0, int64_t h1) {
    for (int64_t h = h0; h < h1; ++h) {
      const int kvh = static_cast<int>(h) / group;
      const float* qh = q + h * head_dim;
      float* sc = scores + h * n_pos;
      for (int t = 0; t < n_pos; ++t) sc[t] = dot_f32(qh, k + t * kv_stride + kvh * head_dim, head_dim) * scale;
      softmax(sc, n_pos);
      float* oh = out + h * head_dim;
      std::memset(oh, 0, sizeof(float) * head_dim);
      for (int t = 0; t < n_pos; ++t) axpy_f32(oh, sc[t], v + t * kv_stride + kvh * head_dim, head_dim);
    }
  };

  const int64_t work = static_cast<int64_t>(n_heads) * n_pos * head_dim;
  ThreadPool& tp = pool();
  if (work >= (1 << 15) && tp.size() > 1 && n_heads > 1)
    tp.parallel_for(n_heads, 1, do_heads);
  else
    do_heads(0, n_heads);
}

}  // namespace tinyllm
