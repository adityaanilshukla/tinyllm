// Block quantization (Q8 / Q4, 32 values per block) and quantized matvec kernels.
//
// matvec: x is quantized once per call to int8 blocks of 32 (q8_0-style
// activation quantization), then each output row is an int8 dot product per
// block scaled by d_w * d_x. Rows are split across a small persistent thread
// pool owned by this file.

#include "tinyllm/quant.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
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

namespace tinyllm {

namespace {

// ---------------------------------------------------------------------------
// Thread pool: persistent workers, caller participates. Work is a range [0, n)
// split into contiguous chunks that threads grab via an atomic counter.
class ThreadPool {
 public:
  ThreadPool() {
    unsigned hc = std::thread::hardware_concurrency();
    int n = hc == 0 ? 4 : static_cast<int>(hc);
    n = std::min(n, 32);
    for (int i = 0; i < n - 1; ++i) workers_.emplace_back([this] { worker_loop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }

  int num_threads() const { return static_cast<int>(workers_.size()) + 1; }

  // Runs fn(begin, end) over [0, n) in chunks. Falls back to running inline
  // when the pool is already busy (re-entrant or concurrent callers).
  void parallel_for(int64_t n, int64_t chunk, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    std::unique_lock<std::mutex> submit(submit_mu_, std::try_to_lock);
    if (workers_.empty() || n <= chunk || !submit.owns_lock()) {
      fn(0, n);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = &fn;
      n_ = n;
      chunk_ = chunk;
      next_.store(0, std::memory_order_relaxed);
      active_ = static_cast<int>(workers_.size());
      ++gen_;
    }
    cv_.notify_all();
    run_chunks();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return active_ == 0; });
    fn_ = nullptr;
  }

 private:
  void run_chunks() {
    const auto& fn = *fn_;
    for (;;) {
      int64_t b = next_.fetch_add(chunk_, std::memory_order_relaxed);
      if (b >= n_) break;
      fn(b, std::min(n_, b + chunk_));
    }
  }

  void worker_loop() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
      }
      run_chunks();
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex submit_mu_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  uint64_t gen_ = 0;
  bool stop_ = false;
  int active_ = 0;
  const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t n_ = 0, chunk_ = 1;
  std::atomic<int64_t> next_{0};
};

ThreadPool& pool() {
  static ThreadPool p;
  return p;
}

void check_k(int64_t n) {
  if (n % kQK != 0)
    throw std::runtime_error("quant: row length " + std::to_string(n) + " not a multiple of 32");
}

// ---------------------------------------------------------------------------
// Activation quantization: same as Q8 weights.
void quantize_x(const float* x, std::vector<BlockQ8>& xq, int64_t K) {
  xq.resize(static_cast<size_t>(K / kQK));
  quantize_row_q8(x, xq.data(), K);
}

// ---------------------------------------------------------------------------
// Per-row dot products.

#ifdef TINYLLM_NEON
inline int32x4_t dot_s8(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
  return vdotq_s32(acc, a, b);
#else
  int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
  int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
  acc = vpadalq_s16(acc, lo);
  return vpadalq_s16(acc, hi);
#endif
}

float row_dot_q8(const BlockQ8* w, const BlockQ8* x, int64_t nb) {
  float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
  int64_t b = 0;
  for (; b + 1 < nb; b += 2) {
    const BlockQ8& w0 = w[b];
    const BlockQ8& w1 = w[b + 1];
    const BlockQ8& x0 = x[b];
    const BlockQ8& x1 = x[b + 1];
    int32x4_t p0 = vdupq_n_s32(0), p1 = vdupq_n_s32(0);
    p0 = dot_s8(p0, vld1q_s8(w0.qs), vld1q_s8(x0.qs));
    p0 = dot_s8(p0, vld1q_s8(w0.qs + 16), vld1q_s8(x0.qs + 16));
    p1 = dot_s8(p1, vld1q_s8(w1.qs), vld1q_s8(x1.qs));
    p1 = dot_s8(p1, vld1q_s8(w1.qs + 16), vld1q_s8(x1.qs + 16));
    s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(p0), w0.d * x0.d);
    s1 = vmlaq_n_f32(s1, vcvtq_f32_s32(p1), w1.d * x1.d);
  }
  for (; b < nb; ++b) {
    int32x4_t p = vdupq_n_s32(0);
    p = dot_s8(p, vld1q_s8(w[b].qs), vld1q_s8(x[b].qs));
    p = dot_s8(p, vld1q_s8(w[b].qs + 16), vld1q_s8(x[b].qs + 16));
    s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(p), w[b].d * x[b].d);
  }
  return vaddvq_f32(vaddq_f32(s0, s1));
}

inline void unpack_q4(const uint8_t* qs, int8x16_t& lo, int8x16_t& hi) {
  const uint8x16_t v = vld1q_u8(qs);
  const int8x16_t eight = vdupq_n_s8(8);
  lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F))), eight);
  hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
}

float row_dot_q4(const BlockQ4* w, const BlockQ8* x, int64_t nb) {
  float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
  int64_t b = 0;
  for (; b + 1 < nb; b += 2) {
    int8x16_t l0, h0, l1, h1;
    unpack_q4(w[b].qs, l0, h0);
    unpack_q4(w[b + 1].qs, l1, h1);
    int32x4_t p0 = vdupq_n_s32(0), p1 = vdupq_n_s32(0);
    p0 = dot_s8(p0, l0, vld1q_s8(x[b].qs));
    p0 = dot_s8(p0, h0, vld1q_s8(x[b].qs + 16));
    p1 = dot_s8(p1, l1, vld1q_s8(x[b + 1].qs));
    p1 = dot_s8(p1, h1, vld1q_s8(x[b + 1].qs + 16));
    s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(p0), w[b].d * x[b].d);
    s1 = vmlaq_n_f32(s1, vcvtq_f32_s32(p1), w[b + 1].d * x[b + 1].d);
  }
  for (; b < nb; ++b) {
    int8x16_t l, h;
    unpack_q4(w[b].qs, l, h);
    int32x4_t p = vdupq_n_s32(0);
    p = dot_s8(p, l, vld1q_s8(x[b].qs));
    p = dot_s8(p, h, vld1q_s8(x[b].qs + 16));
    s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(p), w[b].d * x[b].d);
  }
  return vaddvq_f32(vaddq_f32(s0, s1));
}

#else  // scalar fallback

float row_dot_q8(const BlockQ8* w, const BlockQ8* x, int64_t nb) {
  float sum = 0.f;
  for (int64_t b = 0; b < nb; ++b) {
    int32_t s = 0;
    for (int j = 0; j < kQK; ++j) s += int32_t(w[b].qs[j]) * int32_t(x[b].qs[j]);
    sum += float(s) * w[b].d * x[b].d;
  }
  return sum;
}

float row_dot_q4(const BlockQ4* w, const BlockQ8* x, int64_t nb) {
  float sum = 0.f;
  for (int64_t b = 0; b < nb; ++b) {
    int32_t s = 0;
    for (int j = 0; j < kQK / 2; ++j) {
      int lo = (w[b].qs[j] & 0x0F) - 8;
      int hi = (w[b].qs[j] >> 4) - 8;
      s += lo * int32_t(x[b].qs[j]) + hi * int32_t(x[b].qs[j + kQK / 2]);
    }
    sum += float(s) * w[b].d * x[b].d;
  }
  return sum;
}

#endif

template <typename Block, typename DotFn>
void matvec_impl(float* y, const void* W, const float* x, int64_t N, int64_t K, DotFn dot) {
  check_k(K);
  const int64_t nb = K / kQK;
  thread_local std::vector<BlockQ8> xq;
  quantize_x(x, xq, K);
  const BlockQ8* xqp = xq.data();
  const Block* Wb = static_cast<const Block*>(W);

  auto body = [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) y[r] = dot(Wb + r * nb, xqp, nb);
  };

  ThreadPool& p = pool();
  // Skip threading for tiny problems where wakeup cost dominates.
  const int64_t work = N * K;
  if (work < (1 << 16)) {
    body(0, N);
    return;
  }
  const int nt = p.num_threads();
  int64_t chunk = std::max<int64_t>(1, N / (int64_t(nt) * 4));
  chunk = std::min<int64_t>(chunk, 64);
  p.parallel_for(N, chunk, body);
}

}  // namespace

// ---------------------------------------------------------------------------

size_t quant_row_bytes(DType dtype, int64_t n) {
  switch (dtype) {
    case DType::F32:
      return static_cast<size_t>(n) * 4;
    case DType::Q8:
      check_k(n);
      return static_cast<size_t>(n / kQK) * sizeof(BlockQ8);
    case DType::Q4:
      check_k(n);
      return static_cast<size_t>(n / kQK) * sizeof(BlockQ4);
  }
  throw std::runtime_error("quant_row_bytes: unknown dtype");
}

void quantize_row_q8(const float* x, void* out, int64_t n) {
  check_k(n);
  BlockQ8* blocks = static_cast<BlockQ8*>(out);
  const int64_t nb = n / kQK;
  for (int64_t b = 0; b < nb; ++b) {
    const float* xb = x + b * kQK;
    BlockQ8& blk = blocks[b];
    float amax = 0.f;
    for (int j = 0; j < kQK; ++j) amax = std::max(amax, std::fabs(xb[j]));
    if (amax == 0.f) {
      blk.d = 0.f;
      std::memset(blk.qs, 0, sizeof(blk.qs));
      continue;
    }
    const float d = amax / 127.f;
    const float id = 1.f / d;
    blk.d = d;
    for (int j = 0; j < kQK; ++j) {
      float q = std::round(xb[j] * id);
      q = std::min(127.f, std::max(-127.f, q));
      blk.qs[j] = static_cast<int8_t>(q);
    }
  }
}

void quantize_row_q4(const float* x, void* out, int64_t n) {
  check_k(n);
  BlockQ4* blocks = static_cast<BlockQ4*>(out);
  const int64_t nb = n / kQK;
  for (int64_t b = 0; b < nb; ++b) {
    const float* xb = x + b * kQK;
    BlockQ4& blk = blocks[b];
    float amax = 0.f, m = 0.f;
    for (int j = 0; j < kQK; ++j) {
      const float a = std::fabs(xb[j]);
      if (a > amax) {
        amax = a;
        m = xb[j];
      }
    }
    if (amax == 0.f) {
      blk.d = 0.f;
      std::memset(blk.qs, 0x88, sizeof(blk.qs));  // q = 8 -> value 0
      continue;
    }
    const float d = m / -8.f;
    const float id = 1.f / d;
    blk.d = d;
    for (int j = 0; j < kQK / 2; ++j) {
      float q0 = std::round(xb[j] * id) + 8.f;
      float q1 = std::round(xb[j + kQK / 2] * id) + 8.f;
      q0 = std::min(15.f, std::max(0.f, q0));
      q1 = std::min(15.f, std::max(0.f, q1));
      blk.qs[j] = static_cast<uint8_t>(int(q0) | (int(q1) << 4));
    }
  }
}

void dequantize_row(DType dtype, const void* in, float* out, int64_t n) {
  switch (dtype) {
    case DType::F32:
      std::memcpy(out, in, static_cast<size_t>(n) * sizeof(float));
      return;
    case DType::Q8: {
      check_k(n);
      const BlockQ8* blocks = static_cast<const BlockQ8*>(in);
      for (int64_t b = 0; b < n / kQK; ++b) {
        const float d = blocks[b].d;
        for (int j = 0; j < kQK; ++j) out[b * kQK + j] = float(blocks[b].qs[j]) * d;
      }
      return;
    }
    case DType::Q4: {
      check_k(n);
      const BlockQ4* blocks = static_cast<const BlockQ4*>(in);
      for (int64_t b = 0; b < n / kQK; ++b) {
        const float d = blocks[b].d;
        float* o = out + b * kQK;
        for (int j = 0; j < kQK / 2; ++j) {
          o[j] = float(int(blocks[b].qs[j] & 0x0F) - 8) * d;
          o[j + kQK / 2] = float(int(blocks[b].qs[j] >> 4) - 8) * d;
        }
      }
      return;
    }
  }
  throw std::runtime_error("dequantize_row: unknown dtype");
}

void matvec_q8(float* y, const void* W, const float* x, int64_t N, int64_t K) {
  matvec_impl<BlockQ8>(y, W, x, N, K, row_dot_q8);
}

void matvec_q4(float* y, const void* W, const float* x, int64_t N, int64_t K) {
  matvec_impl<BlockQ4>(y, W, x, N, K, row_dot_q4);
}

}  // namespace tinyllm
