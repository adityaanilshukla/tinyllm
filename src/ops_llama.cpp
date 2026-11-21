// LLaMA-specific CPU ops: RMSNorm, RoPE (HF rotate_half pairing), SwiGLU.
#include <cmath>
#include <vector>

#include "tinyllm/ops.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tinyllm {

void rmsnorm(float* y, const float* x, const float* gamma, int D, float eps) {
  int i = 0;
  float ss = 0.0f;
#if defined(__ARM_NEON)
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f), a2 = vdupq_n_f32(0.f),
              a3 = vdupq_n_f32(0.f);
  for (; i + 16 <= D; i += 16) {
    float32x4_t v0 = vld1q_f32(x + i), v1 = vld1q_f32(x + i + 4), v2 = vld1q_f32(x + i + 8),
                v3 = vld1q_f32(x + i + 12);
    a0 = vfmaq_f32(a0, v0, v0);
    a1 = vfmaq_f32(a1, v1, v1);
    a2 = vfmaq_f32(a2, v2, v2);
    a3 = vfmaq_f32(a3, v3, v3);
  }
  ss = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#endif
  for (; i < D; ++i) ss += x[i] * x[i];
  const float scale = 1.0f / std::sqrt(ss / static_cast<float>(D) + eps);
  for (int j = 0; j < D; ++j) y[j] = x[j] * scale * gamma[j];
}

namespace {
// Cache of cos/sin for the last (pos, head_dim, theta) seen. q and k at the same
// position share it, so each position costs one table build per forward.
struct RopeCache {
  int pos = -1, head_dim = 0;
  float theta = 0.f;
  std::vector<float> inv_freq, cos_t, sin_t;

  void prepare(int p, int hd, float th) {
    const int half = hd / 2;
    if (hd != head_dim || th != theta) {
      head_dim = hd;
      theta = th;
      pos = -1;
      inv_freq.resize(half);
      cos_t.resize(half);
      sin_t.resize(half);
      for (int i = 0; i < half; ++i)
        inv_freq[i] = static_cast<float>(
            1.0 / std::pow(static_cast<double>(th), (2.0 * i) / static_cast<double>(hd)));
    }
    if (p != pos) {
      pos = p;
      for (int i = 0; i < half; ++i) {
        const float ang = static_cast<float>(p) * inv_freq[i];  // matches HF float32 math
        cos_t[i] = std::cos(ang);
        sin_t[i] = std::sin(ang);
      }
    }
  }
};
thread_local RopeCache g_rope;
}  // namespace

void rope(float* x, int n_heads, int head_dim, int pos, float theta) {
  g_rope.prepare(pos, head_dim, theta);
  const int half = head_dim / 2;
  const float* c = g_rope.cos_t.data();
  const float* s = g_rope.sin_t.data();
  for (int h = 0; h < n_heads; ++h) {
    float* xa = x + static_cast<size_t>(h) * head_dim;
    float* xb = xa + half;
    for (int i = 0; i < half; ++i) {
      const float a = xa[i], b = xb[i];
      xa[i] = a * c[i] - b * s[i];
      xb[i] = b * c[i] + a * s[i];
    }
  }
}

void silu_mul(float* gate, const float* up, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const float g = gate[i];
    gate[i] = g / (1.0f + std::exp(-g)) * up[i];
  }
}

}  // namespace tinyllm
