#pragma once
#include <cstdint>

#include "tinyllm/tensor.h"

namespace tinyllm {

// CPU ops. Single-token decoding: every op works on one position at a time.
// Implemented in src/ops_cpu.cpp unless noted.

// y[N] = W[N, K] x[K] (+ bias[N] if non-null). Dispatches on W.dtype
// (F32 here, Q8/Q4 via matvec_q8/matvec_q4 from quant.h).
void linear(float* y, const float* x, const Tensor& W, const float* bias, int K, int N);

void layernorm(float* y, const float* x, const float* gamma, const float* beta, int D, float eps);
void gelu(float* x, int64_t n);  // tanh approximation (GPT-2 "gelu_new")
void softmax(float* x, int n);   // numerically stable, in place
void add_inplace(float* y, const float* x, int64_t n);  // y += x

// Causal attention for one query token at position n_pos - 1.
//   q:   [n_heads * head_dim]
//   k,v: cache for this layer, n_pos rows, row stride n_kv_heads * head_dim
//   out: [n_heads * head_dim]
//   scores: scratch of at least n_heads * n_pos floats
// Query head h reads KV head h / (n_heads / n_kv_heads) (GQA; MHA when equal).
void attention(float* out, const float* q, const float* k, const float* v, int n_pos, int n_heads,
               int n_kv_heads, int head_dim, float* scores);

// --- LLaMA ops, implemented in src/ops_llama.cpp ---
void rmsnorm(float* y, const float* x, const float* gamma, int D, float eps);
// In-place RoPE on x [n_heads * head_dim] at position pos, HF rotate_half pairing:
// (x[i], x[i + head_dim/2]) rotated by angle pos * theta^(-2i/head_dim).
void rope(float* x, int n_heads, int head_dim, int pos, float theta);
void silu_mul(float* gate, const float* up, int64_t n);  // gate = silu(gate) * up

}  // namespace tinyllm
