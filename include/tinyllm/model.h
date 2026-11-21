#pragma once
#include <memory>
#include <string>

#include "tinyllm/arena.h"
#include "tinyllm/format.h"

namespace tinyllm {

// Pre-allocated KV cache, layout [layer][pos][kv_dim], kv_dim = n_kv_heads * head_dim.
struct KVCache {
  int n_layers = 0, max_ctx = 0, kv_dim = 0;
  float* k = nullptr;
  float* v = nullptr;

  void init(Arena& arena, int layers, int ctx, int dim) {
    n_layers = layers;
    max_ctx = ctx;
    kv_dim = dim;
    size_t n = static_cast<size_t>(layers) * ctx * dim;
    k = arena.alloc_n<float>(n);
    v = arena.alloc_n<float>(n);
  }
  float* k_at(int layer, int pos) { return k + (static_cast<size_t>(layer) * max_ctx + pos) * kv_dim; }
  float* v_at(int layer, int pos) { return v + (static_cast<size_t>(layer) * max_ctx + pos) * kv_dim; }
};

class Model {
 public:
  virtual ~Model() = default;
  // Run one token at position pos (0-based; positions must be fed in order). Writes
  // into the KV cache and returns logits [vocab_size], valid until the next call.
  virtual const float* forward(int token, int pos) = 0;
  const Config& config() const { return config_; }
  int max_ctx() const { return max_ctx_; }

 protected:
  Config config_{};
  int max_ctx_ = 0;
};

// max_ctx <= 0 means config.max_ctx (capped at 2048 for LLaMA to bound the cache).
std::unique_ptr<Model> load_model(const std::string& path, int max_ctx = 0);  // src/model.cpp

// Architecture factories. The model keeps the WeightFile alive.
std::unique_ptr<Model> make_gpt2(std::shared_ptr<WeightFile> wf, int max_ctx);   // src/gpt2.cpp
std::unique_ptr<Model> make_llama(std::shared_ptr<WeightFile> wf, int max_ctx);  // src/llama.cpp

}  // namespace tinyllm
