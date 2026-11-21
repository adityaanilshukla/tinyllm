#pragma once
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace tinyllm {

struct SamplerParams {
  float temperature = 0.8f;  // <= 0 means greedy argmax
  int top_k = 40;            // <= 0 disables
  float top_p = 0.95f;       // >= 1 disables
  uint64_t seed = 0;         // 0 = random seed
};

// Implemented in src/sampler.cpp.
class Sampler {
 public:
  explicit Sampler(const SamplerParams& params);
  int sample(const float* logits, int n);

 private:
  SamplerParams params_;
  std::mt19937_64 rng_;
  std::vector<std::pair<float, int>> scratch_;
};

}  // namespace tinyllm
