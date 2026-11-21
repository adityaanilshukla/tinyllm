// Temperature / top-k / top-p sampling.
#include "tinyllm/sampler.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace tinyllm {

Sampler::Sampler(const SamplerParams& p) : params_(p) {
  uint64_t seed = p.seed;
  if (seed == 0) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }
  rng_.seed(seed);
}

int Sampler::sample(const float* logits, int n) {
  if (n <= 0) return -1;
  if (params_.temperature <= 0.0f) {
    return static_cast<int>(std::max_element(logits, logits + n) - logits);
  }

  const float inv_t = 1.0f / params_.temperature;
  scratch_.resize(n);
  for (int i = 0; i < n; ++i) scratch_[i] = {logits[i] * inv_t, i};

  auto by_logit_desc = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
    return a.first > b.first;
  };
  int kept = n;
  if (params_.top_k > 0 && params_.top_k < n) {
    kept = params_.top_k;
    std::partial_sort(scratch_.begin(), scratch_.begin() + kept, scratch_.end(), by_logit_desc);
  } else if (params_.top_p < 1.0f) {
    std::sort(scratch_.begin(), scratch_.end(), by_logit_desc);  // top-p needs sorted order
  }

  // Stable softmax over the kept set (in place: .first becomes the probability).
  float mx = scratch_[0].first;
  for (int i = 1; i < kept; ++i) mx = std::max(mx, scratch_[i].first);
  double sum = 0.0;
  for (int i = 0; i < kept; ++i) {
    float e = std::exp(scratch_[i].first - mx);
    scratch_[i].first = e;
    sum += e;
  }
  for (int i = 0; i < kept; ++i) scratch_[i].first = static_cast<float>(scratch_[i].first / sum);

  // Nucleus: kept set is sorted descending whenever top_p < 1.
  if (params_.top_p < 1.0f) {
    double cum = 0.0;
    int cut = kept;
    for (int i = 0; i < kept; ++i) {
      cum += scratch_[i].first;
      if (cum >= params_.top_p) {
        cut = i + 1;
        break;
      }
    }
    kept = std::max(1, cut);
  }

  double total = 0.0;
  for (int i = 0; i < kept; ++i) total += scratch_[i].first;
  std::uniform_real_distribution<double> dist(0.0, total);
  double r = dist(rng_);
  double acc = 0.0;
  for (int i = 0; i < kept; ++i) {
    acc += scratch_[i].first;
    if (r < acc) return scratch_[i].second;
  }
  return scratch_[kept - 1].second;
}

}  // namespace tinyllm
