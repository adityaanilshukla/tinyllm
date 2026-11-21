#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "test_util.h"
#include "tinyllm/sampler.h"

namespace tinyllm {
namespace {

SamplerParams params(float temp, int top_k, float top_p, uint64_t seed) {
  SamplerParams p;
  p.temperature = temp;
  p.top_k = top_k;
  p.top_p = top_p;
  p.seed = seed;
  return p;
}

// Indices of logits sorted by value, descending.
std::vector<int> ranking(const std::vector<float>& logits) {
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  return idx;
}

TEST(Sampler, ZeroTemperatureIsArgmax) {
  for (float t : {0.0f, -1.0f}) {
    Sampler s(params(t, 40, 0.95f, 1));
    for (uint32_t seed = 0; seed < 20; ++seed) {
      auto logits = test::rand_vec(1000, seed, -5.0f, 5.0f);
      int want = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
      EXPECT_EQ(s.sample(logits.data(), 1000), want);
    }
  }
  // First index wins ties.
  std::vector<float> tie = {1.0f, 3.0f, 3.0f, 2.0f};
  EXPECT_EQ(Sampler(params(0, 0, 1, 1)).sample(tie.data(), 4), 1);
}

TEST(Sampler, EmptyInput) { EXPECT_EQ(Sampler(params(0.8f, 40, 0.95f, 1)).sample(nullptr, 0), -1); }

TEST(Sampler, FixedSeedIsDeterministic) {
  auto logits = test::rand_vec(500, 3, -2.0f, 2.0f);
  for (auto p : {params(1.0f, 0, 1.0f, 42), params(0.8f, 40, 0.95f, 42), params(1.5f, 0, 0.9f, 42)}) {
    Sampler a(p), b(p);
    std::vector<int> sa, sb;
    for (int i = 0; i < 200; ++i) {
      sa.push_back(a.sample(logits.data(), 500));
      sb.push_back(b.sample(logits.data(), 500));
    }
    EXPECT_EQ(sa, sb);
    // Non-degenerate: a flat-ish distribution should not return the same token every time.
    EXPECT_GT(std::set<int>(sa.begin(), sa.end()).size(), 5u);
  }
  // A different seed gives a different stream.
  Sampler a(params(1.0f, 0, 1.0f, 1)), b(params(1.0f, 0, 1.0f, 2));
  std::vector<int> sa, sb;
  for (int i = 0; i < 50; ++i) {
    sa.push_back(a.sample(logits.data(), 500));
    sb.push_back(b.sample(logits.data(), 500));
  }
  EXPECT_NE(sa, sb);
}

TEST(Sampler, TopKNeverPicksExcluded) {
  auto logits = test::rand_vec(300, 4, -1.0f, 1.0f);
  auto rank = ranking(logits);
  for (int k : {1, 2, 5, 50}) {
    std::set<int> allowed(rank.begin(), rank.begin() + k);
    Sampler s(params(2.0f, k, 1.0f, 7));
    std::set<int> seen;
    for (int i = 0; i < 3000; ++i) {
      int t = s.sample(logits.data(), 300);
      ASSERT_TRUE(allowed.count(t)) << "k=" << k << " picked " << t;
      seen.insert(t);
    }
    EXPECT_EQ(seen.size(), size_t(k)) << "k=" << k;  // high temperature: all of top-k show up
  }
}

TEST(Sampler, TopPNeverPicksExcluded) {
  // Probabilities 0.5, 0.3, 0.15, 0.05 at shuffled indices.
  std::vector<float> logits(4);
  const int order[4] = {2, 0, 3, 1};
  const double probs[4] = {0.5, 0.3, 0.15, 0.05};
  for (int i = 0; i < 4; ++i) logits[order[i]] = float(std::log(probs[i]));
  // Smallest prefix with cumulative mass >= p.
  const std::vector<std::pair<float, int>> cases = {{0.4f, 1}, {0.7f, 2}, {0.75f, 2}, {0.9f, 3}, {0.99f, 4}};
  for (const auto& [p, n_keep] : cases) {
    for (int top_k : {0, 3}) {
      const int keep = top_k > 0 ? std::min(n_keep, top_k) : n_keep;
      Sampler s(params(1.0f, top_k, p, 11));
      std::vector<int> counts(4, 0);
      for (int i = 0; i < 5000; ++i) ++counts[s.sample(logits.data(), 4)];
      for (int r = 0; r < 4; ++r) {
        if (r < keep) EXPECT_GT(counts[order[r]], 0) << "p=" << p << " k=" << top_k << " rank " << r;
        else EXPECT_EQ(counts[order[r]], 0) << "p=" << p << " k=" << top_k << " rank " << r;
      }
    }
  }
}

TEST(Sampler, FrequenciesMatchSoftmax) {
  std::vector<float> logits = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f};
  const float temp = 0.7f;
  std::vector<double> p(5);
  double z = 0.0;
  for (int i = 0; i < 5; ++i) z += (p[i] = std::exp(logits[i] / temp));
  Sampler s(params(temp, 0, 1.0f, 123));
  const int n = 40000;
  std::vector<int> counts(5, 0);
  for (int i = 0; i < n; ++i) ++counts[s.sample(logits.data(), 5)];
  // 5 sigma for a binomial with n = 40000 is at most 0.0125.
  for (int i = 0; i < 5; ++i) EXPECT_NEAR(double(counts[i]) / n, p[i] / z, 0.0125) << "token " << i;
}

}  // namespace
}  // namespace tinyllm
