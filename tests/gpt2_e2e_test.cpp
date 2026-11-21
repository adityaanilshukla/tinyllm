// End-to-end GPT-2 small: convert models/gpt2 with the built tinyllm-convert, then run
// greedy decoding. Skipped when models/gpt2 is absent.
//
// The golden tokens and logits below were pinned from this engine itself (f32, Apple
// M-series, 2026-11-21), not from HF transformers; they are a regression check, not a
// correctness reference. CLAUDE.md records a one-off comparison of GPT-2 f32 logits
// against an independent implementation (max diff 6e-5).
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "test_util.h"
#include "tinyllm/model.h"
#include "tinyllm/sampler.h"
#include "tinyllm/tokenizer.h"

namespace tinyllm {
namespace {

namespace fs = std::filesystem;

// Converted file, shared by every test process (ctest runs each test separately).
// Reconverted when the converter or the source weights are newer than the cache.
std::string converted_gpt2() {
  const fs::path src = fs::path(test::models_dir()) / "gpt2";
  if (!fs::exists(src / "model.safetensors") || !fs::exists(src / "config.json")) return "";
  const fs::path root = test::env_or("TINYLLM_TEST_TMPDIR", TINYLLM_TEST_TMPDIR);
  fs::create_directories(root);
  const fs::path out = root / "gpt2-f32.tllm";
  const fs::path bin = TINYLLM_CONVERT_BIN;
  if (fs::exists(out) && fs::last_write_time(out) > fs::last_write_time(bin) &&
      fs::last_write_time(out) > fs::last_write_time(src / "model.safetensors"))
    return out.string();
  // Convert to a per-process name, then rename, so parallel test processes don't clash.
  const fs::path tmp = test::tmp_dir() / "gpt2-f32.tllm";
  const std::string cmd = "\"" + bin.string() + "\" \"" + src.string() + "\" \"" + tmp.string() + "\" > /dev/null";
  if (std::system(cmd.c_str()) != 0) return "";
  fs::rename(tmp, out);
  return out.string();
}

const std::vector<int> kPrompt = {464, 3139, 286, 4881, 318};  // "The capital of France is"

class Gpt2E2E : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string path = converted_gpt2();
    if (!path.empty()) model_ = load_model(path, 64);
  }
  static void TearDownTestSuite() { model_.reset(); }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "models/gpt2 not found (or conversion failed)";
  }
  // Fresh prefill each time; positions restart at 0 and overwrite the KV cache.
  static const float* prefill() {
    const float* logits = nullptr;
    for (size_t i = 0; i < kPrompt.size(); ++i) logits = model_->forward(kPrompt[i], int(i));
    return logits;
  }
  static std::unique_ptr<Model> model_;
};
std::unique_ptr<Model> Gpt2E2E::model_;

TEST_F(Gpt2E2E, ConfigIsGpt2Small) {
  const Config& c = model_->config();
  EXPECT_EQ(c.arch, Arch::GPT2);
  EXPECT_EQ(c.vocab_size, 50257);
  EXPECT_EQ(c.d_model, 768);
  EXPECT_EQ(c.n_layers, 12);
  EXPECT_EQ(c.n_heads, 12);
  EXPECT_EQ(c.max_ctx, 1024);
  EXPECT_EQ(model_->max_ctx(), 64);
}

TEST_F(Gpt2E2E, GreedyCapitalOfFrance) {
  const float* logits = prefill();
  Sampler greedy(SamplerParams{0.0f, 0, 1.0f, 1});
  std::vector<int> out;
  int pos = int(kPrompt.size());
  for (int i = 0; i < 12; ++i) {
    const int next = greedy.sample(logits, model_->config().vocab_size);
    out.push_back(next);
    logits = model_->forward(next, pos++);
  }
  // " the capital of the French Republic, and the capital of the"
  const std::vector<int> want = {262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286, 262};
  EXPECT_EQ(out, want);

  const std::string dir = test::models_dir() + "/gpt2";
  if (fs::exists(dir + "/vocab.json")) {
    auto tok = make_gpt2_tokenizer(dir);
    EXPECT_EQ(tok->encode("The capital of France is"), kPrompt);
    std::string text;
    for (int t : out) text += tok->decode_token(t);
    EXPECT_EQ(text, " the capital of the French Republic, and the capital of the");
  }
}

TEST_F(Gpt2E2E, PinnedLogits) {
  const float* logits = prefill();
  // Next-token logits after the prompt. Tolerance allows for summation-order changes.
  const std::vector<std::pair<int, float>> want = {
      {0, -108.954254f},   {11, -103.402267f},  {262, -100.249763f},  // "!", ",", " the"
      {6342, -101.214249f}, {257, -100.855499f}, {50256, -110.377899f},  // " Paris", " a", <|endoftext|>
  };
  for (const auto& [id, v] : want) EXPECT_NEAR(logits[id], v, 2e-3) << "token " << id;
  int argmax = 0;
  for (int i = 1; i < model_->config().vocab_size; ++i)
    if (logits[i] > logits[argmax]) argmax = i;
  EXPECT_EQ(argmax, 262);
}

// Re-running the same prompt from position 0 gives bit-identical logits: the KV cache
// rows are fully overwritten, and nothing depends on stale state.
TEST_F(Gpt2E2E, PrefillIsRepeatable) {
  const int V = model_->config().vocab_size;
  const float* a = prefill();
  std::vector<float> first(a, a + V);
  model_->forward(50256, int(kPrompt.size()));  // extend, then start over
  const float* b = prefill();
  EXPECT_EQ(first, std::vector<float>(b, b + V));
}

}  // namespace
}  // namespace tinyllm
