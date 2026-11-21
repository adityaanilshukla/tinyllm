// Tokenizers against the real vocab files. Skipped when models/ is absent.
//
// GPT-2 ids are from the HF vocab (openai-community/gpt2). SentencePiece ids were
// produced by the Python sentencepiece package (SentencePieceProcessor.encode) on
// TinyLlama's tokenizer.model, independent of src/sp_tokenizer.cpp.
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "test_util.h"
#include "tinyllm/tokenizer.h"

namespace tinyllm {
namespace {

namespace fs = std::filesystem;

std::string decode(const Tokenizer& tok, const std::vector<int>& ids) {
  std::string s;
  for (int id : ids) s += tok.decode_token(id);
  return s;
}

const std::vector<std::string> kRoundTrip = {
    "Hello world",
    "The capital of France is",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\n\n",
    "don't you'll we've I'M",
    "numbers 1234567 and 3.14159",
    "naïve café 日本語 🙂",
    "punctuation!? ... ---",
    "",
};

class Gpt2Tok : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string dir = test::models_dir() + "/gpt2";
    if (fs::exists(dir + "/vocab.json") && fs::exists(dir + "/merges.txt")) tok_ = make_gpt2_tokenizer(dir);
  }
  static void TearDownTestSuite() { tok_.reset(); }
  void SetUp() override {
    if (!tok_) GTEST_SKIP() << "models/gpt2/{vocab.json,merges.txt} not found";
  }
  static std::unique_ptr<Tokenizer> tok_;
};
std::unique_ptr<Tokenizer> Gpt2Tok::tok_;

TEST_F(Gpt2Tok, KnownIds) {
  EXPECT_EQ(tok_->vocab_size(), 50257);
  EXPECT_EQ(tok_->encode("Hello world"), (std::vector<int>{15496, 995}));
  EXPECT_EQ(tok_->encode("The capital of France is"), (std::vector<int>{464, 3139, 286, 4881, 318}));
  EXPECT_EQ(tok_->encode("Hello, world!"), (std::vector<int>{15496, 11, 995, 0}));
  // \s+(?!\S): of two spaces before a word, the first stands alone (220) and the second
  // joins the word.
  EXPECT_EQ(tok_->encode("a  b"), (std::vector<int>{64, 220, 275}));
  EXPECT_EQ(tok_->decode_token(50256), "<|endoftext|>");
  EXPECT_EQ(tok_->decode_token(220), " ");
  EXPECT_EQ(tok_->decode_token(-1), "");
  EXPECT_EQ(tok_->decode_token(50257), "");
}

TEST_F(Gpt2Tok, RoundTrip) {
  for (const std::string& s : kRoundTrip) EXPECT_EQ(decode(*tok_, tok_->encode(s)), s) << "'" << s << "'";
  EXPECT_TRUE(tok_->encode("").empty());
}

TEST_F(Gpt2Tok, ArbitraryBytesRoundTrip) {
  // Byte-level BPE must be lossless for any byte string, including invalid UTF-8.
  std::string s;
  for (int b = 1; b < 256; ++b) s += char(b);
  s += "\xff\xfe\x80 end";
  EXPECT_EQ(decode(*tok_, tok_->encode(s)), s);
}

class SpTok : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string path = test::models_dir() + "/tinyllama/tokenizer.model";
    if (fs::exists(path)) tok_ = make_sp_tokenizer(path);
  }
  static void TearDownTestSuite() { tok_.reset(); }
  void SetUp() override {
    if (!tok_) GTEST_SKIP() << "models/tinyllama/tokenizer.model not found";
  }
  static std::unique_ptr<Tokenizer> tok_;
};
std::unique_ptr<Tokenizer> SpTok::tok_;

TEST_F(SpTok, KnownIds) {
  EXPECT_EQ(tok_->vocab_size(), 32000);
  EXPECT_EQ(tok_->encode("Hello world"), (std::vector<int>{15043, 3186}));
  EXPECT_EQ(tok_->encode("The capital of France is"), (std::vector<int>{450, 7483, 310, 3444, 338}));
  EXPECT_EQ(tok_->encode("a\nb\tc"), (std::vector<int>{263, 13, 29890, 12, 29883}));
  // Byte fallback for the emoji (F0 9F 99 82).
  EXPECT_EQ(tok_->encode("naïve café 日本語 🙂"),
            (std::vector<int>{1055, 30085, 345, 274, 28059, 29871, 30325, 30346, 30968, 29871, 243, 162, 156, 133}));
  // encode() never adds BOS; control tokens decode to nothing.
  EXPECT_EQ(tok_->decode_token(1), "");
  EXPECT_EQ(tok_->decode_token(2), "");
  EXPECT_EQ(tok_->decode_token(13), "\n");  // <0x0A>
}

// SentencePiece adds a dummy-prefix space, so decoding gives " " + text for text that
// doesn't already start with a space, and runs of spaces are kept.
TEST_F(SpTok, RoundTrip) {
  for (const std::string& s : kRoundTrip) {
    if (s.empty()) {
      EXPECT_TRUE(tok_->encode(s).empty());
      continue;
    }
    EXPECT_EQ(decode(*tok_, tok_->encode(s)), " " + s) << "'" << s << "'";
  }
}

}  // namespace
}  // namespace tinyllm
