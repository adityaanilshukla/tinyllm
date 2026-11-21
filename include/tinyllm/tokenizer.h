#pragma once
#include <memory>
#include <string>
#include <vector>

namespace tinyllm {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  virtual std::vector<int> encode(const std::string& text) const = 0;  // no BOS added
  // Text for a single token, suitable for streaming output (bytes may be a partial
  // UTF-8 sequence for byte-level tokens).
  virtual std::string decode_token(int id) const = 0;
  virtual int vocab_size() const = 0;
};

// GPT-2 byte-level BPE from a dir with vocab.json + merges.txt. src/bpe_gpt2.cpp
std::unique_ptr<Tokenizer> make_gpt2_tokenizer(const std::string& dir);
// SentencePiece tokenizer.model (LLaMA). src/sp_tokenizer.cpp
std::unique_ptr<Tokenizer> make_sp_tokenizer(const std::string& model_path);

}  // namespace tinyllm
