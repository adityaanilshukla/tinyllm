// SentencePiece tokenizer wrapper (LLaMA / TinyLlama tokenizer.model).
#include <sentencepiece_processor.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "tinyllm/tokenizer.h"

namespace tinyllm {
namespace {

class SpTokenizer : public Tokenizer {
 public:
  explicit SpTokenizer(const std::string& path) {
    auto st = sp_.Load(path);
    if (!st.ok()) throw std::runtime_error("failed to load sentencepiece model " + path + ": " + st.ToString());
  }

  std::vector<int> encode(const std::string& text) const override {
    std::vector<int> ids;
    auto st = sp_.Encode(text, &ids);
    if (!st.ok()) throw std::runtime_error("sentencepiece encode failed: " + st.ToString());
    return ids;
  }

  std::string decode_token(int id) const override {
    if (id < 0 || id >= sp_.GetPieceSize()) return "";
    if (sp_.IsControl(id) || sp_.IsUnknown(id)) return "";
    const std::string& piece = sp_.IdToPiece(id);
    if (sp_.IsByte(id) || (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>')) {
      return std::string(1, static_cast<char>(std::stoi(piece.substr(3, 2), nullptr, 16)));
    }
    static const std::string kSpace = "\xE2\x96\x81";  // U+2581
    std::string out;
    out.reserve(piece.size());
    for (size_t i = 0; i < piece.size();) {
      if (piece.compare(i, kSpace.size(), kSpace) == 0) {
        out += ' ';
        i += kSpace.size();
      } else {
        out += piece[i++];
      }
    }
    return out;
  }

  int vocab_size() const override { return sp_.GetPieceSize(); }

 private:
  sentencepiece::SentencePieceProcessor sp_;
};

}  // namespace

std::unique_ptr<Tokenizer> make_sp_tokenizer(const std::string& model_path) {
  return std::make_unique<SpTokenizer>(model_path);
}

}  // namespace tinyllm
