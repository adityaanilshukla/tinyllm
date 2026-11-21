// GPT-2 byte-level BPE tokenizer (vocab.json + merges.txt).
#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tinyllm/tokenizer.h"

namespace tinyllm {
namespace {

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// Decodes one UTF-8 code point starting at s[i]; advances i. Invalid bytes yield the byte value.
uint32_t next_utf8(const std::string& s, size_t& i) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
  if (i + len > s.size()) len = 1;
  uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
  for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  i += len;
  return cp;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Minimal parser for a flat JSON object {"string": int, ...}.
class FlatJsonParser {
 public:
  explicit FlatJsonParser(const std::string& s) : s_(s) {}

  std::vector<std::pair<std::string, int>> parse() {
    std::vector<std::pair<std::string, int>> out;
    skip_ws();
    expect('{');
    skip_ws();
    if (peek() == '}') return out;
    for (;;) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      int val = parse_int();
      out.emplace_back(std::move(key), val);
      skip_ws();
      char c = get();
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'");
    }
    return out;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* msg) {
    throw std::runtime_error(std::string("vocab.json parse error at ") + std::to_string(i_) + ": " + msg);
  }
  char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }
  char get() {
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_++];
  }
  void expect(char c) {
    if (get() != c) fail("unexpected character");
  }
  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }
  uint32_t hex4() {
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = get();
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }
  std::string parse_string() {
    expect('"');
    std::string out;
    for (;;) {
      char c = get();
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      char e = get();
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && peek() == '\\') {
            size_t save = i_;
            ++i_;
            if (peek() == 'u') {
              ++i_;
              uint32_t lo = hex4();
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                i_ = save;
              }
            } else {
              i_ = save;
            }
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
  int parse_int() {
    bool neg = false;
    if (peek() == '-') {
      neg = true;
      ++i_;
    }
    if (!(peek() >= '0' && peek() <= '9')) fail("expected integer");
    long v = 0;
    while (peek() >= '0' && peek() <= '9') v = v * 10 + (get() - '0');
    return static_cast<int>(neg ? -v : v);
  }
};

bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_other(unsigned char c) { return !is_space(c) && !is_letter(c) && !is_digit(c); }

// Hand-written approximation of the GPT-2 pre-tokenization regex:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> pretokenize(const std::string& text) {
  std::vector<std::string> out;
  const size_t n = text.size();
  auto at = [&](size_t k) -> unsigned char { return static_cast<unsigned char>(text[k]); };
  size_t i = 0;
  while (i < n) {
    unsigned char c = at(i);
    if (c == '\'' && i + 1 < n) {
      static const char* kContractions[] = {"re", "ve", "ll", "s", "t", "m", "d"};
      bool matched = false;
      for (const char* suf : kContractions) {
        size_t len = std::char_traits<char>::length(suf);
        if (i + 1 + len <= n && text.compare(i + 1, len, suf) == 0) {
          out.push_back(text.substr(i, 1 + len));
          i += 1 + len;
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    size_t j = i;
    if (c == ' ' && i + 1 < n && !is_space(at(i + 1))) j = i + 1;
    unsigned char d = at(j);
    if (j != i || !is_space(d)) {
      size_t k = j;
      if (is_letter(d)) {
        while (k < n && is_letter(at(k))) ++k;
      } else if (is_digit(d)) {
        while (k < n && is_digit(at(k))) ++k;
      } else {
        while (k < n && is_other(at(k))) ++k;
      }
      out.push_back(text.substr(i, k - i));
      i = k;
      continue;
    }
    // Whitespace run.
    size_t k = i;
    while (k < n && is_space(at(k))) ++k;
    if (k == n || k - i == 1) {
      out.push_back(text.substr(i, k - i));
      i = k;
    } else {
      // \s+(?!\S): leave the last whitespace char for the following token.
      out.push_back(text.substr(i, k - 1 - i));
      i = k - 1;
    }
  }
  return out;
}

class Gpt2Tokenizer : public Tokenizer {
 public:
  explicit Gpt2Tokenizer(const std::string& dir) {
    // bytes_to_unicode
    std::vector<bool> direct(256, false);
    for (int b = '!'; b <= '~'; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    uint32_t extra = 0;
    for (int b = 0; b < 256; ++b) {
      uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : 256 + extra++;
      append_utf8(byte_enc_[b], cp);
      byte_dec_[cp] = static_cast<unsigned char>(b);
    }

    std::string json = read_file(dir + "/vocab.json");
    auto entries = FlatJsonParser(json).parse();
    int max_id = -1;
    for (auto& e : entries) max_id = std::max(max_id, e.second);
    id_to_token_.resize(max_id + 1);
    for (auto& e : entries) {
      if (e.second < 0) continue;
      id_to_token_[e.second] = e.first;
      token_to_id_[e.first] = e.second;
    }

    std::ifstream mf(dir + "/merges.txt");
    if (!mf) throw std::runtime_error("cannot open " + dir + "/merges.txt");
    std::string line;
    int rank = 0;
    while (std::getline(mf, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line.rfind("#version", 0) == 0) continue;
      size_t sp = line.find(' ');
      if (sp == std::string::npos) continue;
      merge_rank_.emplace(line, rank++);  // key "a b"
    }
  }

  std::vector<int> encode(const std::string& text) const override {
    std::vector<int> ids;
    for (const std::string& piece : pretokenize(text)) {
      auto it = cache_.find(piece);
      if (it == cache_.end()) it = cache_.emplace(piece, bpe(piece)).first;
      ids.insert(ids.end(), it->second.begin(), it->second.end());
    }
    return ids;
  }

  std::string decode_token(int id) const override {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return "";
    const std::string& tok = id_to_token_[id];
    std::string out;
    size_t i = 0;
    while (i < tok.size()) {
      uint32_t cp = next_utf8(tok, i);
      auto it = byte_dec_.find(cp);
      if (it != byte_dec_.end()) out += static_cast<char>(it->second);
      else append_utf8(out, cp);
    }
    return out;
  }

  int vocab_size() const override { return static_cast<int>(id_to_token_.size()); }

 private:
  std::string byte_enc_[256];
  std::unordered_map<uint32_t, unsigned char> byte_dec_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> merge_rank_;
  mutable std::unordered_map<std::string, std::vector<int>> cache_;

  std::vector<int> bpe(const std::string& piece) const {
    std::vector<std::string> word;
    word.reserve(piece.size());
    for (unsigned char b : piece) word.push_back(byte_enc_[b]);

    std::string key;
    while (word.size() > 1) {
      int best = INT_MAX;
      size_t best_i = 0;
      for (size_t i = 0; i + 1 < word.size(); ++i) {
        key.assign(word[i]).append(" ").append(word[i + 1]);
        auto it = merge_rank_.find(key);
        if (it != merge_rank_.end() && it->second < best) {
          best = it->second;
          best_i = i;
        }
      }
      if (best == INT_MAX) break;
      const std::string a = word[best_i], b = word[best_i + 1];
      std::vector<std::string> merged;
      merged.reserve(word.size());
      for (size_t i = 0; i < word.size();) {
        if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
          merged.push_back(a + b);
          i += 2;
        } else {
          merged.push_back(word[i]);
          ++i;
        }
      }
      word.swap(merged);
    }

    std::vector<int> ids;
    ids.reserve(word.size());
    for (const std::string& w : word) {
      auto it = token_to_id_.find(w);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
      } else {
        // Should not happen with a complete byte-level vocab; fall back to per-byte tokens.
        size_t i = 0;
        while (i < w.size()) {
          size_t s = i;
          next_utf8(w, i);
          auto bt = token_to_id_.find(w.substr(s, i - s));
          if (bt != token_to_id_.end()) ids.push_back(bt->second);
        }
      }
    }
    return ids;
  }
};

}  // namespace

std::unique_ptr<Tokenizer> make_gpt2_tokenizer(const std::string& dir) {
  return std::make_unique<Gpt2Tokenizer>(dir);
}

}  // namespace tinyllm
