// tinyllm-convert: HF safetensors model dir -> .tllm
//
//   tinyllm-convert <hf_model_dir> <out.tllm> [--quant q8|q4]
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyllm/format.h"
#include "tinyllm/quant.h"
#include "tinyllm/tensor.h"

using namespace tinyllm;

namespace {

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error(msg); }

// ---------------------------------------------------------------- minimal JSON

struct Json {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool has(const std::string& k) const { return kind == Obj && obj.count(k) != 0; }
  const Json& at(const std::string& k) const {
    if (kind != Obj) die("json: not an object when looking up '" + k + "'");
    auto it = obj.find(k);
    if (it == obj.end()) die("json: missing key '" + k + "'");
    return it->second;
  }
  double number() const {
    if (kind != Num) die("json: expected number");
    return num;
  }
  const std::string& string() const {
    if (kind != Str) die("json: expected string");
    return str;
  }
};

class JsonParser {
 public:
  JsonParser(const char* p, size_t n) : p_(p), end_(p + n) {}
  Json parse() {
    Json v = value();
    ws();
    if (p_ != end_) err("trailing characters");
    return v;
  }

 private:
  const char* p_;
  const char* end_;

  [[noreturn]] void err(const std::string& m) { die("json parse error: " + m); }
  void ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
  }
  char peek() {
    ws();
    if (p_ >= end_) err("unexpected end of input");
    return *p_;
  }
  void expect(char c) {
    if (peek() != c) err(std::string("expected '") + c + "'");
    ++p_;
  }
  bool lit(const char* s) {
    size_t n = std::strlen(s);
    if (size_t(end_ - p_) >= n && std::memcmp(p_, s, n) == 0) {
      p_ += n;
      return true;
    }
    return false;
  }
  Json value() {
    char c = peek();
    Json v;
    if (c == '{') {
      ++p_;
      v.kind = Json::Obj;
      if (peek() == '}') { ++p_; return v; }
      for (;;) {
        if (peek() != '"') err("expected object key");
        std::string k = strval();
        expect(':');
        v.obj[k] = value();
        char d = peek();
        ++p_;
        if (d == '}') break;
        if (d != ',') err("expected ',' or '}'");
      }
    } else if (c == '[') {
      ++p_;
      v.kind = Json::Arr;
      if (peek() == ']') { ++p_; return v; }
      for (;;) {
        v.arr.push_back(value());
        char d = peek();
        ++p_;
        if (d == ']') break;
        if (d != ',') err("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.kind = Json::Str;
      v.str = strval();
    } else if (lit("true")) {
      v.kind = Json::Bool; v.b = true;
    } else if (lit("false")) {
      v.kind = Json::Bool; v.b = false;
    } else if (lit("null")) {
      v.kind = Json::Null;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      v.kind = Json::Num;
      const char* s = p_;
      while (p_ < end_ && (std::strchr("+-.eE", *p_) || (*p_ >= '0' && *p_ <= '9'))) ++p_;
      std::string tok(s, p_);
      char* e = nullptr;
      v.num = std::strtod(tok.c_str(), &e);
      if (e != tok.c_str() + tok.size()) err("bad number '" + tok + "'");
    } else {
      err(std::string("unexpected character '") + c + "'");
    }
    return v;
  }
  static void utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += char(cp);
    } else if (cp < 0x800) {
      out += char(0xC0 | (cp >> 6));
      out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += char(0xE0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xF0 | (cp >> 18));
      out += char(0x80 | ((cp >> 12) & 0x3F));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    }
  }
  uint32_t hex4() {
    if (end_ - p_ < 4) err("short \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char h = *p_++;
      v <<= 4;
      if (h >= '0' && h <= '9') v |= h - '0';
      else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
      else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
      else err("bad hex in \\u escape");
    }
    return v;
  }
  std::string strval() {
    expect('"');
    std::string out;
    for (;;) {
      if (p_ >= end_) err("unterminated string");
      char c = *p_++;
      if (c == '"') break;
      if (c != '\\') { out += c; continue; }
      if (p_ >= end_) err("unterminated escape");
      char e = *p_++;
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
          if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
            p_ += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(out, cp);
          break;
        }
        default: err(std::string("bad escape \\") + e);
      }
    }
    return out;
  }
};

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---------------------------------------------------------------- safetensors

enum class SrcType { F32, F16, BF16 };

struct SrcTensor {
  SrcType type;
  std::vector<int64_t> shape;
  const uint8_t* data;
  size_t nbytes;
};

struct Safetensors {
  void* map = MAP_FAILED;
  size_t size = 0;
  std::map<std::string, SrcTensor> tensors;
  ~Safetensors() {
    if (map != MAP_FAILED) munmap(map, size);
  }
};

void open_safetensors(const std::string& path, Safetensors& st) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) die("cannot open " + path + ": " + std::strerror(errno));
  struct stat sb {};
  fstat(fd, &sb);
  st.size = size_t(sb.st_size);
  if (st.size < 8) { ::close(fd); die(path + ": too small for safetensors"); }
  st.map = mmap(nullptr, st.size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (st.map == MAP_FAILED) die("mmap failed for " + path);
  const auto* base = static_cast<const uint8_t*>(st.map);
  uint64_t n = 0;
  for (int i = 0; i < 8; ++i) n |= uint64_t(base[i]) << (8 * i);
  if (n > st.size - 8) die(path + ": header length out of bounds");
  Json h = JsonParser(reinterpret_cast<const char*>(base + 8), n).parse();
  if (h.kind != Json::Obj) die(path + ": header is not a JSON object");
  const uint8_t* data = base + 8 + n;
  size_t data_size = st.size - 8 - n;
  for (auto& kv : h.obj) {
    if (kv.first == "__metadata__") continue;
    const Json& j = kv.second;
    const std::string& dt = j.at("dtype").string();
    SrcTensor t;
    size_t esz;
    if (dt == "F32") { t.type = SrcType::F32; esz = 4; }
    else if (dt == "F16") { t.type = SrcType::F16; esz = 2; }
    else if (dt == "BF16") { t.type = SrcType::BF16; esz = 2; }
    else die("tensor " + kv.first + ": unsupported dtype " + dt);
    size_t numel = 1;
    for (auto& d : j.at("shape").arr) {
      t.shape.push_back(int64_t(d.number()));
      numel *= size_t(d.number());
    }
    const auto& offs = j.at("data_offsets").arr;
    if (offs.size() != 2) die("tensor " + kv.first + ": bad data_offsets");
    size_t b = size_t(offs[0].number()), e = size_t(offs[1].number());
    if (e < b || e > data_size) die("tensor " + kv.first + ": data_offsets out of bounds");
    if (e - b != numel * esz) die("tensor " + kv.first + ": size mismatch");
    t.data = data + b;
    t.nbytes = e - b;
    st.tensors.emplace(kv.first, std::move(t));
  }
}

float half_to_f32(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t man = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // subnormal
      int e = -1;
      do { man <<= 1; ++e; } while (!(man & 0x400));
      man &= 0x3FF;
      bits = sign | uint32_t(127 - 15 - e) << 23 | man << 13;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | man << 13;
  } else {
    bits = sign | (exp + 127 - 15) << 23 | man << 13;
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

void to_f32(const SrcTensor& t, float* out) {
  size_t n = 1;
  for (auto d : t.shape) n *= size_t(d);
  switch (t.type) {
    case SrcType::F32:
      std::memcpy(out, t.data, n * 4);
      break;
    case SrcType::BF16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t v;
        std::memcpy(&v, t.data + 2 * i, 2);
        uint32_t bits = uint32_t(v) << 16;
        std::memcpy(&out[i], &bits, 4);
      }
      break;
    case SrcType::F16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t v;
        std::memcpy(&v, t.data + 2 * i, 2);
        out[i] = half_to_f32(v);
      }
      break;
  }
}

// ---------------------------------------------------------------- mapping

struct OutTensor {
  std::string name;
  const SrcTensor* src;
  bool transpose;
  DType dtype;
  std::vector<int64_t> shape;  // output shape
  uint64_t offset = 0;
  uint64_t nbytes = 0;
};

bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
bool ends_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Splits "h.{i}.rest" / "model.layers.{i}.rest". Returns false if no match.
bool split_layer(const std::string& s, const std::string& prefix, int& layer, std::string& rest) {
  if (!starts_with(s, prefix)) return false;
  size_t i = prefix.size(), j = i;
  while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
  if (j == i || j >= s.size() || s[j] != '.') return false;
  layer = std::atoi(s.substr(i, j - i).c_str());
  rest = s.substr(j + 1);
  return true;
}

// Returns canonical name, "" to skip; sets transpose. Throws on unknown names.
std::string map_gpt2(std::string n, bool& transpose) {
  transpose = false;
  if (starts_with(n, "transformer.")) n = n.substr(12);
  if (n == "lm_head.weight") return "";  // tied to wte
  if (n == "wte.weight") return "tok_emb";
  if (n == "wpe.weight") return "pos_emb";
  if (n == "ln_f.weight") return "ln_f.w";
  if (n == "ln_f.bias") return "ln_f.b";
  int l;
  std::string r;
  if (!split_layer(n, "h.", l, r)) die("unrecognized GPT-2 tensor: " + n);
  std::string p = "l" + std::to_string(l) + ".";
  static const std::map<std::string, std::pair<std::string, bool>> m = {
      {"ln_1.weight", {"ln1.w", false}},           {"ln_1.bias", {"ln1.b", false}},
      {"ln_2.weight", {"ln2.w", false}},           {"ln_2.bias", {"ln2.b", false}},
      {"attn.c_attn.weight", {"attn.qkv.w", true}}, {"attn.c_attn.bias", {"attn.qkv.b", false}},
      {"attn.c_proj.weight", {"attn.o.w", true}},   {"attn.c_proj.bias", {"attn.o.b", false}},
      {"mlp.c_fc.weight", {"mlp.up.w", true}},      {"mlp.c_fc.bias", {"mlp.up.b", false}},
      {"mlp.c_proj.weight", {"mlp.down.w", true}},  {"mlp.c_proj.bias", {"mlp.down.b", false}},
  };
  if (r == "attn.bias" || r == "attn.masked_bias") return "";  // causal mask buffers
  auto it = m.find(r);
  if (it == m.end()) die("unrecognized GPT-2 tensor: " + n);
  transpose = it->second.second;
  return p + it->second.first;
}

std::string map_llama(const std::string& n, bool& transpose) {
  transpose = false;
  if (n == "model.embed_tokens.weight") return "tok_emb";
  if (n == "model.norm.weight") return "norm_f.w";
  if (n == "lm_head.weight") return "lm_head.w";
  int l;
  std::string r;
  if (!split_layer(n, "model.layers.", l, r)) die("unrecognized LLaMA tensor: " + n);
  if (ends_with(r, "rotary_emb.inv_freq")) return "";
  static const std::map<std::string, std::string> m = {
      {"input_layernorm.weight", "attn_norm.w"},
      {"self_attn.q_proj.weight", "attn.q.w"},
      {"self_attn.k_proj.weight", "attn.k.w"},
      {"self_attn.v_proj.weight", "attn.v.w"},
      {"self_attn.o_proj.weight", "attn.o.w"},
      {"post_attention_layernorm.weight", "mlp_norm.w"},
      {"mlp.gate_proj.weight", "mlp.gate.w"},
      {"mlp.up_proj.weight", "mlp.up.w"},
      {"mlp.down_proj.weight", "mlp.down.w"},
  };
  auto it = m.find(r);
  if (it == m.end()) die("unrecognized LLaMA tensor: " + n);
  return "l" + std::to_string(l) + "." + it->second;
}

int32_t json_int(const Json& cfg, const std::string& k) { return int32_t(cfg.at(k).number()); }

int32_t json_token_id(const Json& cfg, const std::string& k, int32_t def) {
  if (!cfg.has(k)) return def;
  const Json& v = cfg.at(k);
  if (v.kind == Json::Num) return int32_t(v.num);
  if (v.kind == Json::Arr && !v.arr.empty() && v.arr[0].kind == Json::Num) return int32_t(v.arr[0].num);
  return def;
}

Config read_config(const Json& cfg) {
  Config c{};
  const std::string& mt = cfg.at("model_type").string();
  if (mt == "gpt2") {
    c.arch = Arch::GPT2;
    c.d_model = json_int(cfg, "n_embd");
    c.n_layers = json_int(cfg, "n_layer");
    c.n_heads = json_int(cfg, "n_head");
    c.n_kv_heads = c.n_heads;
    c.head_dim = c.d_model / c.n_heads;
    c.d_ff = 4 * c.d_model;
    c.max_ctx = json_int(cfg, "n_positions");
    c.vocab_size = json_int(cfg, "vocab_size");
    c.norm_eps = cfg.has("layer_norm_epsilon") ? float(cfg.at("layer_norm_epsilon").number()) : 1e-5f;
    c.bos_id = 50256;
    c.eos_id = 50256;
  } else if (mt == "llama") {
    c.arch = Arch::LLAMA;
    c.d_model = json_int(cfg, "hidden_size");
    c.n_layers = json_int(cfg, "num_hidden_layers");
    c.n_heads = json_int(cfg, "num_attention_heads");
    c.n_kv_heads = cfg.has("num_key_value_heads") ? json_int(cfg, "num_key_value_heads") : c.n_heads;
    c.head_dim = c.d_model / c.n_heads;
    c.d_ff = json_int(cfg, "intermediate_size");
    c.max_ctx = json_int(cfg, "max_position_embeddings");
    c.vocab_size = json_int(cfg, "vocab_size");
    c.norm_eps = cfg.has("rms_norm_eps") ? float(cfg.at("rms_norm_eps").number()) : 1e-5f;
    c.rope_theta = (cfg.has("rope_theta") && cfg.at("rope_theta").kind == Json::Num)
                       ? float(cfg.at("rope_theta").number())
                       : 10000.0f;
    c.bos_id = json_token_id(cfg, "bos_token_id", 1);
    c.eos_id = json_token_id(cfg, "eos_token_id", 2);
  } else {
    die("unsupported model_type '" + mt + "' (expected gpt2 or llama)");
  }
  return c;
}

uint64_t align64(uint64_t x) { return (x + 63) & ~uint64_t(63); }

void write_all(FILE* f, const void* p, size_t n, const std::string& path) {
  if (n && std::fwrite(p, 1, n, f) != n) die("write failed: " + path + ": " + std::strerror(errno));
}

void pad_to(FILE* f, uint64_t& pos, uint64_t target, const std::string& path) {
  static const char zeros[64] = {};
  while (pos < target) {
    size_t k = size_t(std::min<uint64_t>(64, target - pos));
    write_all(f, zeros, k, path);
    pos += k;
  }
}

int run(int argc, char** argv) {
  std::string in_dir, out_path, quant;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--quant") {
      if (i + 1 >= argc) die("--quant needs an argument (q8|q4)");
      quant = argv[++i];
    } else if (starts_with(a, "--quant=")) {
      quant = a.substr(8);
    } else if (a == "-h" || a == "--help") {
      pos.clear();
      break;
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() != 2) {
    std::fprintf(stderr, "usage: tinyllm-convert <hf_model_dir> <out.tllm> [--quant q8|q4]\n");
    return 2;
  }
  in_dir = pos[0];
  out_path = pos[1];
  DType qtype = DType::F32;
  if (quant == "q8") qtype = DType::Q8;
  else if (quant == "q4") qtype = DType::Q4;
  else if (!quant.empty()) die("unknown --quant '" + quant + "' (expected q8 or q4)");

  std::string cfg_text = read_file(in_dir + "/config.json");
  Json cfg = JsonParser(cfg_text.data(), cfg_text.size()).parse();
  Config config = read_config(cfg);

  Safetensors st;
  open_safetensors(in_dir + "/model.safetensors", st);

  // Build output tensor list.
  std::vector<OutTensor> outs;
  for (auto& kv : st.tensors) {
    bool tr = false;
    std::string name = config.arch == Arch::GPT2 ? map_gpt2(kv.first, tr) : map_llama(kv.first, tr);
    if (name.empty()) continue;
    if (name.size() >= sizeof(TensorEntry::name)) die("canonical name too long: " + name);
    OutTensor o;
    o.name = name;
    o.src = &kv.second;
    o.shape = kv.second.shape;
    if (o.shape.empty() || o.shape.size() > 4) die("tensor " + kv.first + ": unsupported ndim");
    if (tr) {
      if (o.shape.size() != 2) die("tensor " + kv.first + ": transpose needs 2D");
      std::swap(o.shape[0], o.shape[1]);
    }
    o.transpose = tr;
    o.dtype = DType::F32;
    bool quantizable = o.shape.size() == 2 && ends_with(name, ".w") && name != "tok_emb" &&
                       name != "pos_emb";
    if (qtype != DType::F32 && quantizable) {
      if (o.shape[1] % kQK == 0) {
        o.dtype = qtype;
      } else {
        std::fprintf(stderr, "warning: %s row length %lld not divisible by %d, keeping F32\n",
                     name.c_str(), (long long)o.shape[1], kQK);
      }
    }
    outs.push_back(std::move(o));
  }
  // Read source sequentially.
  std::sort(outs.begin(), outs.end(),
            [](const OutTensor& a, const OutTensor& b) { return a.src->data < b.src->data; });

  // Sanity: required names present.
  {
    std::map<std::string, bool> have;
    for (auto& o : outs) have[o.name] = true;
    std::vector<std::string> need = {"tok_emb"};
    if (config.arch == Arch::GPT2) {
      need.insert(need.end(), {"pos_emb", "ln_f.w", "ln_f.b"});
    } else {
      need.insert(need.end(), {"norm_f.w", "lm_head.w"});
    }
    for (int l = 0; l < config.n_layers; ++l) {
      std::string p = "l" + std::to_string(l) + ".";
      if (config.arch == Arch::GPT2) {
        for (auto s : {"ln1.w", "ln1.b", "attn.qkv.w", "attn.qkv.b", "attn.o.w", "attn.o.b", "ln2.w",
                       "ln2.b", "mlp.up.w", "mlp.up.b", "mlp.down.w", "mlp.down.b"})
          need.push_back(p + s);
      } else {
        for (auto s : {"attn_norm.w", "attn.q.w", "attn.k.w", "attn.v.w", "attn.o.w", "mlp_norm.w",
                       "mlp.gate.w", "mlp.up.w", "mlp.down.w"})
          need.push_back(p + s);
      }
    }
    for (auto& n : need)
      if (!have.count(n)) die("source model is missing tensor for '" + n + "'");
  }

  // Layout.
  FileHeader hdr{};
  std::memcpy(hdr.magic, kMagic, 4);
  hdr.version = kVersion;
  hdr.n_tensors = uint32_t(outs.size());
  hdr.config = config;
  hdr.index_offset = sizeof(FileHeader);
  hdr.data_offset = align64(hdr.index_offset + outs.size() * sizeof(TensorEntry));
  uint64_t cur = 0;
  for (auto& o : outs) {
    o.offset = cur;
    int64_t numel = 1;
    for (auto d : o.shape) numel *= d;
    if (o.dtype == DType::F32) o.nbytes = uint64_t(numel) * 4;
    else o.nbytes = uint64_t(o.shape[0]) * quant_row_bytes(o.dtype, o.shape[1]);
    cur = align64(cur + o.nbytes);
  }

  std::vector<TensorEntry> index(outs.size());
  for (size_t i = 0; i < outs.size(); ++i) {
    TensorEntry& e = index[i];
    std::memset(&e, 0, sizeof(e));
    std::memcpy(e.name, outs[i].name.c_str(), outs[i].name.size());
    e.dtype = uint32_t(outs[i].dtype);
    e.ndim = uint32_t(outs[i].shape.size());
    for (size_t d = 0; d < outs[i].shape.size(); ++d) e.shape[d] = outs[i].shape[d];
    e.offset = outs[i].offset;
    e.nbytes = outs[i].nbytes;
  }

  std::string tmp_path = out_path + ".tmp";
  FILE* f = std::fopen(tmp_path.c_str(), "wb");
  if (!f) die("cannot create " + tmp_path + ": " + std::strerror(errno));
  std::unique_ptr<FILE, int (*)(FILE*)> guard(f, std::fclose);
  uint64_t fpos = 0;
  write_all(f, &hdr, sizeof(hdr), tmp_path);
  fpos += sizeof(hdr);
  write_all(f, index.data(), index.size() * sizeof(TensorEntry), tmp_path);
  fpos += index.size() * sizeof(TensorEntry);
  pad_to(f, fpos, hdr.data_offset, tmp_path);

  const char* arch_name = config.arch == Arch::GPT2 ? "gpt2" : "llama";
  std::printf("converting %s (%s): %zu tensors, d_model=%d n_layers=%d n_heads=%d n_kv=%d d_ff=%d "
              "vocab=%d ctx=%d -> %s [%s]\n",
              in_dir.c_str(), arch_name, outs.size(), config.d_model, config.n_layers,
              config.n_heads, config.n_kv_heads, config.d_ff, config.vocab_size, config.max_ctx,
              out_path.c_str(), quant.empty() ? "f32" : quant.c_str());

  std::vector<float> buf, tbuf;
  std::vector<uint8_t> qbuf;
  for (size_t i = 0; i < outs.size(); ++i) {
    const OutTensor& o = outs[i];
    pad_to(f, fpos, hdr.data_offset + o.offset, tmp_path);
    size_t numel = 1;
    for (auto d : o.shape) numel *= size_t(d);
    buf.resize(numel);
    to_f32(*o.src, buf.data());
    const float* src = buf.data();
    if (o.transpose) {
      // src is [in, out] -> [out, in]
      int64_t rows = o.shape[0], cols = o.shape[1];  // out, in
      tbuf.resize(numel);
      for (int64_t r = 0; r < cols; ++r)
        for (int64_t c = 0; c < rows; ++c) tbuf[size_t(c) * cols + r] = buf[size_t(r) * rows + c];
      src = tbuf.data();
    }
    if (o.dtype == DType::F32) {
      write_all(f, src, numel * 4, tmp_path);
    } else {
      int64_t rows = o.shape[0], cols = o.shape[1];
      size_t rb = quant_row_bytes(o.dtype, cols);
      qbuf.resize(size_t(rows) * rb);
      for (int64_t r = 0; r < rows; ++r) {
        if (o.dtype == DType::Q8) quantize_row_q8(src + r * cols, qbuf.data() + r * rb, cols);
        else quantize_row_q4(src + r * cols, qbuf.data() + r * rb, cols);
      }
      write_all(f, qbuf.data(), qbuf.size(), tmp_path);
    }
    fpos += o.nbytes;
    if ((i + 1) % 16 == 0 || i + 1 == outs.size()) {
      std::printf("\r  [%zu/%zu] %.1f MB written", i + 1, outs.size(), fpos / 1e6);
      std::fflush(stdout);
    }
  }
  pad_to(f, fpos, align64(fpos), tmp_path);
  if (std::fflush(f) != 0) die("flush failed: " + tmp_path);
  guard.reset();
  if (std::rename(tmp_path.c_str(), out_path.c_str()) != 0)
    die("rename " + tmp_path + " -> " + out_path + " failed: " + std::strerror(errno));
  std::printf("\ndone: %s, %zu tensors, %.1f MB\n", out_path.c_str(), outs.size(), fpos / 1e6);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tinyllm-convert: error: %s\n", e.what());
    return 1;
  }
}
