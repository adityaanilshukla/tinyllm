#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "tinyllm/tensor.h"

namespace tinyllm {

enum class Arch : uint32_t { GPT2 = 1, LLAMA = 2 };

// POD; written to disk as-is inside FileHeader.
struct Config {
  Arch arch = Arch::GPT2;
  int32_t vocab_size = 0;
  int32_t d_model = 0;
  int32_t n_layers = 0;
  int32_t n_heads = 0;
  int32_t n_kv_heads = 0;  // == n_heads for GPT-2
  int32_t head_dim = 0;    // d_model / n_heads
  int32_t d_ff = 0;
  int32_t max_ctx = 0;
  float norm_eps = 1e-5f;
  float rope_theta = 10000.0f;  // LLaMA only
  int32_t bos_id = -1;
  int32_t eos_id = -1;
};

// .tllm file layout (little-endian):
//   FileHeader
//   TensorEntry[n_tensors]           starting at index_offset
//   tensor data                      starting at data_offset (64-byte aligned);
//                                    each tensor's offset (relative to data_offset)
//                                    is also 64-byte aligned, so mmap gives aligned views.
constexpr char kMagic[4] = {'T', 'L', 'L', 'M'};
constexpr uint32_t kVersion = 1;

struct FileHeader {
  char magic[4];
  uint32_t version;
  uint32_t n_tensors;
  uint32_t reserved;
  Config config;
  uint64_t index_offset;
  uint64_t data_offset;
};

struct TensorEntry {
  char name[64];  // NUL-terminated canonical name (see below)
  uint32_t dtype; // DType
  uint32_t ndim;
  int64_t shape[4];
  uint64_t offset;  // relative to data_offset
  uint64_t nbytes;
};

// Canonical tensor names. All linear weights are [out, in]. The converter transposes
// GPT-2's Conv1D weights (HF stores them [in, out]).
//
// GPT-2:  tok_emb [vocab, d]   pos_emb [max_ctx, d]   ln_f.w  ln_f.b [d]
//   per layer i ("l{i}."):
//     ln1.w ln1.b [d]
//     attn.qkv.w [3d, d]  attn.qkv.b [3d]     (rows: q then k then v)
//     attn.o.w   [d, d]   attn.o.b   [d]
//     ln2.w ln2.b [d]
//     mlp.up.w   [4d, d]  mlp.up.b   [4d]
//     mlp.down.w [d, 4d]  mlp.down.b [d]
//   LM head is tied to tok_emb.
//
// LLaMA:  tok_emb [vocab, d]   norm_f.w [d]   lm_head.w [vocab, d]
//   per layer i ("l{i}."):
//     attn_norm.w [d]
//     attn.q.w [n_heads*hd, d]  attn.k.w [n_kv*hd, d]  attn.v.w [n_kv*hd, d]  attn.o.w [d, n_heads*hd]
//     mlp_norm.w [d]
//     mlp.gate.w [d_ff, d]  mlp.up.w [d_ff, d]  mlp.down.w [d, d_ff]
//   q/k weights keep HF's layout, so RoPE uses HF's rotate_half pairing (i, i + hd/2).
//
// Quantization (--quant): only 2D linear weights (names ending in ".w" with ndim 2,
// plus lm_head.w) are quantized. tok_emb, pos_emb, norms and biases stay F32.

// Read-only mmapped .tllm file. Implemented in src/loader.cpp.
class WeightFile {
 public:
  static std::shared_ptr<WeightFile> open(const std::string& path);  // throws std::runtime_error
  ~WeightFile();
  const Config& config() const { return config_; }
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const Tensor& get(const std::string& name) const;  // throws if missing

 private:
  WeightFile() = default;
  Config config_{};
  std::unordered_map<std::string, Tensor> tensors_;
  void* map_ = nullptr;
  size_t map_size_ = 0;
};

}  // namespace tinyllm
