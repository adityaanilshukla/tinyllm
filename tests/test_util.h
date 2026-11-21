// Shared helpers for the test suite: deterministic random data, paths, and a minimal
// .tllm writer that follows the layout in format.h (same as tools/convert.cpp).
#pragma once
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyllm/format.h"
#include "tinyllm/quant.h"

namespace tinyllm::test {

inline std::vector<float> rand_vec(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

inline std::string env_or(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v && *v ? v : fallback;
}

inline std::string models_dir() { return env_or("TINYLLM_MODELS_DIR", TINYLLM_MODELS_DIR); }

// Per-process scratch dir under the build tree, so parallel ctest runs don't collide.
inline std::filesystem::path tmp_dir() {
  std::filesystem::path p = env_or("TINYLLM_TEST_TMPDIR", TINYLLM_TEST_TMPDIR);
  p /= "pid" + std::to_string(::getpid());
  std::filesystem::create_directories(p);
  return p;
}

// ---------------------------------------------------------------------------
// .tllm writer.

struct OutTensor {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

inline OutTensor f32_tensor(const std::string& name, std::vector<int64_t> shape, const std::vector<float>& v) {
  OutTensor t{name, DType::F32, std::move(shape), {}};
  t.bytes.resize(v.size() * sizeof(float));
  std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
  return t;
}

// Quantizes a row-major [rows, cols] matrix with the library's own quantize_row_*.
inline OutTensor quant_tensor(const std::string& name, DType dtype, int64_t rows, int64_t cols,
                              const std::vector<float>& v) {
  if (dtype == DType::F32) return f32_tensor(name, {rows, cols}, v);
  OutTensor t{name, dtype, {rows, cols}, {}};
  const size_t rb = quant_row_bytes(dtype, cols);
  t.bytes.resize(static_cast<size_t>(rows) * rb);
  for (int64_t r = 0; r < rows; ++r) {
    if (dtype == DType::Q8) quantize_row_q8(v.data() + r * cols, t.bytes.data() + r * rb, cols);
    else quantize_row_q4(v.data() + r * cols, t.bytes.data() + r * rb, cols);
  }
  return t;
}

// Dequantized copy of an OutTensor's data (identity for F32).
inline std::vector<float> dequant(const OutTensor& t) {
  int64_t n = 1;
  for (int64_t d : t.shape) n *= d;
  std::vector<float> out(static_cast<size_t>(n));
  if (t.dtype == DType::F32) {
    std::memcpy(out.data(), t.bytes.data(), t.bytes.size());
    return out;
  }
  const int64_t rows = t.shape[0], cols = t.shape[1];
  const size_t rb = quant_row_bytes(t.dtype, cols);
  for (int64_t r = 0; r < rows; ++r) dequantize_row(t.dtype, t.bytes.data() + r * rb, out.data() + r * cols, cols);
  return out;
}

inline uint64_t align64(uint64_t x) { return (x + 63) & ~uint64_t(63); }

inline void write_tllm(const std::string& path, const Config& config, const std::vector<OutTensor>& tensors) {
  FileHeader hdr{};
  std::memcpy(hdr.magic, kMagic, 4);
  hdr.version = kVersion;
  hdr.n_tensors = static_cast<uint32_t>(tensors.size());
  hdr.config = config;
  hdr.index_offset = sizeof(FileHeader);
  hdr.data_offset = align64(hdr.index_offset + tensors.size() * sizeof(TensorEntry));

  std::vector<TensorEntry> index(tensors.size());
  uint64_t cur = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const OutTensor& t = tensors[i];
    TensorEntry& e = index[i];
    std::memset(&e, 0, sizeof(e));
    if (t.name.size() >= sizeof(e.name)) throw std::runtime_error("write_tllm: name too long");
    std::memcpy(e.name, t.name.c_str(), t.name.size());
    e.dtype = static_cast<uint32_t>(t.dtype);
    e.ndim = static_cast<uint32_t>(t.shape.size());
    for (size_t d = 0; d < t.shape.size(); ++d) e.shape[d] = t.shape[d];
    e.offset = cur;
    e.nbytes = t.bytes.size();
    cur = align64(cur + e.nbytes);
  }

  std::vector<uint8_t> file(hdr.data_offset + cur, 0);
  std::memcpy(file.data(), &hdr, sizeof(hdr));
  std::memcpy(file.data() + hdr.index_offset, index.data(), index.size() * sizeof(TensorEntry));
  for (size_t i = 0; i < tensors.size(); ++i)
    std::memcpy(file.data() + hdr.data_offset + index[i].offset, tensors[i].bytes.data(), tensors[i].bytes.size());

  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("write_tllm: cannot create " + path);
  const size_t n = std::fwrite(file.data(), 1, file.size(), f);
  std::fclose(f);
  if (n != file.size()) throw std::runtime_error("write_tllm: short write to " + path);
}

}  // namespace tinyllm::test
