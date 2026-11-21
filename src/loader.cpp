// Read-only mmapped .tllm weight file.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include "tinyllm/format.h"
#include "tinyllm/quant.h"

namespace tinyllm {

namespace {

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
  throw std::runtime_error("WeightFile: " + path + ": " + msg);
}

const char* dtype_name(uint32_t d) {
  switch (d) {
    case 0: return "F32";
    case 1: return "Q8";
    case 2: return "Q4";
    default: return "?";
  }
}

}  // namespace

std::shared_ptr<WeightFile> WeightFile::open(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail(path, std::string("cannot open: ") + std::strerror(errno));
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    int e = errno;
    ::close(fd);
    fail(path, std::string("fstat failed: ") + std::strerror(e));
  }
  size_t size = static_cast<size_t>(st.st_size);
  if (size < sizeof(FileHeader)) {
    ::close(fd);
    fail(path, "file too small (" + std::to_string(size) + " bytes) to be a .tllm file");
  }
  void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  int map_errno = errno;
  ::close(fd);
  if (map == MAP_FAILED) fail(path, std::string("mmap failed: ") + std::strerror(map_errno));

  // From here on the WeightFile owns the mapping (destructor unmaps on throw).
  std::shared_ptr<WeightFile> wf(new WeightFile());
  wf->map_ = map;
  wf->map_size_ = size;

  const auto* base = static_cast<const uint8_t*>(map);
  FileHeader hdr;
  std::memcpy(&hdr, base, sizeof(hdr));
  if (std::memcmp(hdr.magic, kMagic, 4) != 0) fail(path, "bad magic (not a .tllm file)");
  if (hdr.version != kVersion)
    fail(path, "unsupported version " + std::to_string(hdr.version) + " (expected " +
                   std::to_string(kVersion) + ")");
  if (hdr.config.arch != Arch::GPT2 && hdr.config.arch != Arch::LLAMA)
    fail(path, "unknown arch " + std::to_string(static_cast<uint32_t>(hdr.config.arch)));

  uint64_t index_bytes = uint64_t(hdr.n_tensors) * sizeof(TensorEntry);
  if (hdr.index_offset > size || index_bytes > size - hdr.index_offset)
    fail(path, "tensor index out of bounds (truncated file?)");
  if (hdr.data_offset > size) fail(path, "data_offset beyond end of file (truncated file?)");
  if (hdr.data_offset % 64 != 0) fail(path, "data_offset is not 64-byte aligned");

  wf->config_ = hdr.config;
  wf->tensors_.reserve(hdr.n_tensors);
  const uint64_t data_size = size - hdr.data_offset;
  for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
    TensorEntry e;
    std::memcpy(&e, base + hdr.index_offset + uint64_t(i) * sizeof(TensorEntry), sizeof(e));
    if (std::memchr(e.name, '\0', sizeof(e.name)) == nullptr)
      fail(path, "tensor #" + std::to_string(i) + " has an unterminated name");
    std::string name(e.name);
    if (e.dtype > 2)
      fail(path, "tensor '" + name + "' has unknown dtype " + std::to_string(e.dtype));
    if (e.ndim < 1 || e.ndim > 4)
      fail(path, "tensor '" + name + "' has invalid ndim " + std::to_string(e.ndim));
    if (e.offset > data_size || e.nbytes > data_size - e.offset)
      fail(path, "tensor '" + name + "' data out of bounds (truncated file?)");

    Tensor t;
    t.dtype = static_cast<DType>(e.dtype);
    t.ndim = static_cast<int32_t>(e.ndim);
    for (int d = 0; d < 4; ++d) t.shape[d] = d < t.ndim ? e.shape[d] : 0;
    t.data = base + hdr.data_offset + e.offset;
    t.nbytes = static_cast<size_t>(e.nbytes);

    // Sanity-check the byte size against the shape.
    size_t expect;
    if (t.dtype == DType::F32) {
      expect = static_cast<size_t>(t.numel()) * 4;
    } else {
      if (t.ndim != 2) fail(path, "quantized tensor '" + name + "' is not 2D");
      expect = static_cast<size_t>(t.shape[0]) * quant_row_bytes(t.dtype, t.shape[1]);
    }
    if (expect != t.nbytes)
      fail(path, "tensor '" + name + "' (" + dtype_name(e.dtype) + ") has " +
                     std::to_string(t.nbytes) + " bytes, expected " + std::to_string(expect));

    if (!wf->tensors_.emplace(name, t).second) fail(path, "duplicate tensor '" + name + "'");
  }
  return wf;
}

WeightFile::~WeightFile() {
  if (map_ && map_ != MAP_FAILED) munmap(map_, map_size_);
}

const Tensor& WeightFile::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("WeightFile: missing tensor '" + name + "'");
  return it->second;
}

}  // namespace tinyllm
