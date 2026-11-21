#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace tinyllm {

inline void* aligned_alloc64(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0) throw std::bad_alloc();
  return p;
}
inline void aligned_free(void* p) { std::free(p); }

// Bump allocator for activations, scratch and the KV cache. One big aligned block,
// freed all at once.
class Arena {
 public:
  explicit Arena(size_t capacity) : base_(static_cast<uint8_t*>(aligned_alloc64(capacity))), cap_(capacity) {}
  ~Arena() { aligned_free(base_); }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void* alloc(size_t bytes, size_t align = 64) {
    size_t off = (off_ + align - 1) & ~(align - 1);
    if (off + bytes > cap_) throw std::bad_alloc();
    off_ = off + bytes;
    return base_ + off;
  }
  template <class T>
  T* alloc_n(size_t n) { return static_cast<T*>(alloc(n * sizeof(T))); }
  void reset() { off_ = 0; }
  size_t used() const { return off_; }

 private:
  uint8_t* base_;
  size_t cap_;
  size_t off_ = 0;
};

}  // namespace tinyllm
