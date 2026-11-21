#include <gtest/gtest.h>

#include <cstdint>
#include <new>

#include "tinyllm/arena.h"
#include "tinyllm/model.h"

namespace tinyllm {
namespace {

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

TEST(Arena, DefaultAllocationsAre64ByteAligned) {
  Arena arena(4096);
  for (size_t bytes : {1, 3, 64, 65, 100, 7}) {
    void* p = arena.alloc(bytes);
    EXPECT_TRUE(aligned(p, 64)) << "after alloc of " << bytes << " bytes";
  }
}

TEST(Arena, AllocationsDoNotOverlap) {
  Arena arena(1024);
  auto* a = static_cast<uint8_t*>(arena.alloc(10));
  auto* b = static_cast<uint8_t*>(arena.alloc(10));
  EXPECT_GE(b, a + 10);
  EXPECT_EQ(b - a, 64);
}

TEST(Arena, SmallerAlignmentPacksTighter) {
  Arena arena(1024);
  auto* a = static_cast<uint8_t*>(arena.alloc(3, 4));
  auto* b = static_cast<uint8_t*>(arena.alloc(1, 4));
  EXPECT_TRUE(aligned(b, 4));
  EXPECT_EQ(b - a, 4);
  EXPECT_EQ(arena.used(), 5u);
}

TEST(Arena, AllocNIsTypedAndAligned) {
  Arena arena(1024);
  arena.alloc(1);
  float* f = arena.alloc_n<float>(17);
  EXPECT_TRUE(aligned(f, 64));
  for (int i = 0; i < 17; ++i) f[i] = float(i);  // writable
  EXPECT_EQ(arena.used(), 64u + 17 * sizeof(float));
}

TEST(Arena, ThrowsWhenFullAndResetReclaims) {
  Arena arena(256);
  arena.alloc(200);
  EXPECT_THROW(arena.alloc(100), std::bad_alloc);
  // Alignment padding counts against capacity too: 200 rounds up to 256.
  EXPECT_THROW(arena.alloc(1), std::bad_alloc);
  arena.reset();
  EXPECT_EQ(arena.used(), 0u);
  EXPECT_NO_THROW(arena.alloc(256));
}

TEST(Arena, KVCacheRowsAreContiguousPerLayer) {
  Arena arena(1 << 16);
  KVCache kv;
  kv.init(arena, 3, 5, 8);
  EXPECT_TRUE(aligned(kv.k, 64));
  EXPECT_TRUE(aligned(kv.v, 64));
  EXPECT_EQ(kv.k_at(0, 1) - kv.k_at(0, 0), 8);
  EXPECT_EQ(kv.k_at(1, 0) - kv.k_at(0, 0), 5 * 8);
  EXPECT_EQ(kv.v_at(2, 4) - kv.v, (2 * 5 + 4) * 8);
  // k and v don't overlap.
  EXPECT_GE(kv.v, kv.k + 3 * 5 * 8);
}

}  // namespace
}  // namespace tinyllm
