// WeightFile: write a tiny .tllm with the documented layout, load it back, and reject
// malformed files.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.h"
#include "tinyllm/format.h"

namespace tinyllm {
namespace {

namespace fs = std::filesystem;
using test::OutTensor;
using test::rand_vec;

Config tiny_config() {
  Config c;
  c.arch = Arch::LLAMA;
  c.vocab_size = 11;
  c.d_model = 64;
  c.n_layers = 1;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 16;
  c.d_ff = 96;
  c.max_ctx = 128;
  c.norm_eps = 1e-6f;
  c.rope_theta = 500000.0f;
  c.bos_id = 1;
  c.eos_id = 2;
  return c;
}

std::vector<OutTensor> tiny_tensors() {
  return {test::f32_tensor("tok_emb", {11, 64}, rand_vec(11 * 64, 1)),
          test::f32_tensor("norm_f.w", {64}, rand_vec(64, 2)),
          test::quant_tensor("l0.attn.q.w", DType::Q8, 64, 64, rand_vec(64 * 64, 3)),
          test::quant_tensor("l0.mlp.up.w", DType::Q4, 96, 64, rand_vec(96 * 64, 4)),
          test::f32_tensor("odd", {3, 1, 5}, rand_vec(15, 5))};
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

void write_bytes(const fs::path& p, const std::vector<uint8_t>& b) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}

class Loader : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = test::tmp_dir() / (std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + ".tllm");
    test::write_tllm(path_.string(), tiny_config(), tiny_tensors());
  }
  void TearDown() override { fs::remove(path_); }

  // Patch the file, then expect open() to throw with `what` in the message.
  void expect_rejected(const std::vector<uint8_t>& bytes, const std::string& what) {
    write_bytes(path_, bytes);
    try {
      WeightFile::open(path_.string());
      ADD_FAILURE() << "expected rejection containing '" << what << "'";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
    }
  }

  fs::path path_;
};

TEST_F(Loader, RoundTripsConfigAndTensors) {
  auto wf = WeightFile::open(path_.string());
  const Config& c = wf->config();
  const Config want = tiny_config();
  EXPECT_EQ(c.arch, want.arch);
  EXPECT_EQ(c.vocab_size, want.vocab_size);
  EXPECT_EQ(c.d_model, want.d_model);
  EXPECT_EQ(c.n_kv_heads, want.n_kv_heads);
  EXPECT_EQ(c.d_ff, want.d_ff);
  EXPECT_EQ(c.max_ctx, want.max_ctx);
  EXPECT_EQ(c.norm_eps, want.norm_eps);
  EXPECT_EQ(c.rope_theta, want.rope_theta);
  EXPECT_EQ(c.bos_id, 1);
  EXPECT_EQ(c.eos_id, 2);

  for (const OutTensor& t : tiny_tensors()) {
    ASSERT_TRUE(wf->has(t.name)) << t.name;
    const Tensor& v = wf->get(t.name);
    EXPECT_EQ(v.dtype, t.dtype) << t.name;
    ASSERT_EQ(v.ndim, int32_t(t.shape.size())) << t.name;
    for (size_t d = 0; d < t.shape.size(); ++d) EXPECT_EQ(v.shape[d], t.shape[d]) << t.name;
    for (size_t d = t.shape.size(); d < 4; ++d) EXPECT_EQ(v.shape[d], 0) << t.name;
    EXPECT_EQ(v.nbytes, t.bytes.size()) << t.name;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(v.data) % 64, 0u) << t.name << " not 64-byte aligned";
    EXPECT_EQ(std::memcmp(v.data, t.bytes.data(), t.bytes.size()), 0) << t.name;
  }
  EXPECT_FALSE(wf->has("nope"));
  EXPECT_THROW(wf->get("nope"), std::runtime_error);
}

TEST_F(Loader, MissingFile) {
  EXPECT_THROW(WeightFile::open((test::tmp_dir() / "does-not-exist.tllm").string()), std::runtime_error);
}

TEST_F(Loader, RejectsBadMagic) {
  auto b = read_bytes(path_);
  b[0] = 'X';
  expect_rejected(b, "bad magic");
}

TEST_F(Loader, RejectsWrongVersion) {
  auto b = read_bytes(path_);
  const uint32_t v = kVersion + 1;
  std::memcpy(b.data() + offsetof(FileHeader, version), &v, 4);
  expect_rejected(b, "unsupported version");
}

TEST_F(Loader, RejectsUnknownArch) {
  auto b = read_bytes(path_);
  const uint32_t a = 7;
  std::memcpy(b.data() + offsetof(FileHeader, config) + offsetof(Config, arch), &a, 4);
  expect_rejected(b, "unknown arch");
}

TEST_F(Loader, RejectsTruncatedFile) {
  auto b = read_bytes(path_);
  expect_rejected(std::vector<uint8_t>(b.begin(), b.begin() + 10), "too small");
  expect_rejected(std::vector<uint8_t>(b.begin(), b.end() - 100), "out of bounds");
  expect_rejected(std::vector<uint8_t>(b.begin(), b.begin() + sizeof(FileHeader) + 10), "index out of bounds");
}

TEST_F(Loader, RejectsSizeMismatch) {
  auto b = read_bytes(path_);
  FileHeader hdr;
  std::memcpy(&hdr, b.data(), sizeof(hdr));
  // Find the Q4 tensor's entry and claim one more row than its data holds.
  for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
    TensorEntry e;
    uint8_t* p = b.data() + hdr.index_offset + i * sizeof(TensorEntry);
    std::memcpy(&e, p, sizeof(e));
    if (std::string(e.name) == "l0.mlp.up.w") {
      e.shape[0] += 1;
      std::memcpy(p, &e, sizeof(e));
    }
  }
  expect_rejected(b, "expected");
}

TEST_F(Loader, RejectsDuplicateNames) {
  auto ts = tiny_tensors();
  ts.push_back(ts[1]);
  test::write_tllm(path_.string(), tiny_config(), ts);
  expect_rejected(read_bytes(path_), "duplicate");
}

TEST_F(Loader, RejectsQuantizedNon2D) {
  auto ts = tiny_tensors();
  OutTensor bad = test::quant_tensor("bad", DType::Q8, 2, 32, rand_vec(64, 9));
  bad.shape = {2, 32, 1};
  ts.push_back(bad);
  test::write_tllm(path_.string(), tiny_config(), ts);
  expect_rejected(read_bytes(path_), "not 2D");
}

}  // namespace
}  // namespace tinyllm
