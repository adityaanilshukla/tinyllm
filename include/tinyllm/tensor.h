#pragma once
#include <cstddef>
#include <cstdint>

namespace tinyllm {

enum class DType : uint32_t { F32 = 0, Q8 = 1, Q4 = 2 };

// Non-owning view. 2D weight matrices are always [out, in] row-major (rows contiguous
// along the input dim), so y = W x walks rows. Quantized tensors keep their logical
// shape; data holds shape[0] rows of quant_row_bytes(dtype, shape[1]) bytes each.
struct Tensor {
  DType dtype = DType::F32;
  int32_t ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};
  const void* data = nullptr;
  size_t nbytes = 0;

  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return n;
  }
  const float* f32() const { return static_cast<const float*>(data); }
};

}  // namespace tinyllm
