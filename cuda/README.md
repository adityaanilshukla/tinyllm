# tinyllm CUDA kernels

Hand-written fp32 kernels (`ops_cuda.h`, namespace `tinyllm::cuda`, device pointers, async on a stream):

- `matmul.cu`: tiled shared-memory GEMM `C[M,N] = A[M,K] W[N,K]^T + bias` (64x64 block tile, BK=16, 4x4 register tile per thread, 256 threads), and a warp-per-row matvec for M=1 (float4 loads when aligned).
- `attention.cu`: fused single-query GQA attention over a KV cache, one block per head.
- `layernorm.cu`: fused layer norm, one block per row.
- `softmax.cu`: stable row softmax, one block per row.

Build and benchmark (needs nvcc, cuBLAS, CMake 3.24+, and the SentencePiece dev package for the CPU side; Blackwell cards such as the RTX 50 series need CUDA 12.8+):

```
cmake -S . -B build-cuda -G Ninja -DTINYLLM_CUDA=ON && cmake --build build-cuda && ./build-cuda/cuda/tinyllm-bench-cuda
```

By default the kernels are compiled for the GPU in the build machine (`native`).
Override with e.g. `-DCMAKE_CUDA_ARCHITECTURES="86;120"`. These kernels have not
been compiled yet, so expect fixes on the first build.
The benchmark compares each kernel with the CPU ops in `ops.h` and, for matmul, with
`cublasSgemm`, printing time, throughput and max absolute error.
