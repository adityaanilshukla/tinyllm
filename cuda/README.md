# tinyllm CUDA kernels

Hand-written fp32 kernels (`ops_cuda.h`, namespace `tinyllm::cuda`, device pointers, async on a stream):

- `matmul.cu`: tiled shared-memory GEMM `C[M,N] = A[M,K] W[N,K]^T + bias` (64x64 block tile, BK=16, 4x4 register tile per thread, 256 threads), and a warp-per-row matvec for M=1 (float4 loads when aligned).
- `attention.cu`: fused single-query GQA attention over a KV cache, one block per head.
- `layernorm.cu`: fused layer norm, one block per row.
- `softmax.cu`: stable row softmax, one block per row.

Build and benchmark (needs nvcc / CUDA 12 and cuBLAS):

```
cmake -S . -B build-cuda -G Ninja -DTINYLLM_CUDA=ON && cmake --build build-cuda && ./build-cuda/cuda/tinyllm-bench-cuda
```

Override the GPU targets with `-DCMAKE_CUDA_ARCHITECTURES=86` (default `70;80;86;89;90`).
The benchmark compares each kernel with the CPU ops in `ops.h` and, for matmul, with
`cublasSgemm`, printing time, throughput and max absolute error.
