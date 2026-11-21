# tinyllm architecture

Internal design notes. The headers in `include/tinyllm/` are the contracts between the
pieces below; read them before changing anything.

## Overview

tinyllm is a from-scratch C++17 LLM inference engine with no ML-framework dependencies.
It runs GPT-2 small and LLaMA-family models (tested on TinyLlama-1.1B) on the CPU in f32
or INT8/INT4 weights. Hand-written CUDA kernels live in `cuda/` (off by default); they
build and run on NVIDIA hardware, verified on an RTX 5060 Ti (sm_120) with CUDA 13.5 and
GCC 16.

## Build and run

```sh
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure    # installed GTest, else FetchContent v1.15.2

# HF model dir (config.json + model.safetensors) -> .tllm
./build/tinyllm-convert models/gpt2 gpt2.tllm
./build/tinyllm-convert models/tinyllama tinyllama-q4.tllm --quant q4   # or q8

./build/tinyllm-cli -m gpt2.tllm -p "Hello" -n 64 --temp 0
./build/tinyllm-cli -m tinyllama-q4.tllm --chat -p "What is a KV cache?" -n 128
```

SentencePiece is the only external dependency (`pacman -S sentencepiece`,
`apt install libsentencepiece-dev`, or `brew install sentencepiece`). `-t` overrides the
tokenizer location; defaults are `models/gpt2` (dir with `vocab.json` + `merges.txt`) and
`models/tinyllama/tokenizer.model`. Other CLI flags: `--top-k`, `--top-p`, `--seed`,
`--ctx`. `TINYLLM_THREADS` sets the thread count of the F32 pool only; the Q8/Q4 pool in
`src/quant.cpp` always uses the hardware thread count (capped at 32). The CUDA build is
`cmake -S . -B build-cuda -G Ninja -DTINYLLM_CUDA=ON && cmake --build build-cuda &&
./build-cuda/cuda/tinyllm-bench-cuda`.

## Components

- **Weight format** (`format.h`):
  - Layout: a header with a POD `Config`, a fixed-size `TensorEntry` index, then
    64-byte-aligned data.
  - Loading: `WeightFile` (`src/loader.cpp`) mmaps the file and hands out zero-copy
    `Tensor` views.
  - Conversion: `tools/convert.cpp` maps HF names to canonical names (`tok_emb`,
    `l{i}.attn.qkv.w`, ..., listed in `format.h`) and converts bf16/f16 to f32. It
    transposes GPT-2's Conv1D weights, so **every linear weight is `[out, in]`**.
  - Quantization: `--quant` quantizes only 2D linear weights. Embeddings, norms, and
    biases stay F32.
- **Models** (`model.h`):
  - `load_model()` (`src/model.cpp`) dispatches on `Config::arch` to `make_gpt2`
    (`src/gpt2.cpp`) or `make_llama` (`src/llama.cpp`).
  - `Model::forward(token, pos)` processes **one token at a time**, including during
    prefill. Positions must be fed in order. It returns logits.
  - Each model allocates all activations and its `KVCache` (`[layer][pos][kv_dim]`) up
    front from one `Arena`.
- **Ops** (`ops.h`):
  - `linear()` is the single matvec entry point and dispatches on `Tensor::dtype`: F32
    runs in `src/ops_cpu.cpp`, Q8/Q4 in `matvec_q8`/`matvec_q4` (`src/quant.cpp`).
  - `ops_cpu.cpp` and `quant.cpp` each have their **own** persistent thread pool. Do not
    call one pool from inside the other.
  - `attention()` handles both MHA and GQA.
  - LLaMA-only ops (rmsnorm, RoPE with HF `rotate_half` pairing, SwiGLU) are in
    `src/ops_llama.cpp`.
- **Quantization** (`quant.h`): blocks of 32 along the input dim. Q8 is an fp32 scale plus
  int8 values. Q4 is an fp32 scale plus nibbles, where the low nibble is element j and the
  high nibble is element j+16. The quantized matvec also quantizes the activation vector to
  int8 (W8A8 / W4A8). The converter and the kernels both depend on this exact layout.
- **Tokenizers** (`tokenizer.h`):
  - GPT-2 byte-level BPE is hand-written (`src/bpe_gpt2.cpp`). LLaMA wraps SentencePiece
    (`src/sp_tokenizer.cpp`).
  - `encode` never adds BOS; the CLI prepends it for LLaMA.
- **CUDA** (`cuda/`): tiled shared-memory GEMM, a warp-per-row matvec for the single-query
  case, fused single-query GQA attention, fused LayerNorm, and a stable softmax, plus a
  benchmark against the CPU ops and cuBLAS. Nothing in the CPU engine calls the CUDA code
  yet; the kernels are benchmarked, not integrated into `Model::forward`.
- **CMake:** it globs `src/*.cpp` into the `tinyllm` library, so new source files need no
  CMake edit. `absl_status` is linked explicitly when pkg-config finds it, because some
  `sentencepiece.pc` files do not list abseil. CUDA targets default to `native` (the build
  machine's GPU).

## Verification

- **Element-wise vs PyTorch/HuggingFace:** `tests/reference_pytorch_test.cpp` checks GPT-2
  and TinyLlama intermediates (every layer's norm/qkv/attn/mlp/residual, final norm,
  logits) and the shipped `Model::forward()` logits element-wise against a HuggingFace
  reference, plus op-level checks (LayerNorm, GELU, softmax, MHA/GQA attention, RMSNorm,
  RoPE, SwiGLU) against torch/transformers. Worst case: GPT-2 logits 9e-5, TinyLlama
  shipped logits 3e-5, ops 1e-7 to 5e-6. References come from `scripts/gen_reference.py`
  (isolated `uv` venv, Python 3.12, CPU torch); the tests skip when `models/ref/` is
  absent.
- **Quantized fidelity:** the Q8/Q4 tests run the shipped quantized forward and check the
  greedy prediction against the f32 HF logits (Q8 matches argmax exactly, every precision
  stays within the f32 top-5), reporting the logit divergence as the measured quantization
  cost. A W8A8/W4A8 pipeline cannot be pinned f32-tight; see the comment above
  `RunQuantFidelity`.
- **CUDA vs CPU and cuBLAS:** `bench/bench_cuda.cpp` times every kernel against the CPU
  ops and matmul against `cublasSgemm`; `cuda/correctness_test.cpp` (the
  `tinyllm-correctness-cuda` ctest) pins kernel-vs-CPU error within tolerances.
- **Unit tests:** ops, quant layout and kernels, sampler, loader, and tiny random
  GPT-2/LLaMA models against naive double-precision references written in the tests.
  Model-dependent tests skip when `models/` is missing (`TINYLLM_MODELS_DIR` overrides the
  path).

## Conventions

- Prefer the public headers in `include/tinyllm/` over reaching into internals.
- Keep the two thread pools separate; never call one from inside the other.
- The Q4 nibble layout and block size are shared by the converter and the kernels;
  changing either breaks existing `.tllm` files.
