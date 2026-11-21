# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

tinyllm is a from-scratch C++17 LLM inference engine with no ML-framework dependencies. It runs GPT-2 small and LLaMA-family models (tested on TinyLlama-1.1B) on CPU, in f32 or INT8/INT4 weights. Hand-written CUDA kernels live in `cuda/` but are off by default. They build and run on NVIDIA hardware, verified on an RTX 5060 Ti (sm_120) with CUDA 13.5 and GCC 16. A GoogleTest suite lives in `tests/`.

## Commands

```sh
brew install sentencepiece                      # only external dependency (pulls in abseil)
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure       # GoogleTest via brew googletest, else FetchContent v1.15.2

# HF model dir (config.json + model.safetensors) -> .tllm
./build/tinyllm-convert models/gpt2 gpt2.tllm
./build/tinyllm-convert models/tinyllama tinyllama-q4.tllm --quant q4   # or q8

./build/tinyllm-cli -m gpt2.tllm -p "Hello" -n 64 --temp 0
./build/tinyllm-cli -m tinyllama-q4.tllm --chat -p "What is a KV cache?" -n 128
```

- `-t` overrides the tokenizer location. Defaults are `models/gpt2` (dir with `vocab.json` + `merges.txt`) and `models/tinyllama/tokenizer.model`.
- Other CLI flags: `--top-k`, `--top-p`, `--seed`, `--ctx`. `TINYLLM_THREADS` sets the thread count of the F32 pool only; the Q8/Q4 pool in `src/quant.cpp` always uses the hardware thread count (capped at 32).
- Models go in `models/` (gitignored), downloaded from `openai-community/gpt2` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- CUDA build, on a Linux box with nvcc: `cmake -S . -B build-cuda -G Ninja -DTINYLLM_CUDA=ON && cmake --build build-cuda && ./build-cuda/cuda/tinyllm-bench-cuda`

## Architecture

The contracts live in `include/tinyllm/`; read them before changing anything.

- **Weight format** (`format.h`):
  - Layout: a header with a POD `Config`, a fixed-size `TensorEntry` index, then 64-byte-aligned data.
  - Loading: `WeightFile` (`src/loader.cpp`) mmaps the file and hands out zero-copy `Tensor` views.
  - Conversion: `tools/convert.cpp` maps HF names to canonical names (`tok_emb`, `l{i}.attn.qkv.w`, …, listed in `format.h`) and converts bf16/f16 to f32. It transposes GPT-2's Conv1D weights, so **every linear weight is `[out, in]`**.
  - Quantization: `--quant` quantizes only 2D linear weights. Embeddings, norms and biases stay F32.
- **Models** (`model.h`):
  - `load_model()` (`src/model.cpp`) dispatches on `Config::arch` to `make_gpt2` (`src/gpt2.cpp`) or `make_llama` (`src/llama.cpp`).
  - `Model::forward(token, pos)` processes **one token at a time**, including during prefill. Positions must be fed in order. It returns logits.
  - Each model allocates all activations and its `KVCache` (`[layer][pos][kv_dim]`) up front from one `Arena`.
- **Ops** (`ops.h`):
  - `linear()` is the single matvec entry point and dispatches on `Tensor::dtype`: F32 runs in `src/ops_cpu.cpp`, Q8/Q4 in `matvec_q8`/`matvec_q4` (`src/quant.cpp`).
  - `ops_cpu.cpp` and `quant.cpp` each have their **own** persistent thread pool. Don't call one pool from inside the other.
  - `attention()` handles both MHA and GQA.
  - LLaMA-only ops (rmsnorm, RoPE with HF `rotate_half` pairing, SwiGLU) are in `src/ops_llama.cpp`.
- **Quantization** (`quant.h`): blocks of 32 along the input dim. Q8 is an fp32 scale plus int8 values. Q4 is an fp32 scale plus nibbles, where the low nibble is element j and the high nibble is element j+16. The converter and the kernels both depend on this exact layout.
- **Tokenizers** (`tokenizer.h`):
  - GPT-2 byte-level BPE is hand-written (`src/bpe_gpt2.cpp`). LLaMA wraps SentencePiece (`src/sp_tokenizer.cpp`).
  - `encode` never adds BOS; the CLI prepends it for LLaMA.
- **CMake:** it globs `src/*.cpp` into the `tinyllm` library, so new source files need no CMake edit. Homebrew's `sentencepiece.pc` doesn't list abseil, so `absl_status` is linked explicitly when pkg-config finds it (distro packages may lack it). CUDA targets default to `native` (the build machine's GPU).

## Status

- **Checked:** GPT-2 f32 logits were compared once against an independent reference implementation that reads the safetensors file directly, and matched to within 6e-5. That check was a throwaway, not a test kept in the repo.
- **Tests (`tests/`):** ops, quant layout and kernels, sampler, loader, and tiny random GPT-2/LLaMA models are checked against naive double-precision references written in the tests. The GPT-2 end-to-end test pins greedy tokens and logits taken from this engine, not from HF. Tests that need `models/` skip when it is missing (`TINYLLM_MODELS_DIR` overrides the path).
- **Element-wise vs PyTorch/HF:** `tests/reference_pytorch_test.cpp` checks GPT-2 intermediates (every layer's ln/qkv/attn/mlp/residual, `ln_f`, logits) and the shipped `Model::forward()` logits element-wise against a HuggingFace reference, plus op-level checks (layernorm, GELU, softmax, MHA/GQA attention, RMSNorm, RoPE, SwiGLU) against torch/HF. Worst case: GPT-2 logits 9e-5, residual 2.4e-4, ops 1e-7 to 5e-6. References are generated by `scripts/gen_reference.py` (isolated `uv` venv, Python 3.12, CPU torch); the test skips when the dumps under `models/ref/` are absent.
- **CUDA benchmarked vs CPU and cuBLAS:** `bench/bench_cuda.cpp` times every kernel against the CPU ops and matmul against `cublasSgemm` on the RTX 5060 Ti; `cuda/correctness_test.cpp` (the `tinyllm-correctness-cuda` ctest) pins kernel-vs-CPU error within tolerances.
- **Spot-checked only:** TinyLlama (f32/q8/q4) and quantized GPT-2 were judged by whether the generated text looked right. TinyLlama's real weights are not yet run end-to-end against HF (only op-level, with its dimensions on synthetic inputs).

## Conventions

- Milestones are tracked in an external career tracker, not in this repo.
- `project-scope.md` is local to each machine and deliberately untracked, so it may be missing. Don't create or commit it. The same goes for `todo.md`.
