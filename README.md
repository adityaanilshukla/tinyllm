# tinyllm

[![CI](https://github.com/adityaanilshukla/tinyllm/actions/workflows/ci.yml/badge.svg)](https://github.com/adityaanilshukla/tinyllm/actions/workflows/ci.yml)

A from-scratch LLM inference engine in C++17 with no ML-framework dependencies. It
converts HuggingFace checkpoints into its own mmap-friendly `.tllm` format and runs
GPT-2 small and LLaMA-family models (tested on TinyLlama-1.1B-Chat) on the CPU in f32,
INT8, or INT4, with hand-written CUDA kernels benchmarked against cuBLAS. Every layer is
verified element-wise against PyTorch/HuggingFace; the quantized paths are checked for
prediction fidelity against the f32 model.

The engine, tokenizers (GPT-2 byte-level BPE and SentencePiece), KV cache, sampler,
INT8/INT4 quantization, and CUDA kernels are all written by hand. There is no PyTorch,
ONNX, GGML, or BLAS in the inference path.

## Why this exists

The interesting problem in deploying models is rarely quality; it is latency, memory,
and cost. This engine is a vehicle for understanding that problem from the bottom up:
what a forward pass actually costs, where quantization spends its accuracy, and how much
of inference speed is memory movement rather than arithmetic. So correctness is proven,
not assumed, and every performance claim is measured.

## Key results

All numbers are produced by committed tests and benchmarks on this repository (CPU: the
host of an RTX 5060 Ti; GPU: that card, sm_120, CUDA 13.5).

- **Correctness vs HuggingFace (element-wise, f32):** GPT-2 logits within **9e-5**,
  TinyLlama-1.1B shipped-forward logits within **3e-5**, and isolated ops (LayerNorm,
  GELU, softmax, MHA/GQA attention, RMSNorm, RoPE, SwiGLU) within **1e-7 to 5e-6**.
- **INT8 is nearly free:** INT8 preserves the f32 model's greedy prediction at every
  tested position and adds only **+0.03 perplexity** on TinyLlama (+0.6%), for a
  **3.5x** smaller model on disk.
- **INT4 is a real, measured tradeoff:** **4.8x** smaller on disk and **6.1x** smaller
  peak RAM on TinyLlama, at **+0.61 perplexity** (+12.4%); every INT4 greedy pick still
  lands inside the f32 model's top-5.
- **CUDA matmul vs cuBLAS:** the hand-written tiled GEMM reaches **70-85%** of cuBLAS
  fp32 throughput from 1024^3 to 4096^3 on the RTX 5060 Ti, and the single-token matvec
  (the kernel autoregressive decode actually uses) matches cuBLAS.

## Features

- **Models:** GPT-2 (learned positions, LayerNorm, GELU, tied LM head) and LLaMA
  (RMSNorm, RoPE with HuggingFace `rotate_half` pairing, SwiGLU, grouped-query attention,
  separate LM head).
- **Weight format:** a converter reads `config.json` + `model.safetensors`
  (f32/f16/bf16) and writes one `.tllm` file; the loader `mmap`s it and hands out
  zero-copy, 64-byte-aligned tensor views.
- **Quantization:** optional Q8 or Q4 for the 2D linear weights, in blocks of 32 with an
  fp32 scale per block. Quantized matvecs quantize the activation to int8 and use int8
  dot products (NEON `sdot` where the CPU has it, scalar elsewhere).
- **CUDA kernels:** tiled shared-memory GEMM with register blocking, a warp-per-row
  matvec for the single-query case, fused single-query GQA attention, fused LayerNorm,
  and a stable softmax, with a benchmark against the CPU ops and cuBLAS.
- **Tokenizers:** a hand-written GPT-2 byte-level BPE, and SentencePiece for LLaMA.
- **Sampling:** greedy, temperature, top-k, top-p, with a seedable RNG.
- **Memory:** each model allocates all activations and its KV cache up front from one
  arena, so the forward pass does not allocate per token.

## Build

Needs CMake 3.20+, a C++17 compiler, and SentencePiece. The loader uses POSIX `mmap`, so
Windows is not supported. Verified on Linux (GCC 16) and Apple Silicon (Apple Clang).

```sh
# Linux (Arch): sudo pacman -S sentencepiece cmake ninja
# Linux (Debian/Ubuntu): sudo apt install libsentencepiece-dev libabsl-dev cmake ninja-build
# macOS: brew install sentencepiece          # pulls in abseil

cmake -S . -B build -G Ninja && cmake --build build
```

This builds `tinyllm-convert`, `tinyllm-cli`, the test suite, and `tinyllm-bench-cpu`.
The default build type is Release (`-O3 -march=native`), so binaries are tuned for the
building machine.

### CUDA kernels (optional)

On a Linux machine with `nvcc` and cuBLAS (Blackwell cards such as the RTX 50 series need
CUDA 12.8+; this was verified on CUDA 13.5):

```sh
cmake -S . -B build-cuda -G Ninja -DTINYLLM_CUDA=ON && cmake --build build-cuda
./build-cuda/cuda/tinyllm-bench-cuda       # kernels vs CPU and cuBLAS
ctest --test-dir build-cuda                # kernel-vs-CPU correctness
```

The kernels compile unchanged on sm_120; `CMAKE_CUDA_ARCHITECTURES` defaults to `native`.
See [`cuda/README.md`](cuda/README.md).

## Get a model, convert, run

Models go in `models/` (gitignored).

```sh
hf download openai-community/gpt2 config.json model.safetensors vocab.json merges.txt \
  --local-dir models/gpt2
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json model.safetensors tokenizer.model \
  --local-dir models/tinyllama

./build/tinyllm-convert models/gpt2 gpt2.tllm
./build/tinyllm-convert models/tinyllama tinyllama-q4.tllm --quant q4   # or --quant q8

./build/tinyllm-cli -m gpt2.tllm -p "The capital of France is" -n 64 --temp 0
./build/tinyllm-cli -m tinyllama-q4.tllm --chat -p "What is a KV cache?" -n 128
```

The converter reads a single `model.safetensors` (sharded checkpoints are not supported).
With `--quant`, only 2D linear weights are quantized; embeddings, norms, and biases stay
f32. CLI flags: `-t/--tokenizer`, `-n/--n-predict`, `--temp`, `--top-k`, `--top-p`,
`--seed`, `--ctx`, `--chat`; run `tinyllm-cli -h` for the full list. `TINYLLM_THREADS`
sets the f32 pool's thread count.

## Correctness

Correctness is the point of this project, so it is proven against an independent
reference rather than asserted. `tests/reference_pytorch_test.cpp` compares the engine to
real PyTorch/HuggingFace outputs element-wise; `scripts/gen_reference.py` generates the
reference dumps (in an isolated `uv` venv, since the paths differ per machine). See
[`scripts/README.md`](scripts/README.md) to reproduce.

| Check | Reference | Worst-case error |
| --- | --- | --- |
| GPT-2, every layer (ln/qkv/attn/mlp/residual, logits) | HF GPT-2, eager attention | 9e-5 |
| TinyLlama-1.1B, every layer + shipped `Model::forward` logits | HF LlamaForCausalLM | 3e-5 |
| LayerNorm, GELU, softmax, MHA/GQA attention | torch / transformers | 1e-7 to 5e-6 |
| RMSNorm, RoPE (`rotate_half`), SwiGLU | transformers | 0 to 1e-4 |

**Quantized paths.** An INT8/INT4 forward is W8A8/W4A8 (the matvec quantizes activations
to int8 as well as weights), so it cannot match f32 element-wise; quantization is lossy
by design. The quantized tests instead check *prediction fidelity* against the f32 HF
logits: the greedy token must be preserved (Q8 exactly, every precision within the f32
top-5) at each position, and the raw logit divergence is reported as the measured
quantization cost. Q8 matches the f32 argmax everywhere; Q4 diverges only at near-tied or
high-entropy positions, with the pick still in the f32 top-5.

## Benchmarks

**CUDA matmul vs cuBLAS** (RTX 5060 Ti, fp32, `bench/bench_cuda.cpp`). Square GEMM,
M=N=K, hand-written tiled kernel as a fraction of cuBLAS GFLOP/s:

| N | tiled GFLOP/s | cuBLAS GFLOP/s | tiled / cuBLAS | max abs err vs cuBLAS |
| --- | --- | --- | --- | --- |
| 1024 | ~8100 | ~9800 | 83% | 8e-5 |
| 2048 | ~9100 | ~12300 | 75% | 2e-4 |
| 4096 | ~8900 | ~12700 | 70% | 3e-4 |

The single-token matvec (M=1), the kernel used during decode, matches or slightly exceeds
cuBLAS on the GPT-2 and LLaMA projection shapes. Fused LayerNorm, softmax, and attention
match the CPU reference to ~1e-7. The CUDA kernels are benchmarked, not yet wired into
`Model::forward`, so end-to-end generation is CPU-only.

**Quantization tradeoff** (`bench/bench_cpu.cpp`; perplexity on a fixed 255-token
public-domain passage, from the engine's own logits; memory is peak RSS during
generation, deterministic):

| Model | Disk | Peak RAM | Perplexity | vs f32 |
| --- | --- | --- | --- | --- |
| GPT-2 f32 | 475 MB | 504 MB | 22.63 | - |
| GPT-2 Q8 | 242 MB | 271 MB | 22.67 | +0.04 |
| GPT-2 Q4 | 201 MB | 230 MB | 24.61 | +1.98 |
| TinyLlama f32 | 4196 MB | 3984 MB | 4.94 | - |
| TinyLlama Q8 | 1360 MB | 1144 MB | 4.97 | +0.03 |
| TinyLlama Q4 | 867 MB | 653 MB | 5.55 | +0.61 |

INT8 buys most of the memory win for a negligible accuracy cost; INT4 roughly halves
memory again for a real, measurable perplexity increase. This is the "memory is the
limit" story in numbers.

**Throughput** is CPU-only and single-token (prefill and decode cost the same per token,
since `Model::forward` processes one token at a time). On the host CPU (16 threads),
TinyLlama-1.1B runs on the order of 15 to 40 tok/s depending on precision and GPT-2 small
well above 100 tok/s. Run `./build/tinyllm-bench-cpu` on an idle machine for clean
figures; the committed numbers above (memory, perplexity, correctness) are deterministic
and do not depend on machine load.

## Architecture

![tinyllm architecture: convert a checkpoint once, then run the forward pass one token at a time through a pre-allocated KV cache](docs/architecture.png)

- `tinyllm-convert` (`tools/convert.cpp`) runs once per model: maps HF names to canonical
  ones, converts f16/bf16 to f32, transposes GPT-2 Conv1D weights so every linear weight
  is `[out, in]`, and optionally quantizes.
- `load_model()` (`src/model.cpp`) `mmap`s the `.tllm` and builds a GPT-2 (`src/gpt2.cpp`)
  or LLaMA (`src/llama.cpp`) model from `Config::arch`.
- `Model::forward(token, pos)` runs one token, writes its K/V to the pre-allocated cache,
  and returns logits. `linear()` (`include/tinyllm/ops.h`) dispatches on dtype: f32 in
  `src/ops_cpu.cpp`, Q8/Q4 in `src/quant.cpp`.

The headers in [`include/tinyllm/`](include/tinyllm/) are the contracts between these
pieces; [`ARCHITECTURE.md`](ARCHITECTURE.md) documents them in depth.

## Testing

```sh
ctest --test-dir build --output-on-failure
```

129 GoogleTests cover the ops, quantization layout and kernels, the sampler, the loader,
the tokenizers, and small random GPT-2/LLaMA models against naive double-precision
references, plus the PyTorch/HuggingFace element-wise and quantization-fidelity tests
above. The HuggingFace tests `GTEST_SKIP` cleanly when `models/` is absent (as in CI).

## Limitations

- CPU for end-to-end generation; the CUDA kernels are benchmarked but not yet integrated
  into the forward pass.
- One token per `forward` call, prefill included, so long prompts are processed serially.
- Single-file `safetensors` only; GPT-2 and LLaMA architectures only.
- No comparison against llama.cpp or a production runtime yet.

## License

MIT. See [LICENSE](LICENSE).
