# scripts/

`gen_reference.py` generates PyTorch/HuggingFace reference dumps consumed by
`tests/reference_pytorch_test.cpp`. It is the only script in this directory.

## Setup (one-time, Python 3.14 is the system default and has no stable PyTorch
wheels, so this uses an isolated `uv`-managed venv at `models/.venv-ref` with Python
3.12 -- placed under `models/` since that whole directory is already gitignored)

```sh
uv venv --python 3.12 models/.venv-ref
uv pip install --python models/.venv-ref torch --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python models/.venv-ref transformers numpy safetensors huggingface_hub
```

Verified working versions: Python 3.12.13, torch 2.15.0+cpu, transformers 5.24.0,
numpy 2.5.4.

## Models

```sh
hf download openai-community/gpt2 --local-dir models/gpt2
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama
cmake -S . -B build -G Ninja && cmake --build build
./build/tinyllm-convert models/gpt2 models/gpt2.tllm
./build/tinyllm-convert models/tinyllama models/tinyllama-f32.tllm   # no --quant: f32, for element-wise comparison
```

`gen_reference.py` exercises both GPT-2's and TinyLlama's real weights end-to-end now
(see "What gets generated" below). The op-level references (rmsnorm, rope, silu_mul,
attention MHA/GQA, layernorm, gelu, softmax) are still checked separately with
fixed-seed synthetic inputs at each architecture's real dimensions, not against real
weights -- that part has not changed.

## What gets generated

```sh
models/.venv-ref/bin/python scripts/gen_reference.py
```

Writes three gitignored directories (they live under `models/`, already gitignored
wholesale), each `manifest.txt` ("<name> <dim0> [dim1 ...]" per line) plus one flat
little-endian float32 `<name>.f32` per tensor:

- `models/ref/gpt2/` -- GPT-2 small, real `models/gpt2` weights, fixed 5-token prompt
  "The capital of France is" -> ids `[464, 3139, 286, 4881, 318]`. Dumps the embedding,
  every block's ln1/qkv/attn_merged/resid1/ln2/mlp_fc/mlp_gelu/resid2, `ln_f`, and
  `logits`.
- `models/ref/tinyllama/` -- TinyLlama-1.1B-Chat-v1.0, real `models/tinyllama` weights,
  fixed 6-token id sequence (BOS + 5-token prompt) for the same prompt text:
  SentencePiece encodes "The capital of France is" (no BOS -- `Tokenizer::encode` never
  adds it, see `tinyllm/tokenizer.h` and `CLAUDE.md`) to `[450, 7483, 310, 3444, 338]`,
  and BOS (id 1) is prepended, giving `[1, 450, 7483, 310, 3444, 338]`. This id list is
  pinned in two places that must agree: `TINYLLAMA_PROMPT_IDS` in this script (checked
  against HF's tokenizer as a secondary sanity check only -- see the comment above it;
  this venv has no `sentencepiece` pip package, so that check runs through HF's fast
  tokenizers-crate backend, not true SentencePiece) and `kTinyLlamaPromptIds` in
  `tests/reference_pytorch_test.cpp`, which is the *authoritative* check: the C++ test
  calls the engine's real SentencePiece `encode()` on the literal prompt string at test
  run time and asserts the result matches before trusting anything downstream. If you
  ever see a mismatch, repin both lists to whatever the engine's SentencePiece actually
  produces, not the other way around. Dumps the embedding, every layer's
  ln1/q/k/v (q and k are PRE-RoPE)/attn_merged (POST-RoPE+GQA-attention, so those two ops
  are exercised together there, not in isolation)/resid1/ln2/gate/up/swiglu/down/resid2,
  the final `norm_f`, and `logits`.
- `models/ref/ops/` -- fixed-seed random inputs through HF/torch's own layernorm, gelu,
  softmax, attention (MHA and GQA, including TinyLlama's GQA dims), rmsnorm, rope (at
  pos 5 and pos 2047, to also exercise RoPE's position-dependent error), and silu_mul.

Regeneration takes a few seconds total (loading both HF models, in float32, and running
one short forward pass each); the TinyLlama HF load is the slower half of that (a ~2.2GB
bf16 safetensors file loaded and upcast to float32 in-process, so needs free RAM on the
order of ~5GB; 26GB was free when this was last run, no problem).

## Run the tests

```sh
ctest --test-dir build -R "Gpt2Reference|TinyLlamaReference|OpLevelReference" --output-on-failure
# or, to see the worst-case-error summary printed even on success:
./build/tests/tinyllm-tests --gtest_filter="Gpt2Reference.*:TinyLlamaReference.*:OpLevelReference.*"
```

All three tests `GTEST_SKIP` cleanly if their weight file(s) (`models/gpt2.tllm` /
`models/tinyllama-f32.tllm` / `models/tinyllama/tokenizer.model`) or the matching
`models/ref/*` dumps are missing. `TinyLlamaReference.ElementWiseMatchesHuggingFace`
also runs the actual shipped `load_model()` + `Model::forward()` path
(`src/llama.cpp`) against the same HF logits, in addition to the hand-rolled
reproduction through the public ops -- this is what closes the "TinyLlama's real
weights are not yet run end-to-end against HF" gap noted in `CLAUDE.md`.
