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
cmake -S . -B build -G Ninja && cmake --build build
./build/tinyllm-convert models/gpt2 models/gpt2.tllm
```

(TinyLlama is downloaded the same way, but `gen_reference.py` only exercises GPT-2's
real weights end-to-end; the LLaMA-specific ops (rmsnorm, rope, silu_mul) are checked
at the op level with synthetic inputs, not against TinyLlama's real weights -- see the
"not covered" note in the test file header.)

## Generate the reference dumps

```sh
models/.venv-ref/bin/python scripts/gen_reference.py
```

Writes `models/ref/gpt2/` and `models/ref/ops/` (manifest.txt + flat float32 `.f32`
binaries per tensor). Both are gitignored, since they live under `models/`.

## Run the test

```sh
ctest --test-dir build -R "Gpt2Reference|OpLevelReference" --output-on-failure
# or, to see the worst-case-error summary printed even on success:
./build/tests/tinyllm-tests --gtest_filter="Gpt2Reference.*:OpLevelReference.*"
```

Both tests `GTEST_SKIP` cleanly if `models/gpt2.tllm` or the `models/ref/*` dumps are
missing.
