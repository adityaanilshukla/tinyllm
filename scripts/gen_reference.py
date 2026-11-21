#!/usr/bin/env python3
"""Generate PyTorch/HuggingFace reference dumps for tinyllm's element-wise verification
test (tests/reference_pytorch_test.cpp).

Run with the uv-managed venv at models/.venv-ref (Python 3.12, torch==2.15.0+cpu,
transformers==5.24.0, numpy, safetensors). It lives under models/ -- already gitignored
wholesale -- rather than at the repo root, so there is nothing new to ignore:

    uv venv --python 3.12 models/.venv-ref
    uv pip install --python models/.venv-ref torch --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip install --python models/.venv-ref transformers numpy safetensors huggingface_hub
    models/.venv-ref/bin/python scripts/gen_reference.py

Everything here runs in float32 (torch's default for these models once loaded without
`torch_dtype=`), matching tinyllm's own dtype, so the comparison in the C++ test is
apples-to-apples rather than float32-vs-float64.

Output layout (gitignored: lives under models/, and models/ is already gitignored):

    models/ref/gpt2/manifest.txt   "<name> <dim0> [dim1 ...]" per line, '#' comments
    models/ref/gpt2/<name>.f32     flat row-major float32, little-endian, no header
    models/ref/ops/manifest.txt
    models/ref/ops/<name>.f32

Two independent things are dumped:

1. gen_gpt2_reference(): runs real openai-community/gpt2 weights (models/gpt2, the same
   HF dir tinyllm-convert reads) through HF's GPT2Model on a fixed 5-token prompt and
   captures the post-op intermediate at every point tinyllm's forward pass also computes
   one: token+pos embedding, each block's ln_1/qkv/merged-attention-output/post-attention
   residual/ln_2/mlp-fc/mlp-gelu/post-mlp residual, the final ln_f, and the logits. This
   is captured via forward hooks on the real submodules (GPT2Block.ln_1, GPT2Attention.
   c_attn, GPT2Attention.c_proj (pre-hook for its input = merged heads, post-hook for its
   output = pre-residual attention contribution), GPT2MLP.c_fc, GPT2MLP.act, GPT2MLP.
   c_proj), run with attn_implementation="eager" so there is a plain softmax(QK^T/sqrt(d))V
   under the hood rather than a fused SDPA kernel.

   Fixed prompt: "The capital of France is" -> token ids [464, 3139, 286, 4881, 318],
   the same prompt tests/gpt2_e2e_test.cpp pins tokens against. Running the whole 5-token
   sequence through HF in one batched causal forward is mathematically identical, position
   by position, to tinyllm's one-token-at-a-time KV-cache forward: causal attention only
   lets position p see positions <= p either way.

2. gen_op_level_reference(): fixed-seed random inputs run through HF/torch's *own*
   reference implementations of each named op, independent of the GPT-2 forward above:
     - LayerNorm:    torch.nn.functional.layer_norm
     - GELU:         transformers.activations.NewGELUActivation (GPT-2's "gelu_new")
     - Softmax:      torch.nn.functional.softmax
     - Attention:    manual torch matmul/softmax/matmul, single query attending over a
                      causal KV cache of n_pos positions (MHA and GQA cases), mirroring
                      tinyllm's attention() contract (one query position, n_pos keys).
     - RMSNorm:      transformers.models.llama.modeling_llama.LlamaRMSNorm
     - RoPE:         transformers.models.llama.modeling_llama.apply_rotary_pos_emb +
                      the library's own default inv_freq formula, HF rotate_half pairing.
     - SwiGLU gate:  torch.nn.functional.silu(gate) * up

   All seeds are fixed (torch.manual_seed per tensor, listed inline) so this script is
   fully reproducible.
"""
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
GPT2_DIR = MODELS / "gpt2"
REF_DIR = MODELS / "ref"

torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- dump helpers

class Dumper:
    """Collects (name -> tensor) and writes manifest.txt + "<name>.f32" flat binaries."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.entries = []  # (name, shape)

    def add(self, name: str, t: torch.Tensor):
        t = t.detach().to(torch.float32).contiguous().cpu()
        flat = t.reshape(-1).numpy()
        path = self.out_dir / f"{name}.f32"
        with open(path, "wb") as f:
            f.write(struct.pack(f"<{flat.size}f", *flat.tolist()))
        self.entries.append((name, list(t.shape)))

    def write_manifest(self, header_comment: str):
        path = self.out_dir / "manifest.txt"
        with open(path, "w") as f:
            f.write(f"# {header_comment}\n")
            f.write("# name dim0 [dim1 ...]\n")
            for name, shape in self.entries:
                f.write(name + " " + " ".join(str(d) for d in shape) + "\n")
        print(f"wrote {len(self.entries)} tensors -> {path}")


# --------------------------------------------------------------------------- GPT-2 (real weights)

PROMPT_IDS = [464, 3139, 286, 4881, 318]  # "The capital of France is"


def gen_gpt2_reference():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    if not (GPT2_DIR / "model.safetensors").exists():
        print(f"skip: {GPT2_DIR} has no model.safetensors", file=sys.stderr)
        return

    tok = GPT2Tokenizer.from_pretrained(str(GPT2_DIR))
    ids = tok.encode("The capital of France is")
    assert ids == PROMPT_IDS, f"tokenizer mismatch: {ids} != {PROMPT_IDS}"

    model = GPT2LMHeadModel.from_pretrained(str(GPT2_DIR), attn_implementation="eager")
    model.eval().to(torch.float32)
    transformer = model.transformer
    n_layers = len(transformer.h)
    d = model.config.n_embd

    captured = {}

    def save(name):
        def hook(_module, _inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out

        return hook

    def save_input(name):
        def hook(_module, inp):
            captured[name] = inp[0]

        return hook

    handles = []
    for i, block in enumerate(transformer.h):
        handles.append(block.ln_1.register_forward_hook(save(f"l{i}.ln1")))
        handles.append(block.attn.c_attn.register_forward_hook(save(f"l{i}.qkv")))
        handles.append(block.attn.c_proj.register_forward_pre_hook(save_input(f"l{i}.attn_merged")))
        handles.append(block.attn.c_proj.register_forward_hook(save(f"l{i}.attn_proj")))
        handles.append(block.ln_2.register_forward_hook(save(f"l{i}.ln2")))
        handles.append(block.mlp.c_fc.register_forward_hook(save(f"l{i}.mlp_fc")))
        handles.append(block.mlp.act.register_forward_hook(save(f"l{i}.mlp_gelu")))
        handles.append(block.mlp.c_proj.register_forward_hook(save(f"l{i}.mlp_proj")))
        # Raw block output (pre-ln_f), independent of however output_hidden_states
        # chooses to represent the last layer (see note below): block.forward returns
        # (hidden_states, ...); ln_f is applied by GPT2Model only *after* this loop, so
        # this hook is always the un-normed residual stream, including for the last block.
        handles.append(block.register_forward_hook(save(f"l{i}.block_out")))

    input_ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    out = model(input_ids, output_hidden_states=True)
    logits = out.logits[0]  # [T, vocab]
    hidden_states = out.hidden_states  # tuple(len n_layers+1); [0] is the embedding.
    # NOTE: in this transformers version hidden_states[-1] is the *normed* final state
    # (equal to transformer.ln_f(raw last block output)), not the raw block output like
    # every other entry. We don't rely on that quirk: block inputs/outputs below come
    # from the block hooks and a manually-recomputed embedding, not from this tuple.

    for h in handles:
        h.remove()

    d_out = Dumper(REF_DIR / "gpt2")
    T = len(PROMPT_IDS)
    position_ids = torch.arange(T).unsqueeze(0)
    embed = transformer.wte(input_ids) + transformer.wpe(position_ids)  # transformer.drop is identity in eval()
    d_out.add("embed", embed[0])
    block_in = embed[0]
    for i in range(n_layers):
        d_out.add(f"l{i}.ln1", captured[f"l{i}.ln1"][0])
        d_out.add(f"l{i}.qkv", captured[f"l{i}.qkv"][0])
        d_out.add(f"l{i}.attn_merged", captured[f"l{i}.attn_merged"][0])
        # resid1 = input to this block + attn contribution (pre-residual c_proj output)
        resid1 = block_in + captured[f"l{i}.attn_proj"][0]
        d_out.add(f"l{i}.resid1", resid1)
        d_out.add(f"l{i}.ln2", captured[f"l{i}.ln2"][0])
        d_out.add(f"l{i}.mlp_fc", captured[f"l{i}.mlp_fc"][0])
        d_out.add(f"l{i}.mlp_gelu", captured[f"l{i}.mlp_gelu"][0])
        resid2 = resid1 + captured[f"l{i}.mlp_proj"][0]
        d_out.add(f"l{i}.resid2", resid2)
        # resid2 must equal the block's own raw forward-hook output; check here, not just
        # in C++, so a hook-wiring mistake in this script fails loudly at generation time.
        want = captured[f"l{i}.block_out"][0]
        err = (resid2 - want).abs().max().item()
        assert err < 1e-4, f"layer {i}: resid2 vs block hook output mismatch, max diff {err}"
        block_in = resid2
    ln_f_out = transformer.ln_f(block_in)
    d_out.add("ln_f", ln_f_out)
    # Cross-check against HF's own last_hidden_state (== hidden_states[-1] here).
    err = (ln_f_out - hidden_states[-1][0]).abs().max().item()
    assert err < 1e-4, f"ln_f vs hidden_states[-1] mismatch, max diff {err}"
    d_out.add("logits", logits)
    d_out.write_manifest(
        f"GPT-2 small, HF weights at {GPT2_DIR}, prompt tokens {PROMPT_IDS}, "
        f"attn_implementation=eager, float32, T={T} d={d} n_layers={n_layers}"
    )


# --------------------------------------------------------------------------- op-level references

def ref_attention(q, k, v, n_pos, H, KVH, hd):
    """Single query attending causally over n_pos cached keys/values (tinyllm's
    attention() contract: one query position, n_pos positions in the KV cache)."""
    q = q.view(H, hd)
    k = k.view(n_pos, KVH, hd)
    v = v.view(n_pos, KVH, hd)
    group = H // KVH
    out = torch.zeros(H, hd, dtype=q.dtype)
    for h in range(H):
        kvh = h // group
        scores = (k[:, kvh, :] @ q[h]) / math.sqrt(hd)  # [n_pos]
        probs = F.softmax(scores, dim=-1)
        out[h] = probs @ v[:, kvh, :]
    return out.reshape(-1)


def gen_op_level_reference():
    from transformers.activations import NewGELUActivation
    from transformers.models.llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb

    d_out = Dumper(REF_DIR / "ops")

    # --- LayerNorm (GPT-2 d_model = 768), eps = 1e-5.
    torch.manual_seed(1001)
    D = 768
    x = torch.randn(D) * 2.0 + 0.3
    g = torch.rand(D) + 0.5
    b = torch.randn(D) * 0.3
    y = F.layer_norm(x, [D], weight=g, bias=b, eps=1e-5)
    d_out.add("layernorm.x", x); d_out.add("layernorm.g", g); d_out.add("layernorm.b", b)
    d_out.add("layernorm.y", y)

    # --- GELU ("gelu_new", GPT-2's d_ff = 3072), HF's own NewGELUActivation module.
    torch.manual_seed(1002)
    Dff = 3072
    xg = torch.randn(Dff) * 4.0
    yg = NewGELUActivation()(xg)
    d_out.add("gelu.x", xg); d_out.add("gelu.y", yg)

    # --- Softmax, two widths.
    for n, seed in [(50, 1003), (2000, 1004)]:
        torch.manual_seed(seed)
        xs = torch.randn(n) * 5.0
        ys = F.softmax(xs, dim=-1)
        d_out.add(f"softmax_{n}.x", xs); d_out.add(f"softmax_{n}.y", ys)

    # --- Attention: MHA (GPT-2 dims: H=KVH=12, hd=64, n_pos=5) and GQA (TinyLlama dims:
    # H=32, KVH=4, hd=64, n_pos=7).
    for tag, H, KVH, hd, n_pos, seed in [("attn_mha", 12, 12, 64, 5, 1005),
                                          ("attn_gqa", 32, 4, 64, 7, 1006)]:
        torch.manual_seed(seed)
        q = torch.randn(H * hd) * 1.5
        k = torch.randn(n_pos * KVH * hd) * 1.5
        v = torch.randn(n_pos * KVH * hd) * 1.5
        out = ref_attention(q, k, v, n_pos, H, KVH, hd)
        d_out.add(f"{tag}.q", q); d_out.add(f"{tag}.k", k); d_out.add(f"{tag}.v", v)
        d_out.add(f"{tag}.out", out)

    # --- RMSNorm (TinyLlama d_model = 2048), eps = 1e-5, HF's own LlamaRMSNorm module.
    torch.manual_seed(1007)
    Dl = 2048
    xr = torch.randn(Dl) * 2.0
    gr = torch.rand(Dl) + 0.5
    norm = LlamaRMSNorm(Dl, eps=1e-5)
    norm.weight.data.copy_(gr)
    yr = norm(xr)
    d_out.add("rmsnorm.x", xr); d_out.add("rmsnorm.g", gr); d_out.add("rmsnorm.y", yr)

    # --- RoPE (TinyLlama dims: H=32, hd=64, theta=10000), HF's rotate_half pairing via
    # apply_rotary_pos_emb, with inv_freq computed the same way LlamaRotaryEmbedding does.
    H, hd, theta = 32, 64, 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    for pos in (5, 2047):
        torch.manual_seed(1100 + pos)
        xq = torch.randn(H * hd) * 1.5
        q = xq.view(1, H, 1, hd)  # [batch, heads, seq=1, head_dim]
        freqs = torch.full((1, 1), float(pos)) * inv_freq  # [1, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [1, hd]
        cos, sin = emb.cos(), emb.sin()
        q_embed, _ = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=1)
        d_out.add(f"rope_p{pos}.x", xq)
        d_out.add(f"rope_p{pos}.y", q_embed.reshape(-1))

    # --- SwiGLU gate (TinyLlama d_ff = 5632): silu(gate) * up.
    torch.manual_seed(1008)
    Dff2 = 5632
    gate = torch.randn(Dff2) * 5.0
    up = torch.randn(Dff2) * 3.0
    y_silu = F.silu(gate) * up
    d_out.add("siluMul.gate", gate); d_out.add("siluMul.up", up); d_out.add("siluMul.y", y_silu)

    d_out.write_manifest(
        "op-level references: fixed-seed random inputs run through HF/torch's own "
        "layer_norm, NewGELUActivation (gelu_new), softmax, manual single-query causal "
        "attention, LlamaRMSNorm, apply_rotary_pos_emb (rotate_half), and silu, all float32."
    )


def main():
    print(f"torch {torch.__version__}")
    import transformers
    print(f"transformers {transformers.__version__}")
    gen_gpt2_reference()
    gen_op_level_reference()


if __name__ == "__main__":
    main()
