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

    models/ref/gpt2/manifest.txt       "<name> <dim0> [dim1 ...]" per line, '#' comments
    models/ref/gpt2/<name>.f32         flat row-major float32, little-endian, no header
    models/ref/tinyllama/manifest.txt
    models/ref/tinyllama/<name>.f32
    models/ref/ops/manifest.txt
    models/ref/ops/<name>.f32

Four independent things are dumped:

1. gen_gpt2_reference() / gen_tinyllama_reference(): runs real HF weights (models/gpt2,
   models/tinyllama -- the same HF dirs tinyllm-convert reads) through HF's GPT2Model /
   LlamaForCausalLM on a fixed prompt and captures the post-op intermediate at every point
   tinyllm's forward pass also computes one. See _capture_and_dump_gpt2 /
   _capture_and_dump_tinyllama below for the exact list; it is unchanged from the original
   version of this script. This is the existing f32 reference; nothing about it changed.

2. gen_op_level_reference(): fixed-seed random inputs run through HF/torch's *own*
   reference implementations of each named op, independent of the GPT-2 forward above.
   Unchanged from the original version of this script; see the function for the full list.
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
TINYLLAMA_DIR = MODELS / "tinyllama"
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


# --------------------------------------------------------------------------- .tllm reader
# --------------------------------------------------------------------------- GPT-2 (real weights)

PROMPT_IDS = [464, 3139, 286, 4881, 318]  # "The capital of France is"


def _load_gpt2_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained(str(GPT2_DIR))
    ids = tok.encode("The capital of France is")
    assert ids == PROMPT_IDS, f"tokenizer mismatch: {ids} != {PROMPT_IDS}"

    model = GPT2LMHeadModel.from_pretrained(str(GPT2_DIR), attn_implementation="eager")
    model.eval().to(torch.float32)
    return model


def _capture_and_dump_gpt2(model, out_dir: Path, header: str):
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
        p = f"l{i}."
        handles.append(block.ln_1.register_forward_hook(save(p + "ln1")))
        handles.append(block.attn.c_attn.register_forward_hook(save(p + "qkv")))
        # c_proj's pre-hook captures its input: the merged attention output, pre-projection.
        handles.append(block.attn.c_proj.register_forward_pre_hook(save_input(p + "attn_merged")))
        handles.append(block.attn.c_proj.register_forward_hook(save(p + "attn_proj")))
        handles.append(block.ln_2.register_forward_hook(save(p + "ln2")))
        handles.append(block.mlp.c_fc.register_forward_hook(save(p + "mlp_fc")))
        handles.append(block.mlp.act.register_forward_hook(save(p + "mlp_gelu")))
        handles.append(block.mlp.c_proj.register_forward_hook(save(p + "mlp_proj")))
        # Raw block output (pre-ln_f), independent of however output_hidden_states
        # chooses to represent the last layer (see note below): block.forward returns
        # (hidden_states, ...); ln_f is applied by GPT2Model only *after* this loop, so
        # this hook is always the un-normed residual stream, including for the last block.
        handles.append(block.register_forward_hook(save(p + "block_out")))

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

    d_out = Dumper(out_dir)
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
        f"attn_implementation=eager, float32, T={T} d={d} n_layers={n_layers}. {header}"
    )


def gen_gpt2_reference():
    if not (GPT2_DIR / "model.safetensors").exists():
        print(f"skip: {GPT2_DIR} has no model.safetensors", file=sys.stderr)
        return
    model = _load_gpt2_model()
    _capture_and_dump_gpt2(model, REF_DIR / "gpt2", "f32 reference.")


# --------------------------------------------------------------------------- TinyLlama (real weights)

# "The capital of France is", SentencePiece ids with no BOS (never added by tinyllm's
# Tokenizer::encode -- see tinyllm/tokenizer.h and CLAUDE.md). The *authoritative* check
# that this list is right is in tests/reference_pytorch_test.cpp: it calls the engine's
# real SentencePiece encode() on this same string and ASSERT_EQs it against this exact
# list before trusting anything downstream. The assert below is a secondary cross-check
# using HF's tokenizer for this script's own sanity, not the source of truth -- if the two
# ever disagree, trust the C++ assertion and repin this list (and TINYLLAMA_FULL_IDS) to
# whatever the engine's SentencePiece actually produces, never the other way around.
TINYLLAMA_BOS_ID = 1  # config.json bos_token_id, matches Config::bos_id in the .tllm header
TINYLLAMA_PROMPT_IDS = [450, 7483, 310, 3444, 338]
TINYLLAMA_FULL_IDS = [TINYLLAMA_BOS_ID] + TINYLLAMA_PROMPT_IDS  # what both HF and the engine run


def _load_tinyllama_model():
    from transformers import AutoTokenizer, LlamaForCausalLM

    # Secondary cross-check only (see module-level comment on TINYLLAMA_PROMPT_IDS above).
    # This venv has no `sentencepiece` pip package, so AutoTokenizer falls back to the
    # fast (tokenizers-crate) backend built from tokenizer.json rather than true
    # SentencePiece; that backend is a lossless conversion of the same tokenizer.model
    # for plain ASCII text like this prompt, but it is not byte-for-byte the same code
    # path the engine runs. The engine's own SentencePiece encode() in the C++ test is
    # what actually gates the comparison.
    tok = AutoTokenizer.from_pretrained(str(TINYLLAMA_DIR))
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    print(f"tinyllama HF tokenizer ids (no BOS, cross-check only): {ids}")
    assert ids == TINYLLAMA_PROMPT_IDS, f"tokenizer mismatch: {ids} != {TINYLLAMA_PROMPT_IDS}"
    print(f"tinyllama full id sequence fed to HF (BOS + prompt): {TINYLLAMA_FULL_IDS}")

    model = LlamaForCausalLM.from_pretrained(str(TINYLLAMA_DIR), attn_implementation="eager")
    model.eval().to(torch.float32)
    return model


def _capture_and_dump_tinyllama(model, out_dir: Path, header: str):
    lm = model.model  # LlamaModel
    n_layers = len(lm.layers)
    d = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or d // n_heads
    d_ff = model.config.intermediate_size

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
    for i, layer in enumerate(lm.layers):
        p = f"l{i}."
        handles.append(layer.input_layernorm.register_forward_hook(save(p + "ln1")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(save(p + "q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(save(p + "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(save(p + "v")))
        # Pre-hook on o_proj captures its input = RoPE'd GQA attention output, heads merged
        # back to [T, n_heads*head_dim], before the output projection. The one point where
        # RoPE and GQA attention are exercised together (not isolated); both are also checked
        # standalone with synthetic data by OpLevelReference, so this catches their wiring.
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(save_input(p + "attn_merged")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(save(p + "attn_proj")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(save(p + "ln2")))
        handles.append(layer.mlp.gate_proj.register_forward_hook(save(p + "gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(save(p + "up")))
        handles.append(layer.mlp.down_proj.register_forward_hook(save(p + "down")))
        # Raw decoder-layer output, independent of however output_hidden_states chooses
        # to represent the last layer -- same cross-check pattern as gen_gpt2_reference.
        handles.append(layer.register_forward_hook(save(p + "block_out")))

    input_ids = torch.tensor([TINYLLAMA_FULL_IDS], dtype=torch.long)
    out = model(input_ids, output_hidden_states=True)
    logits = out.logits[0]  # [T, vocab]
    hidden_states = out.hidden_states  # tuple(len n_layers+1); [0] is the embedding, [-1] normed.

    for h in handles:
        h.remove()

    d_out = Dumper(out_dir)
    T = len(TINYLLAMA_FULL_IDS)
    embed = lm.embed_tokens(input_ids)[0]  # [T, d], no positional embedding table for LLaMA
    d_out.add("embed", embed)
    err = (embed - hidden_states[0][0]).abs().max().item()
    assert err < 1e-6, f"embed vs hidden_states[0] mismatch, max diff {err}"

    block_in = embed
    for i in range(n_layers):
        p = f"l{i}."
        d_out.add(p + "ln1", captured[p + "ln1"][0])
        d_out.add(p + "q", captured[p + "q"][0])
        d_out.add(p + "k", captured[p + "k"][0])
        d_out.add(p + "v", captured[p + "v"][0])
        d_out.add(p + "attn_merged", captured[p + "attn_merged"][0])
        # resid1 = input to this block + attn contribution (pre-residual o_proj output)
        resid1 = block_in + captured[p + "attn_proj"][0]
        d_out.add(p + "resid1", resid1)
        d_out.add(p + "ln2", captured[p + "ln2"][0])
        d_out.add(p + "gate", captured[p + "gate"][0])
        d_out.add(p + "up", captured[p + "up"][0])
        # SwiGLU gate: silu(gate_proj) * up_proj. LlamaMLP.forward computes this inline
        # (no standalone module to hook), so it's recomputed here the same way resid1/
        # resid2 are -- from captured sub-module outputs plus a known closed-form op.
        swiglu = F.silu(captured[p + "gate"][0]) * captured[p + "up"][0]
        d_out.add(p + "swiglu", swiglu)
        d_out.add(p + "down", captured[p + "down"][0])
        resid2 = resid1 + captured[p + "down"][0]
        d_out.add(p + "resid2", resid2)
        want = captured[p + "block_out"][0]
        err = (resid2 - want).abs().max().item()
        assert err < 1e-4, f"layer {i}: resid2 vs block hook output mismatch, max diff {err}"
        block_in = resid2
    norm_f_out = lm.norm(block_in)
    d_out.add("norm_f", norm_f_out)
    err = (norm_f_out - hidden_states[-1][0]).abs().max().item()
    assert err < 1e-4, f"norm_f vs hidden_states[-1] mismatch, max diff {err}"
    d_out.add("logits", logits)
    d_out.write_manifest(
        f"TinyLlama-1.1B-Chat-v1.0, HF weights at {TINYLLAMA_DIR}, full id sequence (BOS + prompt) "
        f"{TINYLLAMA_FULL_IDS}, attn_implementation=eager, float32, T={T} d={d} n_layers={n_layers} "
        f"n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim} d_ff={d_ff}. "
        f"q/k are PRE-RoPE (raw q_proj/k_proj output); attn_merged is POST-RoPE+GQA-attention, "
        f"pre-o_proj, so RoPE and attention are exercised together there, not isolated (see "
        f"OpLevelReference / ops_test.cpp for isolated RoPE and attention checks). {header}"
    )


def gen_tinyllama_reference():
    if not (TINYLLAMA_DIR / "model.safetensors").exists():
        print(f"skip: {TINYLLAMA_DIR} has no model.safetensors", file=sys.stderr)
        return
    model = _load_tinyllama_model()
    _capture_and_dump_tinyllama(model, REF_DIR / "tinyllama", "f32 reference.")


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
    gen_tinyllama_reference()
    gen_op_level_reference()


if __name__ == "__main__":
    main()
