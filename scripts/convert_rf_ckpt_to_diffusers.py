"""Convert a RollingForcing (zhuhz22-style) raw ckpt to a diffusers
CausalWanTransformer3DModel directory for HF upload.

The raw ckpt (e.g. lyx/RollingForcing logs/rf_mix_from_rfdmd_8gpu/
checkpoint_model_002000/model.pt) carries three fp32 state dicts --
generator / critic / generator_ema -- under 'model._fsdp_wrapped_module.'
prefixes (FSDP).  We ship only generator_ema, stripped, remapped onto the
CausalWanTransformer3DModel key layout, cast to bf16, as a single
safetensors file.  The mapping below is a pure rename:  both checkpoints
have 825 keys with identical shapes (verified against the
Wan2.1-T2V-1.3B-Diffusers transformer index).

Run:
  .venv/bin/python scripts/convert_rf_ckpt_to_diffusers.py \
      --ckpt ckpts/lyx/RF-5step-1.3B/model.pt \
      --out  ckpts/WaveForcing/transformer
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

# diffusers submodule -> raw (post-prefix) submodule, per block.
_BLOCK_SUB = {
    "attn1.norm_q":       "self_attn.norm_q",
    "attn1.norm_k":       "self_attn.norm_k",
    "attn1.to_q":         "self_attn.q",
    "attn1.to_k":         "self_attn.k",
    "attn1.to_v":         "self_attn.v",
    "attn1.to_out.0":     "self_attn.o",
    "attn2.norm_q":       "cross_attn.norm_q",
    "attn2.norm_k":       "cross_attn.norm_k",
    "attn2.to_q":         "cross_attn.q",
    "attn2.to_k":         "cross_attn.k",
    "attn2.to_v":         "cross_attn.v",
    "attn2.to_out.0":     "cross_attn.o",
    "ffn.net.0.proj":     "ffn.0",
    "ffn.net.2":          "ffn.2",
    "norm2":              "norm3",
    "scale_shift_table":  "modulation",
}
# Top-level submodule -> raw submodule (bias/weight suffixes appended; the
# bare tensors map directly).
_TOP_SUB = {
    "patch_embedding":                            "patch_embedding",
    "condition_embedder.text_embedder.linear_1":  "text_embedding.0",
    "condition_embedder.text_embedder.linear_2":  "text_embedding.2",
    "condition_embedder.time_embedder.linear_1":  "time_embedding.0",
    "condition_embedder.time_embedder.linear_2":  "time_embedding.2",
    "condition_embedder.time_proj":               "time_projection.1",
    "proj_out":                                   "head.head",
    "scale_shift_table":                          "head.modulation",
}
_RAW_PREFIX = "model._fsdp_wrapped_module."


def to_diffusers_key(raw: str) -> str | None:
    """Map a raw generator_ema key to the diffusers key; None if unmapped."""
    assert raw.startswith(_RAW_PREFIX), raw
    sub = raw[len(_RAW_PREFIX):]
    parts = sub.split(".")
    if parts[0] == "blocks":
        # blocks.N.<rest>
        block = parts[1]
        rest = ".".join(parts[2:])
        for dst, src in _BLOCK_SUB.items():
            if rest == src or rest.startswith(src + "."):
                return f"blocks.{block}.{dst}{rest[len(src):]}"
        raise KeyError(f"unmapped block submodule: {sub}")
    for dst, src in _TOP_SUB.items():
        if sub == src or sub.startswith(src + "."):
            return f"{dst}{sub[len(src):]}"
    raise KeyError(f"unmapped top-level key: {sub}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="raw RollingForcing model.pt (generator_ema used)")
    ap.add_argument("--out", required=True,
                    help="output transformer/ dir (config.json must exist)")
    ap.add_argument("--reference", default=None,
                    help="diffusers safetensors index.json to cross-check "
                         "key sets (default: Wan2.1-T2V-1.3B-Diffusers)")
    args = ap.parse_args()

    ref_path = args.reference or os.path.join(
        "ckpts", "Wan2.1-T2V-1.3B-Diffusers", "transformer",
        "diffusion_pytorch_model.safetensors.index.json")
    with open(ref_path) as f:
        ref_keys = set(json.load(f)["weight_map"])

    print(f"[convert] loading {args.ckpt} (generator_ema, mmap) ...", flush=True)
    raw = torch.load(args.ckpt, map_location="cpu", mmap=True,
                     weights_only=False)["generator_ema"]

    out = {}
    missing, extra = set(ref_keys), set()
    for k, v in raw.items():
        dst = to_diffusers_key(k)
        if dst in ref_keys:
            missing.discard(dst)
        else:
            extra.add(dst)
        out[dst] = v.to(torch.bfloat16)

    if missing or extra:
        raise SystemExit(f"key mismatch: missing {sorted(missing)[:5]} ... "
                         f"extra {sorted(extra)[:5]} ...")
    print(f"[convert] {len(out)} keys remapped, bf16, "
          f"{sum(v.numel() for v in out.values()) / 1e9:.2f}B params", flush=True)

    os.makedirs(args.out, exist_ok=True)
    st_path = os.path.join(args.out, "diffusion_pytorch_model.safetensors")
    save_file(out, st_path)
    idx_path = os.path.join(args.out,
                            "diffusion_pytorch_model.safetensors.index.json")
    with open(idx_path, "w") as f:
        json.dump({
            "metadata": {"format": "pt"},
            "weight_map": {k: os.path.basename(st_path) for k in out},
        }, f, indent=2)
    size = os.path.getsize(st_path) / 1e9
    print(f"[convert] wrote {st_path} ({size:.2f} GB) + index.json", flush=True)


if __name__ == "__main__":
    main()
