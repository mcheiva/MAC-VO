"""Aggregate a trtexec --exportProfile JSON by layer category.

    python Scripts/Parity/profile_agg.py runs/profile_b2.json [--top 25]

Categories are substring matches on the fused layer names TRT reports (ONNX node names
survive fusion as concatenations), so they are approximate but good enough to rank.
"""
import argparse
import json
import re
from collections import defaultdict

CATS = [  # (label, regex on lower-cased layer name); first match wins
    ("gridsample", r"gridsample"),
    ("twins_encoder", r"context_encoder|feat_encoder"),
    ("cost_patch_embed", r"patch_embed"),
    ("cost_perceiver", r"cost_perceiver|input_layer|encoder_layers|vertical_encoder"),
    ("corr_bmm", r"memory_encoder.*(matmul|bmm)|/corr"),
    ("gma_attention", r"\.att/|aggregator"),
    ("gru", r"gru|convz|convr|convq|fused_gru"),
    ("motion_encoder", r"update_block/encoder|convc|convf"),
    ("flow_cov_head", r"flow_head|cov_head"),
    ("cross_attn", r"decoder_layer|cross_attend"),
    ("flow_token_enc", r"flow_token_encoder"),
    ("upsample_mask", r"mask|upsample"),
    ("reformat", r"reformat|shuffle|transpose|reshape"),
    ("cast", r"cast"),
    ("concat", r"concat"),
    ("softmax", r"softmax"),
    ("layernorm", r"layernorm|normalization"),
    ("elementwise", r"add|mul|sub|div|sigmoid|tanh|relu|gelu|pwn|elementwise"),
    ("conv_other", r"conv"),
    ("matmul_other", r"matmul|gemm"),
]


def categorize(name: str) -> str:
    n = name.lower()
    for label, pat in CATS:
        if re.search(pat, n):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    rows = json.load(open(args.profile))
    layers = [r for r in rows if "name" in r and "averageMs" in r]
    total = sum(r["averageMs"] for r in layers)
    by = defaultdict(lambda: [0.0, 0])
    for r in layers:
        c = categorize(r["name"])
        by[c][0] += r["averageMs"]; by[c][1] += 1
    print(f"{len(layers)} layers, total {total:.2f} ms per inference\n")
    print(f"{'category':18s}{'ms':>9s}{'%':>7s}{'layers':>8s}")
    for c, (ms, n) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:18s}{ms:9.2f}{100 * ms / total:7.1f}{n:8d}")
    print(f"\ntop {args.top} layers:")
    for r in sorted(layers, key=lambda r: -r["averageMs"])[:args.top]:
        print(f"{r['averageMs']:8.3f} ms  {100 * r['averageMs'] / total:5.1f}%  {r['name'][:150]}")


if __name__ == "__main__":
    main()
