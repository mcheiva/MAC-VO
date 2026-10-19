"""Self-checks for the exact (weight-preserving) rewrites. Run: python Scripts/Parity/test_exact.py"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT, build_model, list_frames, load_image_u8, run_model, u8_to_input  # noqa: E402



def stats(a: dict, b: dict) -> str:
    out = []
    for k in ("flow", "cov"):
        d = (a[k] - b[k]).abs()
        epe = (a[k] - b[k]).norm(dim=1) if k == "flow" else d
        out.append(f"{k}: mean {epe.mean():.2e} p99 {epe.flatten().quantile(0.99):.2e} max {epe.max():.2e}")
    return " | ".join(out)


def mean_epe(a, b):
    return (a["flow"] - b["flow"]).norm(dim=1).mean().item()


def test_fused_gru(dev: str):
    """FusedGRUPair.step == two SepConvGRU updates on the same input (fp32, random tensors)."""
    from Network.FlowFormer.core.gru import FusedGRUPair, SepConvGRU
    torch.manual_seed(0)
    ga, gb = SepConvGRU(128, 384).to(dev), SepConvGRU(128, 384).to(dev)
    fused = FusedGRUPair(ga, gb)
    B, H, W = 2, 88, 88
    inp, a, b = (torch.randn(B, 128, H, W, device=dev) for _ in range(3))
    ha, hb = torch.tanh(torch.randn(B, 128, H, W, device=dev)), torch.tanh(torch.randn(B, 128, H, W, device=dev))
    x = torch.cat([inp, a, b], 1)
    with torch.inference_mode():
        ra, rb = ga(ha, x), gb(hb, x)
        pre = fused.precompute(inp)
        fa, fb = fused.step(ha, hb, torch.cat([a, b], 1), pre)
    d = max((ra - fa).abs().max().item(), (rb - fb).abs().max().item())
    print(f"fused GRU vs original max|diff| = {d:.2e}")
    assert d < 1e-4, d


def test_cost_front(model, dev: str):
    """PatchEmbed.forward_cf + CrossAttentionLayer.forward_cf == original token-major path."""
    enc = model.memory_encoder.cost_perceiver_encoder
    torch.manual_seed(0)
    cost_maps = torch.randn(2 * 100, 1, 40, 40, device=dev) * 3     # pads 40 -> 40 (multiple of 8), 5x5 tokens
    with torch.inference_mode():
        x0, s0 = enc.patch_embed(cost_maps)
        ref = enc.input_layer(enc.latent_tokens, x0)
        x1, s1 = enc.patch_embed.forward_cf(cost_maps)
        new = enc.input_layer.forward_cf(enc.latent_tokens, x1)
    d_tok = (x0.transpose(1, 2) - x1).abs().max().item()
    d_out = (ref - new).abs().max().item()
    print(f"cost front: tokens max|diff| {d_tok:.2e}  cross-attn out max|diff| {d_out:.2e}  (|out| mean {ref.abs().mean():.3f})")
    assert tuple(s0) == tuple(s1) and d_tok < 1e-4 and d_out < 1e-3, (d_tok, d_out)


def test_patch_stack(dev: str):
    """pixel_unshuffle + conv3x3_cells_to_s2d x2 + 3x3 conv == proj (3 x conv6s2 with ReLUs)."""
    from Network.FlowFormer.core.encoder import PatchEmbed, conv3x3_cells_to_s2d, s2d_weight
    torch.manual_seed(0)
    pe = PatchEmbed(1, 64).to(dev)
    x = torch.randn(9, 1, 88, 88, device=dev) * 3
    with torch.inference_mode():
        ref = pe.proj(x)
        y = F.pixel_unshuffle(x, 2)
        y = F.relu(conv3x3_cells_to_s2d(y, s2d_weight(pe.proj[0]), pe.proj[0].bias))
        y = F.relu(conv3x3_cells_to_s2d(y, s2d_weight(pe.proj[2]), pe.proj[2].bias))
        y = F.conv2d(y, s2d_weight(pe.proj[4]), pe.proj[4].bias, padding=1)
    d = (ref - y).abs().max().item()
    print(f"patch-embed conv stack (stride-folded) max|diff| = {d:.2e}  (|ref| mean {ref.abs().mean():.3f})")
    assert ref.shape == y.shape and d < 1e-3, (ref.shape, y.shape, d)


def test_one_query_attn(dev: str):
    """attend_one_query == MultiHeadAttention for a single query per row."""
    from Network.FlowFormer.core.attention import MultiHeadAttention
    from Network.FlowFormer.core.decoder import attend_one_query
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 8)
    q, k, v = torch.randn(1000, 1, 64, device=dev), torch.randn(1000, 8, 64, device=dev), torch.randn(1000, 8, 64, device=dev)
    with torch.inference_mode():
        d = (mha(q, k, v) - attend_one_query(q, k, v, 8, 8)).abs().max().item()
    print(f"one-query attention vs MultiHeadAttention max|diff| = {d:.2e}")
    assert d < 1e-5, d


def main():
    dev = "cuda"
    test_fused_gru(dev)
    test_patch_stack(dev)
    test_one_query_attn(dev)
    frames = list_frames(DATA_ROOT / "plane_nose")
    t = len(frames) // 2
    L0, _ = (u8_to_input(load_image_u8(p), dev) for p in frames[t])
    L1, R1 = (u8_to_input(load_image_u8(p), dev) for p in frames[t + 1])
    # Production batch layout (Frontend.py / macvo.cpp): sample 0 = stereo L_t->R_t,
    # sample 1 = temporal L_{t-1}->L_t.
    A, B = torch.cat([L1, L0]), torch.cat([R1, L1])

    model = build_model("fp32", device=dev)        # strict load
    test_cost_front(model, dev)
    # batch invariance: a batch-2 call equals the two batch-1 calls (kernel-selection noise only)
    stereo = run_model(model, L1, R1)
    temporal = run_model(model, L0, L1)
    both = run_model(model, A, B)
    cat = {k: torch.cat([stereo[k], temporal[k]]) for k in stereo}
    print("B=2 vs 2xB=1 ->", stats(both, cat))
    assert mean_epe(both, cat) < 2e-2
    print("OK")


if __name__ == "__main__":
    main()
