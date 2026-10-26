"""Self-checks for the exact (weight-preserving) rewrites. Run: python Scripts/Parity/test_exact.py"""
import importlib.util
import subprocess
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
    """PatchEmbed.forward_tm + CrossAttentionLayer.forward_tm == original token-major path."""
    enc = model.memory_encoder.cost_perceiver_encoder
    torch.manual_seed(0)
    cost_maps = torch.randn(2 * 100, 1, 40, 40, device=dev) * 3     # pads 40 -> 40 (multiple of 8), 5x5 tokens
    with torch.inference_mode():
        x0, s0 = enc.patch_embed(cost_maps)
        ref = enc.input_layer(enc.latent_tokens, x0)
        x1, s1 = enc.patch_embed.forward_tm(cost_maps)
        new = enc.input_layer.forward_tm(enc.latent_tokens, x1)
    d_tok = (x0 - x1).abs().max().item()
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


# Last commit before the deployment graph rewrites: the reference the exactness tests compare against.
_REFERENCE_COMMIT = "7b640af"


def _head_module(rel: str, name: str):
    """The pre-rewrite version of a Module/ source file, loaded as a sibling module of the patched one."""
    root = Path(__file__).resolve().parents[2]
    src = subprocess.run(["git", "show", f"{_REFERENCE_COMMIT}:Module/{rel}"], cwd=root, check=True,
                         capture_output=True).stdout.decode()
    pkg = "Network.FlowFormer.core"
    spec = importlib.util.spec_from_loader(f"{pkg}.{name}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[spec.name] = mod
    exec(compile(src, f"{_REFERENCE_COMMIT}:{rel}", "exec"), mod.__dict__)
    return mod


def test_upsample(dev: str):
    """upsample: 9 shifted slices == F.unfold(3x3) in the convex upsampler, bit-identical (fp32)."""
    from Network.FlowFormer.core.decoder import MemoryDecoder
    torch.manual_seed(0)
    flow = torch.randn(2, 2, 11, 13, device=dev) * 4
    mask = torch.randn(2, 9 * 64, 11, 13, device=dev)
    with torch.inference_mode():
        new = MemoryDecoder.upsample_flow(None, flow, mask)
        N, C, H, W = flow.shape
        m = mask.view(N, 9, 8, 8, H, W).softmax(dim=1)
        up = F.unfold(F.pad(8 * flow, (1, 1, 1, 1)), (3, 3), padding=0).view(N, C, 9, H, W)
        ref = (m.unsqueeze(1) * up.unsqueeze(-3).unsqueeze(-3)).sum(dim=2).permute(0, 1, 4, 2, 5, 3).reshape(N, C, 8 * H, 8 * W)
    print(f"upsample vs F.unfold: bit-identical = {torch.equal(new, ref)}")
    assert torch.equal(new, ref)


def test_covsum(dev: str):
    """covsum: one stacked sum of the 12 cov deltas vs the running add from 0 (exact in real arithmetic, fp32 <= 5e-7)."""
    torch.manual_seed(0)
    deltas = [torch.randn(2, 3, 88, 88, device=dev) * 0.1 for _ in range(12)]
    with torch.inference_mode():
        run = torch.zeros_like(deltas[0])
        for d in deltas:
            run = run + d
        once = torch.zeros_like(deltas[0]) + torch.stack(deltas, 0).sum(0)
        exact = torch.stack([d.double() for d in deltas], 0).sum(0)
    d_run, d_exact = (once - run).abs().max().item(), (once.double() - exact).abs().max().item()
    print(f"covsum vs running add max|diff| = {d_run:.2e}, vs fp64 sum {d_exact:.2e}")
    assert d_run <= 5e-7 and d_exact <= 5e-7, (d_run, d_exact)


def test_ctxsplit(dev: str):
    """ctxsplit: the pre-rewrite cat[x, context] attention modules vs the split-context ones, same weights, fp64."""
    from Network.FlowFormer.core import twins
    old = _head_module("Network/FlowFormer/core/twins.py", "twins_head")
    torch.manual_seed(0)
    B0, T, dim, vc = 2, 3, 64, 32
    for name, args, H, W in (("LocallyGroupedAttnRPEContext", dict(dim=dim, num_heads=8, ws=7, vert_c_dim=vc), 10, 10),
                             ("GlobalSubSampleAttnRPEContext", dict(dim=dim, num_heads=8, sr_ratio=4, vert_c_dim=vc), 10, 10)):
        new_m = getattr(twins, name)(**args).to(dev).double().eval()
        old_m = getattr(old, name)(**args).to(dev).double().eval()
        old_m.load_state_dict(new_m.state_dict())
        x = torch.randn(B0 * T, H * W, dim, device=dev, dtype=torch.float64)
        ctx = torch.randn(B0, 256, H, W, device=dev, dtype=torch.float64)
        with torch.inference_mode():
            a, b = new_m(x, (H, W), ctx), old_m(x, (H, W), ctx)
        d = (a - b).abs().max().item()
        print(f"ctxsplit {name} vs {_REFERENCE_COMMIT} (fp64) max|diff| = {d:.2e}  (|out| mean {b.abs().mean():.3f})")
        assert a.shape == b.shape and d < 1e-12, d


def test_pad_constant_zero(dev: str):
    """pad_constant_zero: the expanded one-element zero == the pre-rewrite dense torch.zeros pieces == F.pad, bit-identical."""
    from Network.FlowFormer.core.pad_helper import pad_constant_zero
    old = _head_module("Network/FlowFormer/core/pad_helper.py", "pad_helper_head").pad_constant_zero
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        x = (torch.randn(2, 5, 6, 7, device=dev) * 3).to(dtype)
        for pad in ((0, 2, 0, 3), (1, 1, 1, 1), (0, 0, 2, 0), (1, 2, 3, 4, 0, 1)):
            with torch.inference_mode():
                new, ref, fp = pad_constant_zero(x, pad), old(x, pad), F.pad(x, pad)
            assert new.dtype == ref.dtype == x.dtype and torch.equal(new, ref) and torch.equal(new, fp), (dtype, pad)
    print(f"pad_constant_zero vs {_REFERENCE_COMMIT} dense zeros and F.pad: bit-identical (fp32/fp16/bf16/fp64, 4 pad shapes)")


def main():
    dev = "cuda"
    test_fused_gru(dev)
    test_patch_stack(dev)
    test_one_query_attn(dev)
    test_upsample(dev)
    test_covsum(dev)
    test_ctxsplit(dev)
    test_pad_constant_zero(dev)
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
