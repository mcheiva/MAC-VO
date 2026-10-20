"""Where are the confidently wrong stereo disparities, and do they reach the keypoints MAC-VO tracks?

    python -m Scripts.AdHoc.StereoTailAnalysis --config Config/Train/CovHeadChol.yaml \
        --ckpt Model/MACVO_FrontendCov.pth Model/<run>/25000.pth --pairs 600 --out runs/stereo_tail

For each evaluation pair (frame0, frame1) of Evaluate.data (the seeded pairs of Train/MatchingNet/evaluate.py)
each head runs the frontend calls of Odometry/MACVO.py: stereo on frame0 and frame1 (depth0, depth1, as
FlowFormerCovFrontend.inference_2_depth: depth = f b / |u|, depth_cov = disparity_to_depth_cov) and the
match frame0 -> frame1. Keypoints are CovAwareSelector's candidates before its random subsampling: NMS on
(depth0_cov + depth1_cov) x (sigma_uu + sigma_vv - 2 sigma_uv), border, depth0 and depth1 below max_depth,
depth0_cov and match quality below min(max, 1.5 x their median over the NMS points). Defaults are the
Paper_Reproduce / ablation-study values (max_depth auto = f b).

Reported: stereo and match errors at the keypoints against the ground truth ((-f b / depth, 0) for stereo),
the depth error against MAC-VO's depth sigma, and features of the catastrophic stereo pixels
(r^T Sigma^-1 r > 100): GT depth, match outside the right image (x < d), half-occluded in the right view
(GT disparity z-buffer), depth edge, low local texture, signed disparity error. Panels of the frames with
the most catastrophic keypoints go to --out.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from DataLoader import TrainDataset, StereoFrame
from Module.Network.FlowFormerCov import build_flowformer
from Module.Network.FlowFormerCov.covhead import cov_entries
from Train.MatchingNet.evaluate import draw_pairs, load_batch
from Train.MatchingNet.loss import cov_nll
from Train.MatchingNet.utils import stereo_target
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm

CATASTROPHIC = 100.     # r^T Sigma^-1 r; P(chi^2_2 > 100) ~ 2e-22
DEPTH_BINS = [(0, 5), (5, 20), (20, 80), (80, np.inf)]


@torch.no_grad()
def frontend(model, cur, nxt, sel_cfg: SimpleNamespace) -> dict[str, torch.Tensor]:
    """The frontend outputs MACVO.py feeds the selector, plus the selected keypoint mask of frame0."""
    out = {}
    for name, img1, img2 in (("s0", cur.imageL, cur.imageR), ("s1", nxt.imageL, nxt.imageR), ("m", cur.imageL, nxt.imageL)):
        (flow, _), (c, _) = model(img1.cuda(), img2.cuda())
        out[f"flow_{name}"], out[f"c_{name}"], out[f"e_{name}"] = flow, c, cov_entries(c)
    fxb = (cur.K[:, 0, 0] * cur.baseline.view(-1)).view(-1, 1, 1, 1).cuda()
    for k in ("s0", "s1"):
        disp = out[f"flow_{k}"][:, :1].abs()
        out[f"depth_{k}"] = fxb / disp
        out[f"dcov_{k}"] = fxb ** 2 * out[f"e_{k}"][:, :1] / disp ** 4          # disparity_to_depth_cov
    e = out["e_m"]
    match_q = e[:, 0:1] + e[:, 1:2] - 2 * (e[:, 2:3] if e.size(1) == 3 else 0.)
    quality = (out["dcov_s0"] + out["dcov_s1"]) * match_q
    k = sel_cfg.kernel_size
    nms = (quality == -F.max_pool2d(-quality, k, 1, k // 2)) & ~quality.isnan()
    inside = torch.zeros_like(nms)
    w = sel_cfg.mask_width
    inside[..., w:-w, w:-w] = True
    max_depth = fxb if sel_cfg.max_depth == "auto" else sel_cfg.max_depth
    depth_ok = (out["depth_s0"] < max_depth) & (out["depth_s1"] < max_depth)
    dthr = torch.stack([out["dcov_s0"][i][nms[i]].nanmedian() * 1.5 for i in range(nms.size(0))]).clamp(max=sel_cfg.max_depth_cov)
    mthr = torch.stack([match_q[i][nms[i]].nanmedian() * 1.5 for i in range(nms.size(0))]).clamp(max=sel_cfg.max_match_cov)
    out["sel"] = nms & inside & depth_ok & (out["dcov_s0"] < dthr.view(-1, 1, 1, 1)) & (match_q < mthr.view(-1, 1, 1, 1))
    out["fxb"] = fxb
    return out


def stereo_features(img: torch.Tensor, depth: torch.Tensor, d_gt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-pixel [B, 1, H, W] geometric and photometric features of the left image."""
    W = d_gt.size(-1)
    x = torch.arange(W, device=d_gt.device, dtype=d_gt.dtype).view(1, 1, 1, W).expand_as(d_gt)
    defined = d_gt.isfinite()
    d = torch.where(defined, d_gt, 0.)
    col = (x - d).round().clamp(0, W - 1).long()                       # column of the match in the right image
    nearest = torch.full_like(d, -1.).scatter_reduce(3, col, torch.where(defined, d, -1.), "amax")
    logd = torch.where(depth.isfinite() & (depth > 0), depth, 1e6).log()
    span = F.max_pool2d(logd, 5, 1, 2) + F.max_pool2d(-logd, 5, 1, 2)   # local log-depth range (5x5)
    gray = img.mean(dim=1, keepdim=True)
    sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device).view(1, 1, 3, 3)
    grad = (F.conv2d(gray, sobel, padding=1) ** 2 + F.conv2d(gray, sobel.transpose(2, 3), padding=1) ** 2).sqrt()
    return {
        "out_of_view": defined & (x - d < 0),
        "half_occluded": defined & (x - d >= 0) & (d < nearest.gather(3, col) - 1.),
        "depth_edge": span > 0.1,
        "texture": F.avg_pool2d(grad, 9, 1, 4),
        "depth": depth,
    }


@torch.no_grad()
def analyse(model, seqs, pairs, batch_size: int, sel_cfg: SimpleNamespace):
    rec = {"sel": [], "cat": [], "base": []}
    per_image = []
    for b in ColoredTqdm(range(0, len(pairs), batch_size), desc="Analyse"):
        batch, _ = load_batch(seqs, pairs[b:b + batch_size])
        cur, nxt = batch.cur.stereo, batch.nxt.stereo
        o = frontend(model, cur, nxt, sel_cfg)
        gt_m, gt_s = cur.gt_flow.cuda(), stereo_target(cur).cuda()
        def_m, def_s = gt_m.isfinite().all(1, keepdim=True), gt_s.isfinite().all(1, keepdim=True)
        _, maha_m = cov_nll(o["flow_m"], gt_m, o["c_m"], def_m)
        _, maha_s = cov_nll(o["flow_s0"], gt_s, o["c_s0"], def_s)
        depth = cur.gt_depth.cuda()
        d_gt, d_pred = -gt_s[:, 0:1], -o["flow_s0"][:, 0:1]
        fields = {
            "maha_s": maha_s, "maha_m": maha_m,
            "derr": torch.where(def_s, d_pred - d_gt, 0.),
            "sigma_d": o["e_s0"][:, 0:1].sqrt(),
            "z_ratio2": torch.where(def_s, (o["depth_s0"] - depth) ** 2 / o["dcov_s0"], float("nan")),
        } | {k: v.float() for k, v in stereo_features(cur.imageL.cuda(), depth, d_gt).items()}
        cat = def_s & (maha_s > CATASTROPHIC)
        base = torch.rand_like(maha_s) < 1 / 64
        sel = o["sel"] & def_s
        for name, mask in (("sel", sel), ("cat", cat), ("base", base & def_s)):
            m = mask.reshape(-1)
            rec[name].append({k: v.reshape(-1)[m].cpu().numpy() for k, v in fields.items()})
        for i in range(d_gt.size(0)):
            per_image.append((b + i, sel[i].sum().item(), (sel[i] & cat[i]).sum().item()))
    return {k: {f: np.concatenate([r[f] for r in v]) for f in v[0]} for k, v in rec.items()}, per_image


def describe(name: str, s: dict[str, np.ndarray], base: dict[str, np.ndarray]):
    """Feature shares of set s, with the enrichment over random pixels in brackets."""
    if not len(s["maha_s"]):
        print(f"  {name:<36} n=0")
        return
    tex_lo = np.percentile(base["texture"], 25)
    shares = {"out of view": (s["out_of_view"] > 0).mean(), "half-occl": (s["half_occluded"] > 0).mean(),
              "depth edge": (s["depth_edge"] > 0).mean(), "low texture": (s["texture"] < tex_lo).mean()}
    ref = {"out of view": (base["out_of_view"] > 0).mean(), "half-occl": (base["half_occluded"] > 0).mean(),
           "depth edge": (base["depth_edge"] > 0).mean(), "low texture": 0.25}
    depth = " ".join(f"{((s['depth'] >= lo) & (s['depth'] < hi)).mean():.0%}" for lo, hi in DEPTH_BINS)
    print(f"  {name:<36} n={len(s['maha_s']):>8}  GT depth <5|5-20|20-80|>80 m: {depth}")
    print(f"  {'':<36} " + ", ".join(f"{k} {v:.1%} ({v / max(ref[k], 1e-9):.1f}x)" for k, v in shares.items()))
    print(f"  {'':<36} |disparity err| median {np.median(np.abs(s['derr'])):.2f} px, p90 {np.percentile(np.abs(s['derr']), 90):.1f} px; "
          f"predicted nearer than GT {(s['derr'] > 0).mean():.0%}; median sigma_d {np.median(s['sigma_d']):.3f} px")


def panel(model, seqs, pairs, image_ids: list[int], tag: str, out: Path, sel_cfg: SimpleNamespace):
    """Per frame: left image with keypoints (green; red = catastrophic stereo), GT and predicted disparity,
    log10 of the stereo r^T Sigma^-1 r and of the disparity sigma."""
    out.mkdir(parents=True, exist_ok=True)
    for i in image_ids:
        batch, _ = load_batch(seqs, [pairs[i]])
        cur, nxt = batch.cur.stereo, batch.nxt.stereo
        o = frontend(model, cur, nxt, sel_cfg)
        gt_s = stereo_target(cur).cuda()
        _, maha = cov_nll(o["flow_s0"], gt_s, o["c_s0"], gt_s.isfinite().all(1, keepdim=True))
        sel = o["sel"][0, 0].cpu().numpy()
        bad = sel & (maha[0, 0].cpu().numpy() > CATASTROPHIC)
        rgb = (cur.imageL[0].permute(1, 2, 0).numpy()[..., ::-1] * 255).astype(np.uint8).copy()
        for y, x in np.argwhere(sel & ~bad):
            cv2.circle(rgb, (int(x), int(y)), 1, (0, 255, 0), -1)
        for y, x in np.argwhere(bad):
            cv2.circle(rgb, (int(x), int(y)), 3, (0, 0, 255), -1)
        d_gt = torch.nan_to_num(-gt_s[0, 0], nan=0.).cpu().numpy()
        vmax = max(float(np.percentile(d_gt, 99)), 1.)
        def cmap(a, lo, hi): return cv2.applyColorMap(np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        tiles = [rgb, cmap(d_gt, 0, vmax), cmap(-o["flow_s0"][0, 0].cpu().numpy(), 0, vmax),
                 cmap(np.log10(maha[0, 0].clamp(min=1e-3).cpu().numpy()), -1, 3),
                 cmap(np.log10(o["e_s0"][0, 0].sqrt().cpu().numpy()), -1.5, 0.5), np.zeros_like(rgb)]
        names = ["left, keypoints (red: stereo r'S^-1 r > 100)", "GT disparity", "predicted disparity",
                 "stereo log10 r'S^-1 r (-1..3)", "log10 sigma_d px (-1.5..0.5)", f"{tag}, pair {i}"]
        for t, n in zip(tiles, names):
            cv2.putText(t, n, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{tag}_pair{i:04d}.png"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("Config/Train/CovHeadChol.yaml"))
    ap.add_argument("--ckpt", nargs="+", default=["Model/MACVO_FrontendCov.pth"])
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("runs/stereo_tail"))
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--mask_width", type=int, default=32)
    ap.add_argument("--max_depth", default="auto", type=lambda v: v if v == "auto" else float(v))
    ap.add_argument("--max_depth_cov", type=float, default=250.)
    ap.add_argument("--max_match_cov", type=float, default=100.)
    args = ap.parse_args()
    sel_cfg = SimpleNamespace(**{k: getattr(args, k) for k in ("kernel_size", "mask_width", "max_depth", "max_depth_cov", "max_match_cov")})

    cfg, _ = load_config(args.config)
    modelcfg = namespace_to_cfgnode(cfg.Model)
    seqs = [s for s in TrainDataset[StereoFrame].mp_instantiation(cfg.Evaluate.data, 0, -1, lambda c: True) if s is not None]
    pairs = draw_pairs(seqs, args.pairs, seed=0)

    for ckpt in args.ckpt:
        model = build_flowformer(modelcfg, torch.float32, torch.float32)
        model.load_ddp_state_dict(torch.load(ckpt, weights_only=True))
        model = model.cuda().eval()
        tag = "shipped" if ckpt.endswith("MACVO_FrontendCov.pth") else Path(ckpt).stem
        r, per_image = analyse(model, seqs, pairs, args.batch_size, sel_cfg)
        sel, cat, base = r["sel"], r["cat"], r["base"]
        n_img, n_kp = len(per_image), len(sel["maha_s"])
        counts = np.array([c for *_, c in per_image])
        order = np.argsort(-counts)
        z = sel["z_ratio2"][np.isfinite(sel["z_ratio2"])]
        print(f"\n=== {tag}: {n_img} pairs, {n_kp} keypoint candidates ({n_kp / n_img:.0f} per frame)")
        print(f"  at keypoints, stereo r'S^-1 r > 9.21: {np.mean(sel['maha_s'] > 9.21):.2%}, > 100: {np.mean(sel['maha_s'] > CATASTROPHIC):.2%} "
              f"| match > 9.21: {np.mean(sel['maha_m'] > 9.21):.2%}, > 100: {np.mean(sel['maha_m'] > CATASTROPHIC):.2%}   (calibrated: 1%, ~0)")
        print(f"  depth at keypoints vs MAC-VO's depth sigma, (dz / sigma_z)^2: median {np.median(z):.2f} (calibrated 0.45), "
              f"> 9: {np.mean(z > 9):.2%} (calibrated 0.27%), > 100: {np.mean(z > 100):.2%}")
        print(f"  catastrophic stereo keypoints in {np.mean(counts > 0):.0%} of frames; the worst 5% of frames hold "
              f"{counts[order[:max(1, n_img // 20)]].sum() / max(1, counts.sum()):.0%} of them")
        describe("random pixels (baseline)", base, base)
        describe("keypoints", sel, base)
        describe("keypoints, stereo catastrophic", {k: v[sel["maha_s"] > CATASTROPHIC] for k, v in sel.items()}, base)
        describe("all pixels, stereo catastrophic", cat, base)
        panel(model, seqs, pairs, [per_image[i][0] for i in order[:args.panels]], tag, args.out, sel_cfg)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
