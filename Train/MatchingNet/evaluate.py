"""
Evaluate FlowFormerCov covariance heads on a sequence list (handoff docs/covhead_retrain_handoff.md, section 7).

    python -m Train.MatchingNet.evaluate --config Config/Train/CovHeadChol.yaml \
        --ckpt Model/MACVO_FrontendCov.pth Model/<run>/30000.pth --pairs 600

Pairs are drawn once (seeded) from the config's Evaluate.data and shared by every checkpoint; each checkpoint
gets a fresh model in eval mode (the 2- or 3-channel head follows the checkpoint). Temporal pairs
(imageL_t, imageL_t+1) are scored against gt_flow, stereo pairs (imageL, imageR) against (-fx b / depth, 0).

    epe      mean |flow - gt| on flow_mask-valid pixels (temporal) / pixels with valid depth (stereo)
    nll      2D Gaussian negative log-likelihood in nats, 0.5 (r^T Sigma^-1 r + log det Sigma) + log 2 pi;
             nll_md is its median (the mean is dominated by a few confidently wrong pixels)
    calib    r^T Sigma^-1 r: mean 2 and median 1.386 when calibrated (chi^2 with 2 dof)
    tail>9   share with r^T Sigma^-1 r > 9.21 (1% when calibrated); tail>100 share > 100 (catastrophic, ~0)
    sigma    (det Sigma)^(1/4), the geometric-mean standard deviation in px
    filtered share above 1.5 x the pair's median sigma_uu + sigma_vv (MAC-VO's uncertainty filter, all pixels)
    |rho|    mean |sigma_uv| / sqrt(sigma_uu sigma_vv) (3-channel heads)

conf20 / conf40 restrict to the 20% / 40% of each image's pixels with the smallest sigma_uu + sigma_vv,
a stand-in for the confident pixels a keypoint selector keeps. Sparsification (Ilg et al. 2018), per image
and averaged: remove pixels by decreasing sigma_uu + sigma_vv (or, for the oracle, by decreasing error) and
track the mean error of the rest over 0-95% removal. AUSE is the mean gap between the two curves in px (0 is
a perfect ranking); err@20 is the mean error of the 20% kept, by the head and by the oracle.

Temporal pairs are also split by TartanAir mask class (v1: 0 valid, 1 cross-occluded, 10 self-occluded,
100 out of view); covariance statistics use every pixel with a finite target, as the training loss does.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import center_crop

from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType
from Module.Network.FlowFormerCov import build_flowformer
from Module.Network.FlowFormerCov.covhead import cov_entries
from Train.MatchingNet.loss import cov_nll
from Train.MatchingNet.utils import stereo_target
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm, Logger

MASK_CLASSES = {0: "valid", 1: "cross-occluded", 10: "self-occluded", 100: "out of view"}
SAMPLE_EVERY = 32       # per-pixel statistics use 1 in SAMPLE_EVERY pixels (same pixels for every checkpoint)
CHI2_2_P99   = 9.21     # 99% quantile of chi^2 with 2 dof
CONFIDENT    = {"conf20": 0.2, "conf40": 0.4}
SPARSIFY_AT  = np.linspace(0, 0.95, 20)

COLS_ALL   = ["epe", "nll", "nll_md", "calib", "calib_md", "tail>9", "tail>100", "sigma_md", "|rho|"]
COLS_CONF  = ["epe", "nll", "nll_md", "calib", "calib_md", "tail>9", "tail>100", "sigma_md", "err_md"]
COLS_CLASS = ["share", "nll", "calib", "calib_md", "tail>9", "sigma_md", "err_md", "filtered"]


def draw_pairs(seqs: list[TrainDataset], n: int, seed: int) -> list[tuple[int, int]]:
    offsets = np.cumsum([0] + [len(s) for s in seqs])
    picks = np.random.default_rng(seed).choice(offsets[-1], size=min(n, offsets[-1]), replace=False)
    seq_ids = np.searchsorted(offsets, picks, side="right") - 1
    return [(int(s), int(p - offsets[s])) for s, p in zip(seq_ids, picks)]


def pixel_stats(flow: torch.Tensor, c: torch.Tensor, gt: torch.Tensor, epe_mask: torch.Tensor,
                codes: torch.Tensor | None, first_image: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-pixel quantities of one batch ([B, ...] tensors on the GPU), subsampled."""
    defined = torch.isfinite(gt).all(dim=1, keepdim=True)
    nll, maha = cov_nll(flow, gt, c, defined)
    entries = cov_entries(c)
    trace = entries[:, 0:1] + entries[:, 1:2]
    per_image = lambda q: torch.quantile(trace.flatten(1), q, dim=1).view(-1, 1, 1, 1)
    image = torch.arange(first_image, first_image + trace.size(0), device=trace.device).view(-1, 1, 1, 1)
    stats = {
        "defined" : defined,
        "epe_mask": epe_mask & defined,
        "err"     : torch.where(defined, flow - gt, 0.).norm(dim=1, keepdim=True),
        "nll"     : 0.5 * nll + math.log(2 * math.pi),
        "calib"   : maha,
        "sigma"   : torch.exp(0.5 * (c[:, 0:1] + c[:, 1:2])),
        "trace"   : trace,
        "filtered": trace > 1.5 * per_image(0.5),
        "image"   : image.expand_as(trace),
    } | {name: trace <= per_image(q) for name, q in CONFIDENT.items()}
    if entries.size(1) == 3:
        stats["rho"] = (entries[:, 2:3] / (entries[:, 0:1] * entries[:, 1:2]).sqrt()).abs()
    if codes is not None:
        stats["code"] = codes
    flat = {k: v.permute(0, 2, 3, 1).reshape(-1).cpu().numpy() for k, v in stats.items()}
    pick = rng.choice(len(flat["defined"]), size=len(flat["defined"]) // SAMPLE_EVERY, replace=False)
    return {k: v[pick] for k, v in flat.items()}


def summarize(s: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, float]:
    d = s["defined"] & sel
    row = {
        "share"   : sel.mean(),
        "epe"     : s["err"][s["epe_mask"] & sel].mean() if (s["epe_mask"] & sel).any() else float("nan"),
        "nll"     : s["nll"][d].mean(),
        "nll_md"  : np.median(s["nll"][d]),
        "calib"   : s["calib"][d].mean(),
        "calib_md": np.median(s["calib"][d]),
        "tail>9"  : (s["calib"][d] > CHI2_2_P99).mean(),
        "tail>100": (s["calib"][d] > 100).mean(),
        "sigma_md": np.median(s["sigma"][d]),
        "err_md"  : np.median(s["err"][d]),
        "filtered": s["filtered"][d].mean(),
    }
    if "rho" in s:
        row["|rho|"] = s["rho"][d].mean()
    return row


def sparsification(s: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, float]:
    """Per-image sparsification curves (by sigma_uu + sigma_vv and by the oracle error), averaged over images."""
    d = np.flatnonzero(s["defined"] & sel)
    d = d[np.argsort(s["image"][d], kind="stable")]
    starts = np.flatnonzero(np.diff(s["image"][d], prepend=-1))
    pred, oracle, kept20 = [], [], []
    for idx in np.split(d, starts[1:]):
        if len(idx) < 100:
            continue
        err = s["err"][idx]
        by_head, by_oracle = err[np.argsort(s["trace"][idx], kind="stable")], np.sort(err)
        keep = np.maximum(1, ((1 - SPARSIFY_AT) * len(err)).astype(int))
        pred.append(np.cumsum(by_head)[keep - 1] / keep)
        oracle.append(np.cumsum(by_oracle)[keep - 1] / keep)
        k20 = max(1, len(err) // 5)
        kept20.append((by_head[:k20].mean(), by_oracle[:k20].mean()))
    pred, oracle, kept20 = np.mean(pred, axis=0), np.mean(oracle, axis=0), np.mean(kept20, axis=0)
    return {
        "AUSE"         : np.trapezoid(pred - oracle, SPARSIFY_AT) / SPARSIFY_AT[-1],
        "err@100"      : pred[0],
        "err@20 head"  : kept20[0],
        "err@20 oracle": kept20[1],
    }


def load_batch(seqs: list[TrainDataset], pairs: list[tuple[int, int]]) -> tuple[DataFramePair[StereoFrame], torch.Tensor]:
    """Pairs through the training crop, plus their raw mask codes [B, 1, H, W] cropped the same way."""
    crop, cast = CenterCropFrame(dict(width=640, height=480)), CastDataType(dict(dtype="fp32"))
    frames, codes = [], []
    for s, i in pairs:
        pair = seqs[s][i]
        frames.append(DataFramePair(idx=pair.idx, cur=cast(crop(pair.cur)), nxt=cast(crop(pair.nxt)), time_ns=pair.time_ns))
        seq = seqs[s].sequence
        code = torch.from_numpy(seq.flow_loader.mask_codes(seq.get_index(seqs[s].get_index(i))).astype(np.int16))
        codes.append(center_crop(code[None], [480, 640]))
    return DataFramePair.collate(frames), torch.stack(codes)


@torch.no_grad()
def evaluate(models: dict[str, torch.nn.Module], seqs: list[TrainDataset], pairs: list[tuple[int, int]],
             batch_size: int, stereo: bool, seed: int) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """ckpt -> kind -> per-pixel samples. Each batch is loaded once and run through every model."""
    out = {ckpt: {"temporal": [], "stereo": []} for ckpt in models}
    for b in ColoredTqdm(range(0, len(pairs), batch_size), desc="Evaluate"):
        batch, codes = load_batch(seqs, pairs[b:b + batch_size])
        cur, nxt = batch.cur.stereo, batch.nxt.stereo
        kinds = [("temporal", cur.imageL, nxt.imageL, cur.gt_flow, cur.flow_mask.bool(), codes)]
        if stereo and cur.gt_depth is not None:
            kinds.append(("stereo", cur.imageL, cur.imageR, stereo_target(cur), None, None))
        for kind, img1, img2, gt, valid, code in kinds:
            gt = gt.cuda()
            epe_mask = gt.norm(dim=1, keepdim=True) < 400
            if valid is not None: epe_mask &= valid.cuda()
            for ckpt, model in models.items():
                (flow, _), (c, _) = model(img1.cuda(), img2.cuda())
                rng = np.random.default_rng([seed, b, int(kind == "stereo")])     # same pixels for every model
                out[ckpt][kind].append(pixel_stats(flow, c, gt, epe_mask, None if code is None else code.cuda(), b, rng))
    return {ckpt: {k: {n: np.concatenate([s[n] for s in v]) for n in v[0]} for k, v in r.items() if v}
            for ckpt, r in out.items()}


def print_table(title: str, rows: list[tuple[str, dict[str, float]]], cols: list[str] | None = None):
    cols = cols or list(rows[0][1])
    width = max(len(r) for r, _ in rows)
    widths = [max(9, len(c)) for c in cols]
    print(f"\n{title}\n{'':{width}}  " + "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    for name, row in rows:
        print(f"{name:{width}}  " + "  ".join(f"{row.get(c, float('nan')):{w}.4f}" for c, w in zip(cols, widths)))


def report(results: dict[str, dict[str, dict[str, np.ndarray]]]):
    """Print every table for results keyed by display name."""
    for kind in ("temporal", "stereo"):
        rs = {name: r[kind] for name, r in results.items() if kind in r}
        if not rs:
            continue
        everything = {name: np.ones_like(s["defined"]) for name, s in rs.items()}
        print_table(f"{kind}: all pixels with a finite target", [(n, summarize(s, everything[n])) for n, s in rs.items()], COLS_ALL)
        for conf, q in CONFIDENT.items():
            print_table(f"{kind}: {q:.0%} most confident pixels per image", [(n, summarize(s, s[conf])) for n, s in rs.items()], COLS_CONF)
        print_table(f"{kind}: sparsification by sigma_uu + sigma_vv (px)", [(n, sparsification(s, everything[n])) for n, s in rs.items()])
    rows = [(f"{cls} | {name}", summarize(r["temporal"], r["temporal"]["code"] == code))
            for code, cls in MASK_CLASSES.items() for name, r in results.items()
            if "temporal" in r and (r["temporal"]["code"] == code).any()]
    if rows:
        print_table("temporal: by mask class", rows, COLS_CLASS)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FlowFormerCov covariance heads")
    parser.add_argument("--config", type=Path, default=Path("Config/Train/CovHeadChol.yaml"))
    parser.add_argument("--ckpt", type=str, nargs="+", default=["Model/MACVO_FrontendCov.pth"])
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-stereo", action="store_true")
    args = parser.parse_args()

    cfg, _ = load_config(args.config)
    modelcfg = namespace_to_cfgnode(cfg.Model)
    seqs = TrainDataset[StereoFrame].mp_instantiation(cfg.Evaluate.data, 0, -1, lambda c: c.type in {"TartanAir_NoIMU", "TartanAirv2_NoIMU"})
    seqs = [s for s in seqs if s is not None]
    pairs = draw_pairs(seqs, args.pairs, args.seed)
    Logger.write("info", f"{len(seqs)} sequences, {len(pairs)} pairs")

    models = {}
    for ckpt in args.ckpt:
        model = build_flowformer(modelcfg, torch.float32, torch.float32)
        model.load_ddp_state_dict(torch.load(ckpt, weights_only=True))
        models[ckpt] = model.cuda().eval()
    results = evaluate(models, seqs, pairs, args.batch_size, not args.no_stereo, args.seed)

    unique = len({Path(c).name for c in args.ckpt}) == len(args.ckpt)
    report({(Path(c).name if unique else c): r for c, r in results.items()})


if __name__ == "__main__":
    main()
