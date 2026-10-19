"""
Evaluate FlowFormerCov covariance heads on a sequence list (handoff docs/covhead_retrain_handoff.md, section 7).

    python -m Train.MatchingNet.evaluate --config Config/Train/CovHeadChol.yaml \
        --ckpt Model/MACVO_FrontendCov.pth Model/<run>/30000.pth --pairs 600

Pairs are drawn once (seeded) from the config's Evaluate.data and shared by every checkpoint; each checkpoint
gets a fresh model in eval mode (the 2- or 3-channel head follows the checkpoint). Temporal pairs
(imageL_t, imageL_t+1) are scored against gt_flow, stereo pairs (imageL, imageR) against (-fx b / depth, 0).

    epe      mean |flow - gt| on flow_mask-valid pixels (temporal) / pixels with valid depth (stereo)
    nll      2D Gaussian negative log-likelihood in nats, 0.5 (r^T Sigma^-1 r + log det Sigma) + log 2 pi
    calib    r^T Sigma^-1 r: mean 2 and median 1.386 when calibrated (chi^2 with 2 dof)
    sigma    (det Sigma)^(1/4), the geometric-mean standard deviation in px
    filtered share above 1.5 x the pair's median sigma_uu + sigma_vv (MAC-VO's uncertainty filter, all pixels)
    |rho|    mean |sigma_uv| / sqrt(sigma_uu sigma_vv) (3-channel heads)

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


def draw_pairs(seqs: list[TrainDataset], n: int, seed: int) -> list[tuple[int, int]]:
    offsets = np.cumsum([0] + [len(s) for s in seqs])
    picks = np.random.default_rng(seed).choice(offsets[-1], size=min(n, offsets[-1]), replace=False)
    seq_ids = np.searchsorted(offsets, picks, side="right") - 1
    return [(int(s), int(p - offsets[s])) for s, p in zip(seq_ids, picks)]


def pixel_stats(flow: torch.Tensor, c: torch.Tensor, gt: torch.Tensor, epe_mask: torch.Tensor,
                codes: torch.Tensor | None, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-pixel quantities of one batch ([B, ...] tensors on the GPU), subsampled."""
    defined = torch.isfinite(gt).all(dim=1, keepdim=True)
    nll, maha = cov_nll(flow, gt, c, defined)
    entries = cov_entries(c)
    trace = entries[:, 0:1] + entries[:, 1:2]
    thresh = 1.5 * trace.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    stats = {
        "defined" : defined,
        "epe_mask": epe_mask & defined,
        "err"     : torch.where(defined, flow - gt, 0.).norm(dim=1, keepdim=True),
        "nll"     : 0.5 * nll + math.log(2 * math.pi),
        "calib"   : maha,
        "sigma"   : torch.exp(0.5 * (c[:, 0:1] + c[:, 1:2])),
        "filtered": trace > thresh,
    }
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
        "calib"   : s["calib"][d].mean(),
        "calib_md": np.median(s["calib"][d]),
        "sigma_md": np.median(s["sigma"][d]),
        "err_md"  : np.median(s["err"][d]),
        "filtered": s["filtered"][d].mean(),
    }
    if "rho" in s:
        row["|rho|"] = s["rho"][d].mean()
    return row


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
                out[ckpt][kind].append(pixel_stats(flow, c, gt, epe_mask, None if code is None else code.cuda(), rng))
    return {ckpt: {k: {n: np.concatenate([s[n] for s in v]) for n in v[0]} for k, v in r.items() if v}
            for ckpt, r in out.items()}


def print_table(title: str, rows: list[tuple[str, dict[str, float]]]):
    cols = list(rows[0][1])
    width = max(len(r) for r, _ in rows)
    print(f"\n{title}\n{'':{width}}  " + "  ".join(f"{c:>9}" for c in cols))
    for name, row in rows:
        print(f"{name:{width}}  " + "  ".join(f"{row.get(c, float('nan')):9.4f}" for c in cols))


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

    names = {ckpt: Path(ckpt).name if len({Path(c).name for c in args.ckpt}) == len(args.ckpt) else ckpt for ckpt in args.ckpt}
    for kind in ("temporal", "stereo"):
        rows = [(names[c], summarize(r[kind], np.ones_like(r[kind]["defined"]))) for c, r in results.items() if kind in r]
        if rows:
            print_table(f"{kind} pairs, all pixels with a finite target", rows)
    rows = []
    for code, cls in MASK_CLASSES.items():
        for c, r in results.items():
            sel = r["temporal"]["code"] == code
            if sel.any():
                rows.append((f"{cls} | {names[c]}", summarize(r["temporal"], sel)))
    print_table("temporal pairs by mask class", rows)


if __name__ == "__main__":
    main()
