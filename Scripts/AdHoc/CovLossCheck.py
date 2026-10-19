"""Section 5.1 of docs/covhead_retrain_handoff.md: which quantity does the covariance head emit at
training time, and does Train/MatchingNet/loss.py treat it consistently?

    python -m Scripts.AdHoc.CovLossCheck --config Config/Train/Demo.yaml --batches 8

Runs FlowFormerCov in train mode (as train_flowformer.py does, no gradient) on random training
pairs through the trainer's transforms, and reads the decoder output c = cov_preds[-1] per axis as
    A  c = log sigma   (FlowFormerCov.inference returns exp(2c), evaluate.py uses the same)
    B  c = sigma^2     (cov_loss: err^2 / c + log c)
For a calibrated Gaussian z = err^2 / sigma^2 follows chi^2(1): mean 1, median 0.455; the 2D ratio
r^T Sigma^-1 r averages 2. Pixels: flow_mask valid and |gt| < max_flow.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType, AddImageNoise, ScaleFrame
from Module.Network.FlowFormerCov import build_flowformer
from Train.MatchingNet.loss import cov_loss
from Train.MatchingNet.utils import IndexedDataset
from Utility.Config import load_config, namespace_to_cfgnode

CHI2_1_MEDIAN = 0.4549
ABS_NORMAL_MEDIAN = 0.6745


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("Config/Train/Demo.yaml"))
    ap.add_argument("--ckpt", type=str, default=None, help="default: Model.restore_ckpt")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--pixels-per-batch", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, _ = load_config(args.config)
    modelcfg = namespace_to_cfgnode(cfg.Model)
    modelcfg.update(dict(training_mode="cov"))
    transforms = [CenterCropFrame(dict(width=640, height=480)),
                  CastDataType(dict(dtype=cfg.Model.datatype)),
                  AddImageNoise(dict(stdv=5.0)),
                  ScaleFrame(dict(scale_u=cfg.Model.image_scale, scale_v=cfg.Model.image_scale, interp='nearest'))]
    seqs = TrainDataset[StereoFrame].mp_instantiation(cfg.Train.data, 0, -1, lambda c: c.type in {"TartanAir_NoIMU", "TartanAirv2_NoIMU"})
    seqs = [s for s in seqs if s is not None]
    print(f"{len(seqs)} sequences, {sum(len(s) for s in seqs)} pairs")
    loader = DataLoader(ConcatDataset([IndexedDataset(s.transform_source(transforms)) for s in seqs]),
                        batch_size=modelcfg.batch_size, shuffle=True, drop_last=True, num_workers=4,
                        collate_fn=DataFramePair.collate, generator=torch.Generator().manual_seed(args.seed))

    model = build_flowformer(modelcfg, torch.float32, torch.float32)
    model.load_ddp_state_dict(torch.load(args.ckpt or modelcfg.restore_ckpt, weights_only=True))
    model = model.cuda().train()

    rng = np.random.default_rng(args.seed)
    cs, errs, losses, nan_frac, eval_diff = [], [], [], [], None
    with torch.no_grad():
        for k, frame in enumerate(loader):
            if k == args.batches:
                break
            img1, img2 = frame.cur.stereo.imageL.cuda(), frame.nxt.stereo.imageL.cuda()
            gt = frame.cur.stereo.gt_flow.cuda()
            valid = (gt.norm(dim=1, keepdim=True) < modelcfg.max_flow) & frame.cur.stereo.flow_mask.cuda().bool()
            flow_preds, cov_preds = model(img1, img2)
            if k == 0:   # train-mode output == eval-mode output (no stochastic layers)
                model.eval()
                eval_diff = (model(img1, img2)[1][0] - cov_preds[-1]).abs().max().item()
                model.train()

            # the shipped loss, exactly as train_flowformer.py calls it (cov_mask: False -> no mask)
            losses.append(cov_loss(modelcfg.gamma, flow_preds, gt, cov_preds)[0].item())
            c = cov_preds[-1]
            term = (flow_preds[-1] - gt) ** 2 / (c + 1e-7) + torch.log(c + 1e-7)
            nan_frac.append(term.isnan().float().mean().item())

            # per-pixel samples, both axes kept together: [N, 2]
            m = valid[:, 0].reshape(-1)
            c2 = c.permute(0, 2, 3, 1).reshape(-1, 2)[m].cpu().numpy()
            e2 = (flow_preds[-1] - gt).permute(0, 2, 3, 1).reshape(-1, 2)[m].cpu().numpy()
            pick = rng.choice(len(c2), size=min(args.pixels_per_batch, len(c2)), replace=False)
            cs.append(c2[pick]); errs.append(e2[pick])

    c, err = np.concatenate(cs), np.concatenate(errs)
    err2 = err ** 2
    print(f"\nbatches {len(losses)} x {modelcfg.batch_size} pairs, {len(c)} sampled valid pixels")
    print(f"train vs eval mode max |dc| on batch 0: {eval_diff:.2e}")

    print("\n1. decoder output c = cov_preds[-1]")
    for a, name in enumerate(["u", "v"]):
        q = np.percentile(c[:, a], [0, 1, 50, 99, 100])
        print(f"   {name}: min {q[0]:.2f}  p1 {q[1]:.2f}  median {q[2]:.2f}  p99 {q[3]:.2f}  max {q[4]:.2f}  "
              f"fraction < 0: {(c[:, a] < 0).mean():.3f}")

    print("\n2. shipped cov_loss (err^2/c + log c)")
    print(f"   loss per batch: {['nan' if np.isnan(l) else f'{l:.3f}' for l in losses]}")
    print(f"   NaN fraction of last-iteration per-pixel terms: {np.mean(nan_frac):.3f}")

    print("\n3. reading A, c = log sigma (sigma^2 = exp(2c))")
    zA = err2 * np.exp(-2 * c)
    for a, name in enumerate(["u", "v"]):
        print(f"   {name}: z mean {zA[:, a].mean():.3f}  median {np.median(zA[:, a]):.3f}   (calibrated: 1, {CHI2_1_MEDIAN})")
    nllA = 0.5 * zA + c + 0.5 * np.log(2 * np.pi)
    print(f"   r^T Sigma^-1 r mean {zA.sum(1).mean():.3f}  median {np.median(zA.sum(1)):.3f}   (calibrated: 2, 1.386)")
    print(f"   Gaussian NLL per axis {nllA.mean():.3f}")

    print("\n4. reading B, c = sigma^2 (only where c > 0)")
    for a, name in enumerate(["u", "v"]):
        pos = c[:, a] > 0
        zB = err2[pos, a] / c[pos, a]
        print(f"   {name}: valid variances {pos.mean():.3f}  z mean {zB.mean():.3f}  median {np.median(zB):.3f}")

    print("\n5. median |err| per c decile (u and v pooled) vs what each reading predicts")
    cf, ef = c.reshape(-1), np.abs(err).reshape(-1)
    edges = np.percentile(cf, np.linspace(0, 100, 11))
    print(f"   {'c range':>16}  {'median|err|':>11}  {'A: .674*e^c':>11}  {'B: .674*sqrt(c)':>15}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (cf >= lo) & (cf <= hi)
        cm = np.median(cf[sel])
        b = f"{ABS_NORMAL_MEDIAN * np.sqrt(cm):.3f}" if cm > 0 else "n/a (c<=0)"
        print(f"   {lo:7.2f}..{hi:6.2f}  {np.median(ef[sel]):11.3f}  {ABS_NORMAL_MEDIAN * np.exp(cm):11.3f}  {b:>15}")


if __name__ == "__main__":
    main()
