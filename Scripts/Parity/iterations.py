"""How far from converged is the decoder at 12 iterations, and where?

For reference pairs, runs the decoder for `--max` iterations once, records the flow after every
iteration, and reports the distance of the k-iteration flow to the `--max`-iteration flow:
over all pixels, and over the pixels a covariance-aware selector would keep (flow covariance
below the frame median at 12 iterations). Also prints the per-iteration mean |delta_flow|.

    python Scripts/Parity/iterations.py --max 32 --depths 8 12 16 20 24 --limit 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=ROOT / "runs" / "ref")
    ap.add_argument("--max", type=int, default=32)
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    ap.add_argument("--limit", type=int, default=12, help="frames (x3 pair types)")
    ap.add_argument("--mode", default="fp32", choices=["fp32", "fast"])
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.25, 0.1],
                    help="keep pixels whose flow covariance is below these quantiles of the frame")
    args = ap.parse_args()

    model = build_model(args.mode)
    dec = model.memory_decoder
    dec.depth = args.max
    snaps: list[torch.Tensor] = []
    h = dec.update_block.flow_head.register_forward_hook(lambda m, i, o: snaps.append(o.detach().float()))
    csnaps: list[torch.Tensor] = []
    hc = dec.cov_update.cov_head.register_forward_hook(lambda m, i, o: csnaps.append(o.detach().float()))

    files = sorted(args.ref.glob("*/*_stereo.npz"))[:args.limit]
    res = {p: {k: [] for k in args.depths} for p in ("stereo", "temporal", "kf")}
    curves = {p: [] for p in res}
    for f in files:
        for p in res:
            d = np.load(str(f).replace("_stereo", f"_{p}"))
            i1 = torch.from_numpy(d["img1"][None]).cuda().float() / 255
            i2 = torch.from_numpy(d["img2"][None]).cuda().float() / 255
            snaps.clear(); csnaps.clear()
            dec.iter_log = []
            with torch.inference_mode():
                (flow_max, _), (cov12_dummy, _) = model(i1, i2)
            curves[p].append(np.asarray(dec.iter_log))
            # 1/8-res flow after k iterations = cumulative sum of the per-iteration deltas
            deltas = torch.stack(snaps)                              # [max, 1, 2, H8, W8]
            cum = deltas.cumsum(0)
            final = cum[-1]
            # selection mask from the 12-iteration covariance: run cov at 12 via the reference set
            cov12 = torch.from_numpy(d["cov"][None]).cuda()          # fp32 reference at 12 iterations (log sigma)
            q = torch.exp(2 * cov12).sum(1, keepdim=True)            # sigma_u^2 + sigma_v^2 at full res
            sels8 = [torch.nn.functional.avg_pool2d((q <= q.flatten().quantile(qq)).float(), 8) > 0.5
                     for qq in args.quantiles]                        # selector-like masks at 1/8 res
            ccum = torch.stack(csnaps).cumsum(0)                     # log sigma (1/8 res) after k iterations
            for k in args.depths:
                e8 = (cum[k - 1] - final).norm(dim=1, keepdim=True) * 8.0      # full-res px
                dls = (ccum[k - 1] - ccum[11]).mean(1, keepdim=True)          # signed change of log sigma vs 12 iters
                res[p][k].append([e8.mean().item()] + [e8[m].mean().item() for m in sels8]
                                 + [dls[sels8[0]].abs().mean().item()])
        print(f"{d['seq']} {int(d['frame'])}", flush=True)
    h.remove(); hc.remove()

    print(f"\nmode {args.mode}, reference = {args.max} iterations, {len(files)} frames; EPE in full-res px")
    print(f"{'pass':9s}{'iters':>6s}{'all px':>10s}" + "".join(f"{'cov<q' + str(qq):>10s}" for qq in args.quantiles)
          + f"{'|dlogs|':>9s}   (|dlogs| = mean |log sigma change| vs 12 iters on cov<q{args.quantiles[0]})")
    for p in res:
        for k in args.depths:
            a = np.array(res[p][k])
            print(f"{p:9s}{k:6d}" + "".join(f"{v:10.3f}" for v in a.mean(0)[:-1]) + f"{a.mean(0)[-1]:9.3f}")
    print("\nmean |delta_flow| per iteration (1/8-res px), averaged over frames:")
    for p in res:
        c = np.mean(curves[p], axis=0)
        print(f"{p:9s} " + " ".join(f"{v:.3f}" for v in c))


if __name__ == "__main__":
    main()
