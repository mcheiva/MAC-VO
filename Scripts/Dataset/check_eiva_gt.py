"""
Per-frame consistency of rendered EIVA ground truth (Scripts/Dataset/make_eiva_gt.py) with the stereo images.

    python -m Scripts.Dataset.check_eiva_gt --seq D:/Datasets/EIVA/vobster_quay/plane_nose --gt <dir>/plane_nose

The photogrammetry reconstruction is locally wrong in places: a frame whose pose or surface is off renders a depth
that disagrees with its own rectified stereo pair by a few percent everywhere (Welland frame 800: 5.5%). The
frozen FlowFormer measures that: per frame, the median of predicted / rendered disparity (rendered = fx b / depth)
over the 25% of GT pixels the shipped head is most confident about. It is a per-frame scale statistic, robust to
the per-pixel matching failures the covariance head has to learn. A wrong relative pose between two frames keeps
the scale but shifts the whole flow field (plane_tail2 pair 504 -> 505: 2 px on every confident pixel), so the
temporal pair is checked the same way: the median |predicted - rendered flow| on its confident GT pixels.
Both go into index.json (`disparity_ratio`, `flow_error`); EIVA_NoIMU leaves frames beyond its
`max_scale_error` / `max_flow_error` unsupervised.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Module"))
from Network.FlowFormer.configs.submission import get_cfg     # noqa: E402
from Network.FlowFormerCov import build_flowformer            # noqa: E402


def load_image(path: Path, hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(path)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), hw[::-1], interpolation=cv2.INTER_AREA)
    return image.astype(np.float32).transpose(2, 0, 1) / 255.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="make_eiva_gt.py output (index.json is updated in place)")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "Model" / "MACVO_FrontendCov.pth")
    ap.add_argument("--batch", type=int, default=2, help="frames per forward pass (each gives a stereo and a temporal pair)")
    args = ap.parse_args()

    index_path = args.gt / "index.json"
    index = json.loads(index_path.read_text())
    names, hw = index["frames"], tuple(index["hw"])
    fxb = index["K"][0][0] * index["baseline"]
    cfg = get_cfg()
    cfg.latentcostformer.decoder_depth = 12
    model = build_flowformer(cfg, torch.float32, torch.float32)
    model.load_ddp_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model = model.eval().cuda()
    left, right = args.seq / "processed" / "left", args.seq / "processed" / "right"

    def confident(trace: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
        return valid & (trace <= np.quantile(trace[valid], 0.25)) if valid.sum() >= 1000 else None

    ratios, flow_errors = [], []
    for b in range(0, len(names), args.batch):
        batch = names[b:b + args.batch]
        pairs = [(left / n, right / n.replace("image_left", "image_right")) for n in batch]       # stereo
        pairs += [(left / n, left / names[b + k + 1]) for k, n in enumerate(batch) if b + k + 1 < len(names)]   # temporal
        i1 = torch.from_numpy(np.stack([load_image(p, hw) for p, _ in pairs])).cuda()
        i2 = torch.from_numpy(np.stack([load_image(p, hw) for _, p in pairs])).cuda()
        with torch.inference_mode():
            flow, cov = model.inference(i1, i2)
        flow, trace = flow.float().cpu().numpy(), cov[:, :2].sum(1).float().cpu().numpy()
        for k, name in enumerate(batch):
            gt = np.load(args.gt / f"{Path(name).stem}.npz")
            depth = gt["depth"].astype(np.float32)
            conf = confident(trace[k], depth > 0)
            ratios.append(float(np.median(-flow[k, 0][conf] * depth[conf] / fxb)) if conf is not None else float("nan"))
            t = len(batch) + k
            if "flow" in gt and t < len(flow):
                flow_gt = gt["flow"].astype(np.float32)
                conf = confident(trace[t], np.isfinite(flow_gt).all(0))
                flow_errors.append(float(np.median(np.linalg.norm(flow[t][:, conf] - flow_gt[:, conf], axis=0)))
                                   if conf is not None else float("nan"))
            else:
                flow_errors.append(float("nan"))
        if b % 400 == 0:
            print(f"  {b}/{len(names)}: ratio {ratios[-1]:.4f}, flow error {flow_errors[-1]:.3f} px", flush=True)

    r, fe = np.array(ratios), np.array(flow_errors)
    index["disparity_ratio"] = [round(v, 5) for v in ratios]
    index["flow_error"] = [round(v, 4) for v in flow_errors]
    index_path.write_text(json.dumps(index, indent=1))
    ok, okf = np.isfinite(r), np.isfinite(fe)
    print(f"{args.seq.name}: median ratio {np.median(r[ok]):.4f}, |r - 1| > 1%: {np.mean(np.abs(r[ok] - 1) > .01):.1%}, "
          f"> 2.5%: {np.mean(np.abs(r[ok] - 1) > .025):.1%}, > 5%: {np.mean(np.abs(r[ok] - 1) > .05):.1%}, "
          f"no GT: {np.mean(~ok):.1%}; flow error median {np.median(fe[okf]):.3f} px, "
          f"> 0.5 px: {np.mean(fe[okf] > .5):.1%}, > 0.75 px: {np.mean(fe[okf] > .75):.1%}, > 1.5 px: {np.mean(fe[okf] > 1.5):.1%}")


if __name__ == "__main__":
    main()
