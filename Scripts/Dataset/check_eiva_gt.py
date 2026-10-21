"""
Per-frame consistency of rendered EIVA ground truth (Scripts/Dataset/make_eiva_gt.py) with the stereo images.

    python -m Scripts.Dataset.check_eiva_gt --seq D:/Datasets/EIVA/vobster_quay/plane_nose --gt <dir>/plane_nose

The photogrammetry reconstruction is locally wrong in places: a frame whose pose or surface is off renders a depth
that disagrees with its own rectified stereo pair by a few percent everywhere (Welland frame 800: 5.5%). The
frozen FlowFormer measures that: per frame, the median of predicted / rendered disparity (rendered = fx b / depth)
over the 25% of GT pixels the shipped head is most confident about. It is a per-frame scale statistic, robust to
the per-pixel matching failures the covariance head has to learn. The ratios are stored in index.json as
`disparity_ratio`; EIVA_NoIMU leaves frames beyond its `max_scale_error` unsupervised.
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
    ap.add_argument("--batch", type=int, default=4)
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

    ratios = []
    for b in range(0, len(names), args.batch):
        batch = names[b:b + args.batch]
        i1 = torch.from_numpy(np.stack([load_image(left / n, hw) for n in batch])).cuda()
        i2 = torch.from_numpy(np.stack([load_image(right / n.replace("image_left", "image_right"), hw) for n in batch])).cuda()
        with torch.inference_mode():
            flow, cov = model.inference(i1, i2)
        disparity, trace = -flow[:, 0].float().cpu().numpy(), cov[:, :2].sum(1).float().cpu().numpy()
        for k, name in enumerate(batch):
            depth = np.load(args.gt / f"{Path(name).stem}.npz")["depth"].astype(np.float32)
            valid = depth > 0
            if valid.sum() < 1000:
                ratios.append(float("nan"))
                continue
            conf = valid & (trace[k] <= np.quantile(trace[k][valid], 0.25))
            ratios.append(float(np.median(disparity[k][conf] * depth[conf] / fxb)))
        if b % 400 == 0:
            print(f"  {b}/{len(names)}: ratio {ratios[-1]:.4f}", flush=True)

    r = np.array(ratios)
    index["disparity_ratio"] = [round(v, 5) for v in ratios]
    index_path.write_text(json.dumps(index, indent=1))
    ok = np.isfinite(r)
    print(f"{args.seq.name}: median ratio {np.median(r[ok]):.4f}, |r - 1| > 1%: {np.mean(np.abs(r[ok] - 1) > .01):.1%}, "
          f"> 2.5%: {np.mean(np.abs(r[ok] - 1) > .025):.1%}, > 5%: {np.mean(np.abs(r[ok] - 1) > .05):.1%}, "
          f"no GT: {np.mean(~ok):.1%}")


if __name__ == "__main__":
    main()
