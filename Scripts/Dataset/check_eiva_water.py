"""
Open-water stereo check for a matcher on an EIVA dive: does it report the water as near?

    python -m Scripts.Dataset.check_eiva_water --seq D:/Datasets/EIVA/vobster_quay/plane_tail1 \
        --gt D:/Datasets/EIVA_GT_v2/plane_tail1 --frames 240 251 252 --ckpt Model/<run>/20000.pth [...]

A matcher fine-tuned on EIVA can read water it has not seen as a near surface (plane_tail1 with a Welland fine-tune:
~100 px disparity, 0.7 m, against ~15 px, 5 m, for the original network). MAC-VO then picks keypoints on it with a
confident wrong depth, and the trajectory breaks. The flow metrics cannot see this: the water has no GT. Run this on
a new site's dive before deploying a fine-tuned matcher.

Open water: pixels farther than --margin px from any GT pixel (EIVA_NoIMU's water_margin). Per frame and checkpoint:
the median disparity there and the depth it implies, the share nearer than --near m, and the disparity error on the
GT pixels. fp32, at the GT resolution, images downscaled with INTER_AREA as EIVA_NoIMU does; CPU unless --cuda.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Module"))
sys.path.insert(0, str(ROOT))
from Network.FlowFormer.configs.submission import get_cfg     # noqa: E402
from Network.FlowFormerCov import build_flowformer            # noqa: E402
from Scripts.Dataset.check_eiva_gt import load_image          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True, help="GT frame indices")
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=24.)
    ap.add_argument("--near", type=float, default=2.6, help="m; report the share of open water nearer than this")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    index = json.loads((args.gt / "index.json").read_text())
    names, hw = index["frames"], tuple(index["hw"])
    fxb = index["K"][0][0] * index["baseline"]
    device = "cuda" if args.cuda else "cpu"
    frames = []
    for f in args.frames:
        n = names[f]
        depth = np.load(args.gt / f"{Path(n).stem}.npz")["depth"].astype(np.float32)
        left = load_image(args.seq / "processed" / "left" / n, hw)
        right = load_image(args.seq / "processed" / "right" / n.replace("image_left", "image_right"), hw)
        frames.append((f, depth, torch.from_numpy(left[None]).to(device), torch.from_numpy(right[None]).to(device)))

    cfg = get_cfg()
    cfg.latentcostformer.decoder_depth = 12
    for ckpt in args.ckpt:
        model = build_flowformer(cfg, torch.float32, torch.float32)
        model.load_ddp_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.eval().to(device)
        for f, depth, left, right in frames:
            with torch.inference_mode():
                flow, _ = model.inference(left, right)
            disp = -flow[0, 0].float().cpu().numpy()
            has = depth > 0
            water = distance_transform_edt(~has) > args.margin
            err = np.abs(disp[has] - fxb / depth[has])
            md = float(np.median(disp[water])) if water.any() else float("nan")
            print(f"{ckpt.parent.name} #{f}: open water {water.mean():.0%} of the image, disparity median {md:.1f} px "
                  f"({fxb / max(md, 1e-3):.1f} m), nearer than {args.near} m {np.mean(disp[water] > fxb / args.near):.0%} | "
                  f"GT pixels: disparity error median {np.median(err):.2f} px, > 3 px {np.mean(err > 3):.1%}", flush=True)


if __name__ == "__main__":
    main()
