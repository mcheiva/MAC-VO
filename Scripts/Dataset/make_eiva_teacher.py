"""
Teacher targets for fine-tuning the matcher on EIVA: the frozen original network's predictions on every GT frame.

    python -m Scripts.Dataset.make_eiva_teacher --seq D:/Datasets/EIVA/welland/Welland_test_first_dive \
        --gt D:/Datasets/EIVA_GT_v2/welland --out D:/Datasets/EIVA_teacher/welland

The photogrammetry has no surface in open water and haze, so those pixels have no flow or depth target (EIVA_NoIMU
leaves them NaN / 0). Fine-tuned on the rest, the stereo matcher drifts there: on plane_tail1 the open water's
disparity went from ~15 px (~5 m, consistent with the camera motion) to 40-90 px (~1 m), and the keypoints MAC-VO
picked there broke the trajectory. EIVA_NoIMU's `teacher` option fills the pixels without a target with these
predictions, so the fine-tune keeps the original behaviour where there is nothing to learn from.

Per GT frame (index.json order, at the GT resolution, images downscaled with INTER_AREA as EIVA_NoIMU does), out/<frame
stem>.npz holds `disparity` [H, W] (left -> right, positive) and, except for the last frame, `flow` [2, H, W] to the
next indexed frame; both float16.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Module"))
sys.path.insert(0, str(ROOT))
from Network.FlowFormer.configs.submission import get_cfg     # noqa: E402
from Network.FlowFormerCov import build_flowformer            # noqa: E402
from Scripts.Dataset.check_eiva_gt import load_image          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="make_eiva_gt.py output (for the frame list and resolution)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "Model" / "MACVO_FrontendCov.pth", help="the teacher (original matcher)")
    ap.add_argument("--batch", type=int, default=2, help="frames per forward pass (each gives a stereo and a temporal pair)")
    args = ap.parse_args()

    index = json.loads((args.gt / "index.json").read_text())
    names, hw = index["frames"], tuple(index["hw"])
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = get_cfg()
    cfg.latentcostformer.decoder_depth = 12
    model = build_flowformer(cfg, torch.float32, torch.float32)
    model.load_ddp_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model = model.eval().cuda()
    left, right = args.seq / "processed" / "left", args.seq / "processed" / "right"

    for b in range(0, len(names), args.batch):
        batch = names[b:b + args.batch]
        pairs = [(left / n, right / n.replace("image_left", "image_right")) for n in batch]       # stereo
        pairs += [(left / n, left / names[b + k + 1]) for k, n in enumerate(batch) if b + k + 1 < len(names)]   # temporal
        i1 = torch.from_numpy(np.stack([load_image(p, hw) for p, _ in pairs])).cuda()
        i2 = torch.from_numpy(np.stack([load_image(p, hw) for _, p in pairs])).cuda()
        with torch.inference_mode():
            flow, _ = model.inference(i1, i2)
        flow = flow.float().cpu().numpy()
        for k, name in enumerate(batch):
            out = {"disparity": (-flow[k, 0]).astype(np.float16)}
            if len(batch) + k < len(flow):
                out["flow"] = flow[len(batch) + k].astype(np.float16)
            np.savez_compressed(args.out / f"{Path(name).stem}.npz", **out)
        if b % 400 == 0:
            print(f"  {b}/{len(names)}", flush=True)
    (args.out / "teacher.json").write_text(json.dumps({"ckpt": str(args.ckpt), "gt": str(args.gt), "hw": hw}, indent=1))
    print(f"{args.seq.name}: {len(names)} frames -> {args.out}")


if __name__ == "__main__":
    main()
