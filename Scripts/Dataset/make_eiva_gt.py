"""
Depth and optical-flow ground truth for an EIVA stereo sequence, rendered from its photogrammetry reconstruction
(3DF Zephyr: `pointcloud_gt.txt` and the per-image camera poses `pose_gt.txt`), at the 704 x 704 resolution the
frontend runs at (the 2816 x 2816 images downscaled 4x, K / 4).

    python -m Scripts.Dataset.make_eiva_gt --seq D:/Datasets/EIVA/vobster_quay/plane_nose --out <dir>/plane_nose

pose_gt.txt rows are `<left image> x y z r00 .. r22`, world -> camera (X_c = R X_w + t), as MACVO_TRT's
scripts/gt_to_tum.py reads them. Only frames with a pose are used; flow goes to the next such frame.

Per frame, <out>/<image stem>.npz holds
    depth  [H, W] float16, metres, 0 where no point survives
    flow   [2, H, W] float16, pixels to the next posed frame, NaN where depth is 0 (absent on the last frame)
and <out>/index.json the ordered frames with their next frame, the scaled K and the baseline.

A pixel keeps the nearest projected point (z-buffer). Points behind a nearer surface still show through holes
in it, so a pixel is dropped when it lies more than `--occlusion` (relative) behind the nearest depth within
`--window` pixels. Flow is the displacement of that very point (not of the pixel centre), so it is exact up to
the reconstruction's own error.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_calib(seq: Path, scale: float) -> tuple[np.ndarray, float, tuple[int, int]]:
    """resized_sensor.yaml is JSON for some sequences and YAML flow style for others."""
    txt = (seq / "processed" / "resized_sensor.yaml").read_text()
    try:
        c = json.loads(txt)
        K, baseline, res = c["intrinsics"], c["baseline"], c["resolution"]
    except json.JSONDecodeError:
        baseline = float(re.search(r"baseline:\s*([\d.]+)", txt).group(1))
        K = json.loads(re.search(r"intrinsics:\s*(\[\[.*?\]\])", txt).group(1))
        res = json.loads(re.search(r"resolution:\s*(\[.*?\])", txt).group(1))
    K = np.array(K, np.float64)
    K[:2] /= scale
    return K, float(baseline), (int(res[1] / scale), int(res[0] / scale))


def load_poses(seq: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for line in (seq / "pose_gt.txt").read_text().splitlines():
        p = line.split()
        if len(p) >= 13:
            poses[p[0]] = (np.array(p[4:13], np.float64).reshape(3, 3), np.array(p[1:4], np.float64))
    return poses


def load_cloud(seq: Path) -> np.ndarray:
    """[N, 3] float64 point positions (colours and normals are not needed)."""
    return pd.read_csv(seq / "pointcloud_gt.txt", sep=" ", comment="#", header=None, usecols=[0, 1, 2],
                       dtype=np.float64, engine="c").to_numpy()


class Renderer:
    def __init__(self, cloud: np.ndarray, K: np.ndarray, hw: tuple[int, int], device: torch.device):
        # points relative to their centroid in fp32; poses are re-centred in fp64 so nothing large meets fp32
        self.origin = cloud.mean(axis=0)
        self.P = torch.from_numpy(cloud - self.origin).float().to(device)
        self.K, self.hw, self.device = K, hw, device

    def camera(self, R: np.ndarray, t: np.ndarray) -> torch.Tensor:
        """Camera-frame points [N, 3] for X_c = R X_w + t."""
        Rt = torch.from_numpy(R.T.copy()).float().to(self.device)
        return self.P @ Rt + torch.from_numpy(R @ self.origin + t).float().to(self.device)

    def project(self, Xc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        z = Xc[:, 2]
        return torch.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], dim=1), z

    def zbuffer(self, Xc: torch.Tensor, window: int, occlusion: float) -> tuple[torch.Tensor, torch.Tensor]:
        """-> depth [H, W] (inf where empty) and the index of the point each pixel keeps (-1 where empty)."""
        H, W = self.hw
        uv, z = self.project(Xc)
        pix = torch.round(uv).long()               # integer pixel coordinates are pixel centres (OpenCV)
        ok = (z > 0.05) & (pix[:, 0] >= 0) & (pix[:, 0] < W) & (pix[:, 1] >= 0) & (pix[:, 1] < H)
        idx = torch.nonzero(ok).squeeze(1)
        lin = pix[idx, 1] * W + pix[idx, 0]
        depth = torch.full((H * W,), float("inf"), device=self.device).scatter_reduce(0, lin, z[idx], "amin")
        # the point that set each pixel's minimum (ties are the same depth, any of them will do)
        winner = torch.full((H * W,), -1, dtype=torch.long, device=self.device)
        hit = z[idx] == depth[lin]
        winner[lin[hit]] = idx[hit]
        depth = depth.view(H, W)
        near = -F.max_pool2d(-depth[None, None], window, stride=1, padding=window // 2)[0, 0]
        occluded = depth > near * (1 + occlusion)
        depth = torch.where(occluded, float("inf"), depth)
        winner = torch.where(occluded.view(-1), -1, winner)
        return depth, winner.view(H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True, help="sequence root (pose_gt.txt, pointcloud_gt.txt, processed/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=4.0, help="image downscale (2816 -> 704)")
    ap.add_argument("--window", type=int, default=7, help="occlusion test window (pixels, odd)")
    ap.add_argument("--occlusion", type=float, default=0.05, help="relative depth margin of the occlusion test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    K, baseline, hw = load_calib(args.seq, args.scale)
    poses = load_poses(args.seq)
    images = {p.name for p in (args.seq / "processed" / "left").glob("*.jpg")}
    names = sorted(n for n in poses if n in images)
    print(f"{args.seq.name}: {len(names)} posed frames of {len(images)}, {hw[1]}x{hw[0]}, fx {K[0, 0]:.2f}, "
          f"baseline {baseline:.4f} m", flush=True)
    cloud = load_cloud(args.seq)
    print(f"{len(cloud)} points", flush=True)
    r = Renderer(cloud, K, hw, torch.device(args.device))
    del cloud
    args.out.mkdir(parents=True, exist_ok=True)

    cover = []
    with torch.no_grad():
        for i, name in enumerate(names):
            Xc = r.camera(*poses[name])
            depth, winner = r.zbuffer(Xc, args.window, args.occlusion)
            valid = winner >= 0
            out = {"depth": torch.where(valid, depth, 0.).half().cpu().numpy()}
            if i + 1 < len(names):
                uv, _ = r.project(Xc[winner[valid]])
                uv2, _ = r.project(r.camera(*poses[names[i + 1]])[winner[valid]])
                flow = torch.full((2, *hw), float("nan"), device=r.device)
                flow[:, valid] = (uv2 - uv).T
                out["flow"] = flow.half().cpu().numpy()
            np.savez_compressed(args.out / f"{Path(name).stem}.npz", **out)
            cover.append(valid.float().mean().item())
            if i % 100 == 0:
                print(f"  {i}/{len(names)} {name}: {cover[-1]:.1%} of pixels", flush=True)
    (args.out / "index.json").write_text(json.dumps({
        "seq": str(args.seq), "scale": args.scale, "K": K.tolist(), "baseline": baseline, "hw": list(hw),
        "frames": names, "window": args.window, "occlusion": args.occlusion}, indent=1))
    print(f"done: mean coverage {np.mean(cover):.1%} -> {args.out}")


if __name__ == "__main__":
    main()
