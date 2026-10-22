"""
Depth and optical-flow ground truth for an EIVA stereo sequence, rendered from its photogrammetry reconstruction
(3DF Zephyr: `pointcloud_gt.txt` and the per-image camera poses `pose_gt.txt`), at the 704 x 704 resolution the
frontend runs at (the 2816 x 2816 images downscaled 4x, K / 4).

    python -m Scripts.Dataset.make_eiva_gt --seq D:/Datasets/EIVA/vobster_quay/plane_nose --out <dir>/plane_nose

pose_gt.txt rows are `<left image> x y z r00 .. r22`, world -> camera (X_c = R X_w + t), as MACVO_TRT's
scripts/gt_to_tum.py reads them. Only frames with a pose are used; flow goes to the next such frame.

Per frame, <out>/<image stem>.npz holds
    depth   [H, W] float16, metres, 0 where no point survives
    flow    [2, H, W] float16, pixels to the next posed frame, NaN where depth is 0 (absent on the last frame)
    filled  [H, W] uint8, 1 where the depth comes from the hole fill rather than a point of its own
and <out>/index.json the ordered frames with their next frame, the scaled K, the baseline and the settings.

The cloud first loses its statistical outliers (`--outlier-std`): points whose mean distance to their
`--outlier-k` nearest neighbours is that many standard deviations above the mean. Those are reconstruction
floaters; they carry wrong depth, and each one in front of a surface would punch a hole through the occlusion
test below.

A pixel keeps the nearest projected point (z-buffer); its flow is the displacement of that very point, exact up
to the reconstruction's own error. Pixels no point lands on are filled from a (2 `--fill` + 1)^2 splat of the
cloud; their flow moves the pixel centre, back-projected at the filled depth, with the relative pose. Points
behind a nearer surface still show through holes in it, so a pixel is dropped when it lies more than
`--occlusion` (relative) behind the nearest depth within `--window` pixels.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


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


def remove_outliers(cloud: np.ndarray, k: int, n_std: float, chunk: int = 4_000_000) -> tuple[np.ndarray, int]:
    """Drop points whose mean distance to their k nearest neighbours exceeds the mean by n_std standard deviations."""
    tree = cKDTree(cloud)
    mean_dist = np.empty(len(cloud), np.float32)
    for s in range(0, len(cloud), chunk):       # chunked: the full distance matrix of 56M points does not fit in RAM
        d, _ = tree.query(cloud[s:s + chunk], k=k + 1, workers=-1)
        mean_dist[s:s + chunk] = d[:, 1:].mean(axis=1)
    keep = mean_dist < mean_dist.mean() + n_std * mean_dist.std()
    return cloud[keep], int((~keep).sum())


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

    def zbuffer(self, Xc: torch.Tensor, window: int, occlusion: float, fill: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> depth [H, W] (inf where empty), the index of the point each pixel keeps (-1 where filled or empty) and
        the mask of pixels filled from the splat."""
        H, W = self.hw
        uv, z = self.project(Xc)
        pix = torch.round(uv).long()               # integer pixel coordinates are pixel centres (OpenCV)

        def nearest(du: int, dv: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            u, v = pix[:, 0] + du, pix[:, 1] + dv
            idx = torch.nonzero((z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).squeeze(1)
            lin = v[idx] * W + u[idx]
            return idx, lin, torch.full((H * W,), float("inf"), device=self.device).scatter_reduce(0, lin, z[idx], "amin")

        idx, lin, depth = nearest(0, 0)
        # the point that set each pixel's minimum (ties are the same depth, any of them will do)
        winner = torch.full((H * W,), -1, dtype=torch.long, device=self.device)
        hit = z[idx] == depth[lin]
        winner[lin[hit]] = idx[hit]
        filled = torch.zeros(H * W, dtype=torch.bool, device=self.device)
        if fill > 0:
            splat = depth.clone()
            for du in range(-fill, fill + 1):
                for dv in range(-fill, fill + 1):
                    if du or dv:
                        splat = torch.minimum(splat, nearest(du, dv)[2])
            filled = torch.isinf(depth) & torch.isfinite(splat)
            depth = torch.where(filled, splat, depth)
        depth = depth.view(H, W)
        near = -F.max_pool2d(-depth[None, None], window, stride=1, padding=window // 2)[0, 0]
        occluded = (depth > near * (1 + occlusion)).view(-1)
        depth = torch.where(occluded.view(H, W), float("inf"), depth)
        winner = torch.where(occluded | filled, -1, winner)
        return depth, winner.view(H, W), (filled & ~occluded).view(H, W)

    def pixel_flow(self, depth: torch.Tensor, mask: torch.Tensor, pose1, pose2) -> torch.Tensor:
        """Flow [2, M] of the pixel centres in `mask`, back-projected at `depth` and moved by the relative pose."""
        (R1, t1), (R2, t2) = pose1, pose2
        R = R2 @ R1.T                                                     # camera 1 -> camera 2
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        v, u = torch.nonzero(mask, as_tuple=True)
        z = depth[mask]
        X1 = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], dim=1)
        X2 = X1 @ torch.from_numpy(R.T.copy()).float().to(self.device) + torch.from_numpy(t2 - R @ t1).float().to(self.device)
        uv2, _ = self.project(X2)
        return (uv2 - torch.stack([u, v], dim=1).float()).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True, help="sequence root (pose_gt.txt, pointcloud_gt.txt, processed/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=4.0, help="image downscale (2816 -> 704)")
    ap.add_argument("--window", type=int, default=7, help="occlusion test window (pixels, odd)")
    ap.add_argument("--occlusion", type=float, default=0.05, help="relative depth margin of the occlusion test")
    ap.add_argument("--outlier-std", type=float, default=2.0, help="statistical outlier removal threshold (0: off)")
    ap.add_argument("--outlier-k", type=int, default=8, help="neighbours of the outlier test")
    ap.add_argument("--fill", type=int, default=1, help="hole-fill splat radius in pixels (0: off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    K, baseline, hw = load_calib(args.seq, args.scale)
    poses = load_poses(args.seq)
    images = {p.name for p in (args.seq / "processed" / "left").glob("*.jpg")}
    names = sorted(n for n in poses if n in images)
    print(f"{args.seq.name}: {len(names)} posed frames of {len(images)}, {hw[1]}x{hw[0]}, fx {K[0, 0]:.2f}, "
          f"baseline {baseline:.4f} m", flush=True)
    cloud = load_cloud(args.seq)
    removed = 0
    if args.outlier_std > 0:
        cloud, removed = remove_outliers(cloud, args.outlier_k, args.outlier_std)
    print(f"{len(cloud) + removed} points, {removed} outliers removed ({removed / (len(cloud) + removed):.2%})", flush=True)
    r = Renderer(cloud, K, hw, torch.device(args.device))
    del cloud
    args.out.mkdir(parents=True, exist_ok=True)

    cover, cover_filled = [], []
    with torch.no_grad():
        for i, name in enumerate(names):
            Xc = r.camera(*poses[name])
            depth, winner, filled = r.zbuffer(Xc, args.window, args.occlusion, args.fill)
            own = winner >= 0
            valid = own | filled
            out = {"depth": torch.where(valid, depth, 0.).half().cpu().numpy(), "filled": filled.to(torch.uint8).cpu().numpy()}
            if i + 1 < len(names):
                uv, _ = r.project(Xc[winner[own]])
                uv2, _ = r.project(r.camera(*poses[names[i + 1]])[winner[own]])
                flow = torch.full((2, *hw), float("nan"), device=r.device)
                flow[:, own] = (uv2 - uv).T
                if filled.any():
                    flow[:, filled] = r.pixel_flow(depth, filled, poses[name], poses[names[i + 1]])
                out["flow"] = flow.half().cpu().numpy()
            np.savez_compressed(args.out / f"{Path(name).stem}.npz", **out)
            cover.append(valid.float().mean().item())
            cover_filled.append(filled.float().mean().item())
            if i % 100 == 0:
                print(f"  {i}/{len(names)} {name}: {cover[-1]:.1%} of pixels ({cover_filled[-1]:.1%} filled)", flush=True)
    (args.out / "index.json").write_text(json.dumps({
        "seq": str(args.seq), "scale": args.scale, "K": K.tolist(), "baseline": baseline, "hw": list(hw),
        "frames": names, "window": args.window, "occlusion": args.occlusion, "fill": args.fill,
        "outlier_std": args.outlier_std, "outlier_k": args.outlier_k, "outliers_removed": removed}, indent=1))
    print(f"done: mean coverage {np.mean(cover):.1%} ({np.mean(cover_filled):.1%} filled) -> {args.out}")


if __name__ == "__main__":
    main()
