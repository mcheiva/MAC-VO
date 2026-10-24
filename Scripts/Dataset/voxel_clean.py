"""
Voxel-hash cleaning of a photogrammetry point cloud before make_eiva_gt.py renders it - the nvblox idea: a sparse,
hashed voxel grid at high resolution, statistics per voxel, then filters on those statistics.

    python -m Scripts.Dataset.voxel_clean --seq D:/Datasets/EIVA/orec/OREC_CPS --out <dir>/pointcloud_gt.txt

Points are binned into voxels of `--voxel` metres, keyed by a 64-bit hash of their integer coordinates (21 bits per
axis). Per voxel: point count, centroid, mean normal (3DF Zephyr writes oriented normals). Filters, in order:

  support   voxels holding fewer than `--min-points` points go (floaters, fuzz)
  isolated  voxels with fewer than `--min-neighbours` occupied voxels among their 26 neighbours go
  layers    double surfaces (`--layer-gap min max`, metres; off by default): a voxel that finds another occupied voxel
            facing the same way (normal cosine > 0.8) along its own normal, between min and max away, goes when that
            layer's local support (points in its 3^3 neighbourhood) is more than `--layer-ratio` times its own
  photo     multi-view photo-consistency, as MVS fusion does it (`--photo-ncc`; off by default): every
            `--photo-every`-th posed frame, each voxel that frame sees - inside both images, its left patch textured,
            nothing of the cloud more than 10% in front of it within 7 x 7 pixels - scores the NCC of its 7 x 7 patch
            in the left image against the right image at its stereo disparity. A voxel at the wrong depth matches
            nowhere; it goes when it was scored in at least `--photo-views` frames with a mean NCC below the threshold.
            Voxels without that evidence (untextured, rarely seen) stay.

Each surviving voxel becomes one point at its centroid - positions averaged, the cloud shrinks to one point per
voxel. The output is written in pointcloud_gt.txt's column layout (x y z, then the voxel's mean colour and normal),
so make_eiva_gt.py reads it unchanged from a copy of the sequence folder. The input cloud is never modified.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BITS = 21
NEIGHBOURS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


def load_full(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pointcloud_gt.txt -> positions [N, 3] float64, colours [N, 3] uint8, normals [N, 3] float32."""
    a = pd.read_csv(path, sep=" ", comment="#", header=None, usecols=range(9), engine="c").to_numpy()
    return a[:, :3].astype(np.float64), a[:, 3:6].astype(np.uint8), a[:, 6:9].astype(np.float32)


def keys_of(ijk: np.ndarray) -> np.ndarray:
    """Hash of integer voxel coordinates; -1 for coordinates outside the 21-bit grid."""
    ok = ((ijk >= 0) & (ijk < 1 << BITS)).all(axis=1)
    return np.where(ok, (ijk[:, 0] << 2 * BITS) | (ijk[:, 1] << BITS) | ijk[:, 2], -1)


class VoxelGrid:
    def __init__(self, points: np.ndarray, voxel: float, normals: np.ndarray | None = None, colours: np.ndarray | None = None):
        self.voxel, self.origin = voxel, points.min(axis=0)
        p = points - self.origin
        keys = keys_of(np.floor(p / voxel).astype(np.int64))
        self.keys, inv, self.count = np.unique(keys, return_inverse=True, return_counts=True)
        self.ijk = np.stack([self.keys >> 2 * BITS, (self.keys >> BITS) & ((1 << BITS) - 1), self.keys & ((1 << BITS) - 1)], axis=1)
        mean = lambda x: np.stack([np.bincount(inv, x[:, d], len(self.keys)) for d in range(3)], axis=1) / self.count[:, None]
        self.centroid = mean(p) + self.origin
        self.normal = None if normals is None else mean(normals.astype(np.float64))
        if self.normal is not None:
            self.normal /= np.maximum(np.linalg.norm(self.normal, axis=1, keepdims=True), 1e-9)
        self.colour = None if colours is None else mean(colours.astype(np.float64))

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, ijk: np.ndarray) -> np.ndarray:
        """Index of the voxel at each coordinate, -1 where none."""
        k = keys_of(ijk)
        i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return np.where((self.keys[i] == k) & (k >= 0), i, -1)

    def subset(self, keep: np.ndarray) -> None:
        for name in ("keys", "count", "ijk", "centroid", "normal", "colour"):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name)[keep])

    def neighbourhood(self) -> tuple[np.ndarray, np.ndarray]:
        """Occupied voxels among the 26 neighbours, and points in the 3^3 neighbourhood (self included)."""
        n, support = np.zeros(len(self), np.int32), self.count.copy()
        for off in NEIGHBOURS:
            j = self.lookup(self.ijk + off)
            n += j >= 0
            support += np.where(j >= 0, self.count[j], 0)
        return n, support

    def layer_rivals(self, gap: tuple[float, float], support: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
        """Largest local support of a same-facing voxel along each voxel's normal, gap[0]..gap[1] away (0: none)."""
        steps = np.arange(gap[0], gap[1] + 1e-9, self.voxel / 2)
        rival = np.zeros(len(self), np.int64)
        for s in range(0, len(self), chunk):
            c, nrm = self.centroid[s:s + chunk] - self.origin, self.normal[s:s + chunk]
            for d in np.concatenate([steps, -steps]):
                j = self.lookup(np.floor((c + d * nrm) / self.voxel).astype(np.int64))
                same = (j >= 0) & ((self.normal[np.maximum(j, 0)] * nrm).sum(axis=1) > 0.8)
                rival[s:s + chunk] = np.maximum(rival[s:s + chunk], np.where(same, support[np.maximum(j, 0)], 0))
        return rival

    def photo_consistency(self, seq: Path, every: int = 5, offset: int = 0, scale: float = 4.0, patch: int = 7,
                          occlusion: float = 0.1, min_std: float = 3.0, chunk: int = 300_000) -> tuple[np.ndarray, np.ndarray]:
        """Mean stereo NCC of each voxel over the frames that see it (see the module docstring), and their number."""
        from Scripts.Dataset.make_eiva_gt import load_calib, load_poses
        K, baseline, (H, W) = load_calib(seq, scale)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        poses = load_poses(seq)
        left_dir, right_dir = seq / "processed" / "left", seq / "processed" / "right"
        names = sorted(n for n in poses if (left_dir / n).exists())[offset::every]
        P = torch.from_numpy(self.centroid - self.origin).float()
        r = patch // 2
        dy, dx = torch.meshgrid(torch.arange(-r, r + 1.), torch.arange(-r, r + 1.), indexing="ij")
        off = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)
        grey = lambda p: torch.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (W, H), interpolation=cv2.INTER_AREA),
                                                       cv2.COLOR_BGR2GRAY).astype(np.float32))[None, None]
        # pixel centres are integers (OpenCV); align_corners=True maps 0 and W - 1 to -1 and 1
        sample = lambda img, xy: F.grid_sample(img, (xy / torch.tensor([W - 1., H - 1.]) * 2 - 1)[None], align_corners=True)[0, 0]
        total, views = np.zeros(len(self)), np.zeros(len(self), np.int32)
        with torch.no_grad():
            for name in names:
                R, t = poses[name]
                Xc = P @ torch.from_numpy(R.T.copy()).float() + torch.from_numpy(R @ self.origin + t).float()
                z = Xc[:, 2].clamp_min(1e-3)
                u, v = fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy
                ur = u - fx * baseline / z
                idx = torch.nonzero((Xc[:, 2] > 0.05) & (ur >= r) & (u <= W - 1 - r) & (v >= r) & (v <= H - 1 - r)).squeeze(1)
                lin = torch.round(v[idx]).long() * W + torch.round(u[idx]).long()
                zb = torch.full((H * W,), float("inf")).scatter_reduce(0, lin, z[idx], "amin")
                near = -F.max_pool2d(-zb.view(1, 1, H, W), 7, stride=1, padding=3).view(-1)
                idx = idx[z[idx] <= near[lin] * (1 + occlusion)]
                L, Rimg = grey(left_dir / name), grey(right_dir / name.replace("image_left", "image_right"))
                for s in range(0, len(idx), chunk):
                    j = idx[s:s + chunk]
                    a = sample(L, torch.stack([u[j], v[j]], dim=1)[:, None] + off)
                    b = sample(Rimg, torch.stack([ur[j], v[j]], dim=1)[:, None] + off)
                    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
                    sa, sb = a.pow(2).mean(1).sqrt(), b.pow(2).mean(1).sqrt()
                    ncc = (a * b).mean(1) / (sa * sb).clamp_min(1e-6)
                    ok = (sa > min_std).numpy()
                    jj = j.numpy()[ok]
                    total[jj] += ncc.numpy()[ok]
                    views[jj] += 1
        return total / np.maximum(views, 1), views


def clean(points: np.ndarray, voxel: float, min_points: int = 1, min_neighbours: int = 0,
          layer_gap: tuple[float, float] | None = None, layer_ratio: float = 1.5,
          photo_seq: Path | None = None, photo_ncc: float = 0.4, photo_views: int = 3, photo_every: int = 5,
          photo_offset: int = 0, normals: np.ndarray | None = None, colours: np.ndarray | None = None, log=print) -> VoxelGrid:
    t0 = time.time()
    g = VoxelGrid(points, voxel, normals, colours)
    log(f"  {len(points)} points -> {len(g)} voxels of {voxel * 1e3:.1f} mm ({len(points) / len(g):.1f} points each)")
    if min_points > 1:
        n0 = len(g)
        g.subset(g.count >= min_points)
        log(f"  support: {n0 - len(g)} voxels with < {min_points} points dropped")
    if min_neighbours > 0:
        n0 = len(g)
        g.subset(g.neighbourhood()[0] >= min_neighbours)
        log(f"  isolated: {n0 - len(g)} voxels with < {min_neighbours} occupied neighbours dropped")
    if layer_gap is not None:
        if g.normal is None:
            raise ValueError("the layer filter needs normals")
        n0 = len(g)
        support = g.neighbourhood()[1]
        g.subset(g.layer_rivals(layer_gap, support) <= layer_ratio * support)
        log(f"  layers: {n0 - len(g)} voxels with a {layer_ratio}x better-supported surface "
            f"{layer_gap[0] * 1e3:.0f}-{layer_gap[1] * 1e3:.0f} mm along their normal dropped")
    if photo_seq is not None:
        n0 = len(g)
        score, views = g.photo_consistency(photo_seq, photo_every, photo_offset)
        judged = views >= photo_views
        g.subset(~judged | (score >= photo_ncc))
        log(f"  photo: {judged.mean():.0%} of voxels seen textured in >= {photo_views} frames (every {photo_every}th), "
            f"mean NCC md {np.median(score[judged]):.2f}; {n0 - len(g)} with NCC < {photo_ncc} dropped")
    log(f"  -> {len(g)} points ({time.time() - t0:.0f} s)")
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True, help="sequence folder holding pointcloud_gt.txt")
    ap.add_argument("--out", type=Path, required=True, help="cleaned cloud (.txt in pointcloud_gt.txt layout)")
    ap.add_argument("--voxel", type=float, default=0.002, help="voxel size, metres")
    ap.add_argument("--min-points", type=int, default=1)
    ap.add_argument("--min-neighbours", type=int, default=0)
    ap.add_argument("--layer-gap", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--layer-ratio", type=float, default=1.5)
    ap.add_argument("--photo-ncc", type=float, default=None, help="photo-consistency threshold (off when not given)")
    ap.add_argument("--photo-views", type=int, default=3)
    ap.add_argument("--photo-every", type=int, default=5)
    args = ap.parse_args()
    if args.out.resolve() == (args.seq / "pointcloud_gt.txt").resolve():
        raise SystemExit("refusing to overwrite the input cloud")
    points, colours, normals = load_full(args.seq / "pointcloud_gt.txt")
    g = clean(points, args.voxel, args.min_points, args.min_neighbours, args.layer_gap, args.layer_ratio,
              args.seq if args.photo_ncc is not None else None, args.photo_ncc or 0., args.photo_views, args.photo_every,
              normals=normals, colours=colours)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# voxel_clean of {args.seq / 'pointcloud_gt.txt'}: voxel {args.voxel} m, min points {args.min_points}, "
                f"min neighbours {args.min_neighbours}, layer gap {args.layer_gap}, layer ratio {args.layer_ratio}, "
                f"photo NCC {args.photo_ncc} in >= {args.photo_views} views every {args.photo_every}th frame\n")
        f.write("# PositionX PositionY PositionZ ColorR ColorG ColorB NormalX NormalY NormalZ\n")
        np.savetxt(f, np.hstack([g.centroid, np.round(g.colour), g.normal]), fmt="%.9g %.9g %.9g %d %d %d %.6g %.6g %.6g")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
