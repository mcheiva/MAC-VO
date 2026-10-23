"""
MAC-VO trajectories on an EIVA dive against its photogrammetry poses (pose_gt.txt, world -> camera).

    python -m Scripts.Dataset.eval_eiva_trajectory --seq D:/Datasets/EIVA/vobster_quay/plane_nose \
        --gt D:/Datasets/EIVA_GT_v2/plane_nose --poses <run A>/poses.npy <run B>/poses.npy

Frames are matched by image name (MAC-VO reads processed/left/*.jpg in sorted order; poses.npy rows are time_ns,
x, y, z, qx, qy, qz, qw). Camera positions are aligned with SE3 Umeyama, without scale: the stereo baseline fixes
it. Only positions are compared, so the result does not depend on the estimate's body-axis convention.

The GT trajectory has breaks: pairs whose GT check (check_eiva_gt.py's `flow_error`, the frozen FlowFormer
against the rendered flow) is far off, e.g. plane_nose 513 -> 514 at 318 px, and frames the reconstruction has no
pose for (plane_tail2). One alignment over the dive counts every frame after a break against every estimate, and a
few frames then dominate the result. So by default the dive is split at every break (a pair above --break-px, or a
frame without a pose) and each segment of at least --min-segment frames is aligned on its own. --no-split aligns
the whole dive once (optionally only its first --cut frames).

    ATE             RMS of the aligned camera positions over all evaluated frames (m)
    frame-to-frame  mean |error| of the aligned frame-to-frame displacement, over the pairs inside segments (mm)
    scale           estimated / GT path length inside segments
"""
import argparse
import json
from pathlib import Path

import numpy as np


def gt_positions(seq: Path) -> dict[str, np.ndarray]:
    """Image name -> camera position in the first GT camera's frame."""
    out, first = {}, None
    for line in (seq / "pose_gt.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        H = np.eye(4)                                       # world -> camera: X_c = R X_w + t
        H[:3, 3] = list(map(float, parts[1:4]))
        H[:3, :3] = np.array(list(map(float, parts[4:13]))).reshape(3, 3)
        first = H if first is None else first
        out[parts[0]] = (first @ np.linalg.inv(H))[:3, 3]
    return out


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R src + t - dst|^2 (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return R, md - R @ ms


def segments(names: list[str], gt: dict[str, np.ndarray], flow_error: dict[str, float] | None,
             break_px: float, min_segment: int, cut: int | None) -> list[list[int]]:
    """Image indices of the evaluated GT segments: split at breaks (flow_error given), or the posed prefix."""
    if flow_error is None:
        return [[i for i, n in enumerate(names) if n in gt and (cut is None or i < cut)]]
    out, cur = [], []
    for i, n in enumerate(names):
        if n not in gt:
            out, cur = out + [cur], []
            continue
        cur.append(i)
        if flow_error.get(n, 0.) > break_px:                 # the pair n -> next is a break
            out, cur = out + [cur], []
    return [s for s in out + [cur] if len(s) >= min_segment]


def evaluate(est: np.ndarray, gt_track: np.ndarray, segs: list[list[int]]) -> dict[str, float]:
    sq, steps, path_e, path_g = [], [], 0., 0.
    for keep in segs:
        e, g = est[keep], gt_track[keep]
        R, t = umeyama_se3(e, g)
        a = e @ R.T + t
        sq.append(np.linalg.norm(a - g, axis=1) ** 2)
        steps.append(np.linalg.norm(np.diff(a, axis=0) - np.diff(g, axis=0), axis=1))
        path_e += np.linalg.norm(np.diff(e, axis=0), axis=1).sum()
        path_g += np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
    return {"ATE": float(np.sqrt(np.concatenate(sq).mean())), "f2f": float(np.concatenate(steps).mean() * 1e3),
            "scale": path_e / path_g, "frames": sum(len(s) for s in segs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True, help="dive directory (processed/left, pose_gt.txt)")
    ap.add_argument("--gt", type=Path, help="make_eiva_gt.py output with check_eiva_gt.py's flow_error (for the split)")
    ap.add_argument("--poses", type=Path, nargs="+", required=True, help="MAC-VO poses.npy files")
    ap.add_argument("--no-split", action="store_true", help="one alignment over the whole dive")
    ap.add_argument("--cut", type=int, help="with --no-split: evaluate only the first CUT images")
    ap.add_argument("--break-px", type=float, default=3.0)
    ap.add_argument("--min-segment", type=int, default=10)
    args = ap.parse_args()

    names = sorted(p.name for p in (args.seq / "processed" / "left").glob("*.jpg"))
    gt = gt_positions(args.seq)
    gt_track = np.stack([gt.get(n, np.full(3, np.nan)) for n in names])
    flow_error = None
    if not args.no_split:
        if args.gt is None: ap.error("the split needs --gt (index.json with flow_error); or pass --no-split")
        index = json.loads((args.gt / "index.json").read_text())
        if "flow_error" not in index: ap.error(f"{args.gt / 'index.json'} has no flow_error: run check_eiva_gt.py")
        flow_error = {n: float(e) for n, e in zip(index["frames"], index["flow_error"]) if e is not None}
    segs = segments(names, gt, flow_error, args.break_px, args.min_segment, args.cut)
    print(f"{args.seq.name}: {len(names)} images, {len(segs)} segment(s): "
          + ", ".join(f"{s[0]}-{s[-1]}" for s in segs))

    # label: the run directory (<odometry name>@<sequence>/<time>/poses.npy)
    labels = ["/".join(p.parts[-3:-1]) if len(p.parts) >= 3 else str(p) for p in args.poses]
    width = max(len(label) for label in labels)
    print(f"{'run':{width}}  {'ATE (m)':>9}  {'f2f (mm)':>9}  {'scale':>7}")
    for label, path in zip(labels, args.poses):
        est = np.load(path)[:, 1:4]
        if len(est) != len(names):
            raise ValueError(f"{path}: {len(est)} poses for {len(names)} images")
        r = evaluate(est, gt_track, segs)
        print(f"{label:{width}}  {r['ATE']:9.3f}  {r['f2f']:9.2f}  {r['scale']:7.4f}")


if __name__ == "__main__":
    main()
