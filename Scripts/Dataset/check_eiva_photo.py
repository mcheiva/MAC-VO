"""
Model-free per-frame check of rendered EIVA ground truth (Scripts/Dataset/make_eiva_gt.py) against its own images.

    python -m Scripts.Dataset.check_eiva_photo --seq D:/Datasets/EIVA/vobster_quay/plane_nose --gt <dir>/plane_nose

check_eiva_gt.py's FlowFormer statistics (the median disparity ratio and flow error on confident pixels) miss a pose
that is off in a way that keeps the scale: a small rotation makes the GT too near on one side of the image and too far
on the other. This check asks the images instead, on the GT pixels with texture (9 x 9 grey-level std > 3):

    photo_stereo    share whose 9 x 9 ZNCC against the right image, warped by the GT disparity, exceeds 0.5
    photo_temporal  the same against the next frame warped by the GT flow (pair i -> i + 1)

NaN where fewer than 500 pixels qualify. Both go into index.json (the previous one is kept as index.json.bak);
EIVA_NoIMU's `photo_rel` / `photo_floor` gate frames and pairs on them. A clean dive confirms 90-99% of its GT
(plane_nose); a frame with a wrong pose drops to 20-50%.
"""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

THRESHOLD, MIN_PIXELS, MIN_STD, WINDOW = 0.5, 500, 3.0, 9


def zncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = lambda x: cv2.blur(x, (WINDOW, WINDOW))
    ma, mb = m(a), m(b)
    return (m(a * b) - ma * mb) / np.sqrt(np.maximum((m(a * a) - ma ** 2) * (m(b * b) - mb ** 2), 1e-8))


def grey(path: Path, hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(path)
    return cv2.cvtColor(cv2.resize(image, hw[::-1], interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def check(seq: Path, gt: Path, names: list[str], i: int, fxb: float, hw: tuple[int, int]) -> tuple[float, float]:
    z = np.load(gt / f"{Path(names[i]).stem}.npz")
    depth = z["depth"].astype(np.float32)
    valid = depth > 0
    left = grey(seq / "processed" / "left" / names[i], hw)
    gy, gx = np.mgrid[:hw[0], :hw[1]].astype(np.float32)
    textured = valid & (np.sqrt(np.maximum(cv2.blur(left * left, (WINDOW, WINDOW)) - cv2.blur(left, (WINDOW, WINDOW)) ** 2, 0)) > MIN_STD)
    share = lambda ok, warped: float(np.mean(zncc(left, warped)[ok] > THRESHOLD)) if ok.sum() >= MIN_PIXELS else float("nan")
    disparity = np.where(valid, fxb / np.maximum(depth, 1e-6), 0).astype(np.float32)
    right = grey(seq / "processed" / "right" / names[i].replace("image_left", "image_right"), hw)
    stereo = share(textured & (gx - disparity > WINDOW), cv2.remap(right, gx - disparity, gy, cv2.INTER_LINEAR))
    temporal = float("nan")
    if "flow" in z and i + 1 < len(names):
        flow = np.nan_to_num(z["flow"].astype(np.float32))
        x, y = gx + flow[0], gy + flow[1]
        inside = (x > WINDOW) & (x < hw[1] - 1 - WINDOW) & (y > WINDOW) & (y < hw[0] - 1 - WINDOW)
        nxt = grey(seq / "processed" / "left" / names[i + 1], hw)
        temporal = share(textured & np.isfinite(z["flow"]).all(0) & inside, cv2.remap(nxt, x, y, cv2.INTER_LINEAR))
    return stereo, temporal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="make_eiva_gt.py output (index.json is updated in place)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    index_path = args.gt / "index.json"
    index = json.loads(index_path.read_text())
    names, hw = index["frames"], tuple(index["hw"])
    fxb = index["K"][0][0] * index["baseline"]
    with ThreadPoolExecutor(args.threads) as pool:
        res = list(pool.map(lambda i: check(args.seq, args.gt, names, i, fxb, hw), range(len(names))))
    stereo, temporal = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    shutil.copyfile(index_path, index_path.with_suffix(".json.bak"))
    index["photo_stereo"] = [None if np.isnan(s) else round(s, 4) for s in stereo]
    index["photo_temporal"] = [None if np.isnan(t) else round(t, 4) for t in temporal]
    tmp = index_path.with_suffix(".json.tmp")      # atomic: a training run may be reading index.json right now
    tmp.write_text(json.dumps(index, indent=1))
    tmp.replace(index_path)
    for key, s in (("stereo", stereo), ("temporal", temporal)):
        md = np.nanmedian(s)
        print(f"{args.gt.name} {key}: median {md:.1%}, p10 {np.nanquantile(s, .1):.1%}, below 0.75 x median {np.mean(s < 0.75 * md):.1%}, "
              f"below 50% {np.mean(s < 0.5):.1%}, not judged {np.mean(np.isnan(s)):.1%}")


if __name__ == "__main__":
    main()
