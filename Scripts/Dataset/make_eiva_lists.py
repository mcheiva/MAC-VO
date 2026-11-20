"""
Production sequence lists of solution B: every EIVA dive in DIVES, cross-fit halves (A: even blocks -> the matcher
fine-tune, B: odd blocks -> the head), frame gaps (half one-frame pairs, ~7% each of gaps 2, 3, 4, 6, 8, 12, 16) and
the image gate. The Welland site (Welland + welland_rear, 300-frame blocks) and Vobster (100-frame blocks) are weighted
about equally, EIVA ~25% of the pairs next to TartanAir.

    python -m Scripts.Dataset.make_eiva_lists [--out DIR]      (default: Config/Sequence/Training_Dataset)

Writes EIVA_All_TrainA_v2_water_gaps.yaml (EIVA_RefinedMatcher) and EIVA_All_TrainB_v2_gaps.yaml
(EIVA_RefinedHead). With DIVES unchanged it reproduces the committed lists byte for byte. A new dive
needs its GT first (make_eiva_gt -> check_eiva_gt -> check_eiva_photo, which write index.json); add it to DIVES with
the block length and repetitions that keep the sites balanced (the header's pair counts show the balance).
"""
import argparse
import json
from pathlib import Path

GAPS = [2, 3, 4, 6, 8, 12, 16]
DIVES = [  # root, gt, block length, one-frame reps per block, gap reps per offset
    ("D:/Datasets/EIVA/welland/Welland_test_first_dive", "D:/Datasets/EIVA_GT_v2/welland", 300, 7, 1),
    ("D:/Datasets/EIVA/welland/welland_rear", "D:/Datasets/EIVA_GT_v2/welland_rear", 300, 7, 1),
    ("D:/Datasets/EIVA/vobster_quay/plane_mid", "D:/Datasets/EIVA_GT_v2/plane_mid", 100, 28, 4),
    ("D:/Datasets/EIVA/vobster_quay/plane_nose", "D:/Datasets/EIVA_GT_v2/plane_nose", 100, 28, 4),
    ("D:/Datasets/EIVA/vobster_quay/plane_tail1", "D:/Datasets/EIVA_GT_v2/plane_tail1", 100, 28, 4),
    ("D:/Datasets/EIVA/vobster_quay/plane_tail2", "D:/Datasets/EIVA_GT_v2/plane_tail2", 100, 28, 4),
]


def entries(half: int, extra: list[str]) -> tuple[list[str], dict[str, int]]:
    out, pairs = [], {}
    for root, gt, block, r1, m in DIVES:
        n = len(json.loads((Path(gt) / "index.json").read_text())["frames"])
        dive = Path(gt).name
        for start in range(half * block, n, 2 * block):
            stop = min(start + block, n)
            for k in [1] + GAPS:
                for r in range(r1 if k == 1 else m * k):
                    off = 0 if k == 1 else r % k
                    items = len(range(off, stop - start, k))
                    if items < 2:
                        continue
                    args = [f"        root: {root}", f"        gt: {gt}", f"        frame_range: [{start}, {start + block}]"]
                    if k > 1:
                        args += [f"        frame_stride: {k}", f"        frame_offset: {off}"]
                    args += extra + ["        photo_rel: 0.75", "        photo_floor: 0.5"]
                    out += ["-   type: EIVA_NoIMU", f"    name: gap{k}_rep{r}_eiva_{dive}_{start}", "    args:"] + args
                    pairs[dive] = pairs.get(dive, 0) + items - 1
    return out, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("Config/Sequence/Training_Dataset"))
    args = ap.parse_args()
    for half, name, extra, what in ((0, "EIVA_All_TrainA_v2_water_gaps.yaml", ["        water_disparity: 0.25"],
                                     "matcher fine-tune: open water a far stereo target, no teacher"),
                                    (1, "EIVA_All_TrainB_v2_gaps.yaml", [], "covariance head (frames the half-A matcher fine-tune never saw)")):
        lines, pairs = entries(half, extra)
        head = (f"# Production, half {'AB'[half]} for cross-fitting: every EIVA dive's {'even' if half == 0 else 'odd'} blocks (Welland and welland_rear\n"
                f"# 300 frames, Vobster 100), v2 GT, frame gaps (half one-frame pairs, ~7% each of 2, 3, 4, 6, 8, 12, 16) and the image gate\n"
                f"# (photo_rel 0.75, photo_floor 0.5); Welland site and Vobster about equal; {what}.\n"
                f"# Pairs (before gating): " + ", ".join(f"{d} {p}" for d, p in pairs.items()) + "\n")
        (args.out / name).write_bytes((head + "\n".join(lines) + "\n").replace("\n", "\r\n").encode())
        print(name, sum(line.startswith("-   type") for line in lines), "entries,", sum(pairs.values()), "pairs:", pairs)


if __name__ == "__main__":
    main()
