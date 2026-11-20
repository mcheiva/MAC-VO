"""
MAC-VO on the EIVA dives, scored with the split protocol (Scripts/Dataset/eval_eiva_trajectory.py), side by side.

    python -m Scripts.Eval.eiva_table runs/macvo/shipped runs/macvo/A runs/macvo/B

Each argument is a run_macvo.sh result dir (<odometry name>@eiva_<dive>/<time>/poses.npy; the newest run of a dive
counts). Prints ATE (m) / frame-to-frame (mm) per dive, the Vobster mean over the four plane_* dives and Welland, as
in docs/covhead_retrain_summary.md. The dive directory comes from Config/Sequence/EIVA/<dive>.yaml and the split
from the GT check in <gt root>/<dive>/index.json (check_eiva_gt.py's flow_error).

Mind what was trained on what: the production A and B saw every dive (other frames of it); held-out numbers need
the site folds (fold W: Vobster, fold V: Welland). MAC-VO varies about +-7% in ATE run to run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from Scripts.Dataset.eval_eiva_trajectory import evaluate, gt_positions, segments

ROOT = Path(__file__).resolve().parents[2]
VOBSTER = ("plane_mid", "plane_nose", "plane_tail1", "plane_tail2")
DIVES = VOBSTER + ("welland", "welland_rear")


def dive_dir(dive: str) -> Path:
    """The dive directory (pose_gt.txt, processed/left) of Config/Sequence/EIVA/<dive>.yaml."""
    for line in (ROOT / "Config/Sequence/EIVA" / f"{dive}.yaml").read_text().splitlines():
        if line.strip().startswith("root:"):
            return Path(line.split(":", 1)[1].strip()).parent
    raise ValueError(f"no root in Config/Sequence/EIVA/{dive}.yaml")


def reference(dive: str, gt_root: Path, break_px: float, min_segment: int):
    """Image names, GT track and evaluated segments of a dive."""
    seq = dive_dir(dive)
    names = sorted(p.name for p in (seq / "processed" / "left").glob("*.jpg"))
    gt = gt_positions(seq)
    track = np.stack([gt.get(n, np.full(3, np.nan)) for n in names])
    index = json.loads((gt_root / dive / "index.json").read_text())
    flow_error = {n: float(e) for n, e in zip(index["frames"], index["flow_error"]) if e is not None}
    return names, track, segments(names, gt, flow_error, break_px, min_segment, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+", help="run_macvo.sh result dirs")
    ap.add_argument("--gt-root", type=Path, default=Path("D:/Datasets/EIVA_GT_v2"))
    ap.add_argument("--break-px", type=float, default=3.0)
    ap.add_argument("--min-segment", type=int, default=10)
    args = ap.parse_args()

    table = {}                                                     # (dive, result dir) -> {ATE, f2f, ...}
    for dive in DIVES:
        runs = {r: sorted(r.glob(f"*@eiva_{dive}/*/poses.npy")) for r in args.results}
        if not any(runs.values()):
            continue
        names, track, segs = reference(dive, args.gt_root, args.break_px, args.min_segment)
        for r, poses in runs.items():
            if poses:
                est = np.load(poses[-1])[:, 1:4]
                if len(est) != len(names):
                    raise ValueError(f"{poses[-1]}: {len(est)} poses for {len(names)} images")
                table[dive, r] = evaluate(est, track, segs)

    def cell(rows: list[dict] | None) -> str:
        if not rows:
            return "-"
        return f"{np.mean([x['ATE'] for x in rows]):.3f} / {np.mean([x['f2f'] for x in rows]):.2f}"

    width = 18
    print(f"{'ATE m / f2f mm':<16}" + "".join(f"{r.name:>{width}}" for r in args.results))
    for dive in DIVES:
        if any((dive, r) in table for r in args.results):
            print(f"{dive:<16}" + "".join(f"{cell([table[dive, r]] if (dive, r) in table else None):>{width}}"
                                          for r in args.results))
    print(f"{'Vobster mean':<16}" + "".join(
        f"{cell([table[d, r] for d in VOBSTER] if all((d, r) in table for d in VOBSTER) else None):>{width}}"
        for r in args.results))


if __name__ == "__main__":
    main()
