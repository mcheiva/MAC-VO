"""
MAC-VO on TartanAir: mean ATE / RTE / ROE per sequence and averaged, side by side, over repeats.

    python -m Scripts.Eval.tartanair_table runs/macvo/shipped runs/macvo/A runs/macvo/B

Each argument is a run_macvo.sh result dir; repeats of it in <dir>_r2, <dir>_r3, ... are averaged (shown as
mean (min..max)). Scored by the upstream evaluator (Evaluation/EvalSeq.py). The summary's TartanAir numbers are the
Average row over the 8 held-out seasidetown sequences (Config/Sequence/TartanAirV1_Seaside).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from Evaluation.EvalSeq import EvaluateSequences


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+", help="run_macvo.sh result dirs")
    args = ap.parse_args()

    table = {}                                                      # dir -> (repeats, seq -> [(ate, rte, roe)])
    for root in args.results:
        repeats = [root] + sorted(root.parent.glob(f"{root.name}_r[0-9]*"))
        per_seq = defaultdict(list)
        for run in repeats:
            # runs with GT poses only (an EIVA run in the same dir has none; Scripts.Eval.eiva_table scores those)
            spaces = sorted(str(p.parent) for p in run.rglob("poses.npy") if (p.parent / "ref_poses.npy").exists())
            if not spaces:
                continue
            _, rows = EvaluateSequences(spaces)
            for r in rows:
                if r[0] != "Average" and r[1] is not None:
                    per_seq[r[0].split("@")[-1]].append((r[1], r[4], r[7]))
        table[root] = (len(repeats), per_seq)

    seqs = sorted({s for _, per_seq in table.values() for s in per_seq})
    width = 26
    print(f"\n{'':<13}" + "".join(f"{r.name:>{width}}" for r in args.results))
    print(f"{'repeats':<13}" + "".join(f"{table[r][0]:>{width}}" for r in args.results))
    for metric, i, fmt in (("ATE m", 0, "{:.3f}"), ("RTE", 1, "{:.4f}"), ("ROE", 2, "{:.4f}")):
        print(f"\n{metric}: mean over repeats (min..max)")
        for s in seqs:
            cells = []
            for r in args.results:
                v = [x[i] for x in table[r][1].get(s, [])]
                cells.append(f"{fmt.format(np.mean(v))} ({fmt.format(min(v))}..{fmt.format(max(v))})" if v else "-")
            print(f"  {s:<11}" + "".join(f"{c:>{width}}" for c in cells))
        cells = []
        for r in args.results:
            v = [[x[i] for x in table[r][1][s]] for s in seqs if s in table[r][1]]
            if len(v) != len(seqs) or len({len(x) for x in v}) != 1:
                cells.append("incomplete")
                continue
            per_repeat = np.mean(np.array(v), axis=0)                     # average over sequences, per repeat
            cells.append(f"{fmt.format(per_repeat.mean())} ({fmt.format(per_repeat.min())}..{fmt.format(per_repeat.max())})")
        print(f"  {'Average':<11}" + "".join(f"{c:>{width}}" for c in cells))


if __name__ == "__main__":
    main()
