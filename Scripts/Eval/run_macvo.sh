#!/usr/bin/env bash
# run_macvo.sh <odom config> <result dir> <data config ...>: MAC-VO on each sequence, one after another, on the GPU the
# caller left visible (start it through Scripts/launch.sh, which pins one). A sequence that already has poses in
# <result dir> is skipped, so a stopped run can be restarted. Per-sequence logs go to <result dir>/logs/.
#
#   EIVA (score: python -m Scripts.Eval.eiva_table <result dir> ...):
#     Scripts/launch.sh 0 eval_B_eiva Scripts/Eval/run_macvo.sh Config/Experiment/MACVO/EIVA/MACVO_EIVA_Fast_B.yaml \
#         runs/macvo/B Config/Sequence/EIVA/*.yaml
#   TartanAir seasidetown, the 8 sequences no training sees (score: python -m Scripts.Eval.tartanair_table ...):
#     Scripts/launch.sh 1 eval_B_seaside Scripts/Eval/run_macvo.sh Config/Experiment/MACVO/EIVA/MACVO_EIVA_Fast_B.yaml \
#         runs/macvo/B Config/Sequence/TartanAirV1_Seaside/*.yaml
#
# The MACVO_EIVA_Fast[_A|_B] configs are the evaluated setting for both: MACVO_Performant's pipeline with depth-aware
# keypoint selection, no dense mapping, fp16 encoder / bf16 decoder and min_flow_cov 0.25 px; their Preprocess resizes
# only the EIVA dives (GeneralStereo) to 704x704. MAC-VO is not bit-deterministic run to run (about +-7% ATE on one
# dive), so treat ~10% differences on one sequence as noise; repeat a run into <result dir>_r2, _r3 to average.
set -u
[ $# -ge 3 ] || { sed -n '2,17p' "$0"; exit 2; }
odom=$1 out=$2; shift 2
cd "$(dirname "$0")/../.."
mkdir -p "$out/logs"
export PYTHONIOENCODING=utf-8
failed=0
for data in "$@"; do
  name=$(sed -n 's/^name: *//p' "$data" | tr -d '\r')
  if ls "$out"/*@"$name"/*/poses.npy > /dev/null 2>&1; then echo "$name: done before, skipped"; continue; fi
  if python -u MACVO.py --odom "$odom" --data "$data" --resultRoot "$out" --noeval > "$out/logs/$name.log" 2>&1; then
    echo "$(date +%H:%M) $name done"
  else
    echo "$(date +%H:%M) $name FAILED (see $out/logs/$name.log)"; failed=1
  fi
done
exit $failed
