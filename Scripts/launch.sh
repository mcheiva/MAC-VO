#!/usr/bin/env bash
# launch.sh <gpu> <run name> <command ...>: run one GPU job, pinned to one card, in the foreground, and record it in
# runs/ (the records next to the training logs in Model/*/):
#   runs/<name>_<YYYYmmdd-HHMM>.info      commit (+dirty), pid, gpu, start time, command; exit code when it ends
#   runs/<name>_<stamp>.log / .err.log    stdout / stderr
#
# One job per GPU, always. The two 24 GB cards page to host memory under WDDM when two jobs share one and then crawl
# for hours, and the trainer wraps the model in nn.DataParallel over every *visible* GPU (the configs' batch sizes
# assume one card), so every job must see exactly one GPU. A lock (runs/.gpu<N>.lock) refuses a second job on a busy
# card. For a queue, call launch.sh job after job in a script and start that script in the background, one per GPU:
#
#   Scripts/launch.sh 0 refined_matcher python -u -m Train.MatchingNet.train_flowformer \
#       --config Config/Train/EIVA_RefinedMatcher.yaml --training_mode flow
#   Scripts/launch.sh 1 eval_B Scripts/Eval/run_macvo.sh Config/Experiment/MACVO/EIVA/MACVO_EIVA_Fast_B.yaml \
#       runs/macvo/B Config/Sequence/EIVA/*.yaml
set -u
[ $# -ge 3 ] || { sed -n '2,17p' "$0"; exit 2; }
gpu=$1 name=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p runs

lock=runs/.gpu$gpu.lock
if ! mkdir "$lock" 2> /dev/null; then
  holder=$(cat "$lock/pid" 2> /dev/null)
  if [ -n "$holder" ] && kill -0 "$holder" 2> /dev/null; then
    echo "GPU $gpu is busy: $(cat "$lock/job" 2> /dev/null) (launch.sh pid $holder). One job per GPU." >&2
    exit 3
  fi
  echo "removing the stale lock of a launch.sh that no longer runs ($(cat "$lock/job" 2> /dev/null))" >&2
  rm -rf "$lock" && mkdir "$lock" || exit 3
fi
echo $$ > "$lock/pid"
trap 'rm -rf "$lock"' EXIT

stamp=$(date +%Y%m%d-%H%M)
base=runs/${name}_$stamp
echo "$base" > "$lock/job"
commit=$(git rev-parse --short HEAD)
git diff --quiet HEAD -- . ':!runs' 2> /dev/null || commit="$commit+dirty"
export PYTHONIOENCODING=utf-8
CUDA_VISIBLE_DEVICES=$gpu "$@" > "$base.log" 2> "$base.err.log" &
pid=$!
printf "commit %s\npid %s\ngpu %s\nstarted %s\n%s\n" "$commit" "$pid" "$gpu" "$(date +%Y-%m-%dT%H:%M:%S)" "$*" > "$base.info"
echo "$(date +%H:%M) $base started on GPU $gpu"
wait $pid
code=$?
echo "ended $(date +%Y-%m-%dT%H:%M:%S) exit $code" >> "$base.info"
echo "$(date +%H:%M) $base exited with $code"
exit $code
