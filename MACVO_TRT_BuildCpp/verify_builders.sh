#!/usr/bin/env bash
# GPU verification of the C++ and Python engine builders, one GPU only.
#
# Usage: verify_builders.sh [ONNX [CACHE [REF_PLAN]]]
#   (or the environment variables ONNX, CACHE, REF_PLAN, GPU, OUT, CPP, NO_WS)
#
# 1. Builds the ONNX (default profile 1/1/1 if dynamic, weight streaming on,
#    opt 5, avg timing 8, tiling 0, TF32 on) with both builders from separate
#    copies of the timing cache, with --error-on-timing-cache-miss.
# 2. Diffs the (layer Name, TacticName) pairs of the C++ plan, the Python plan
#    and, if REF_PLAN is given, a reference plan (TensorRT engine inspector,
#    JSON), compares the plan bytes with cmp, and diffs the two <plan>.json
#    sidecars field by field.
# 3. Deliberate miss: builds from an empty cache with
#    --error-on-timing-cache-miss and checks that the build aborts and writes
#    no plan, for both builders.
#
# ONNX      default Model/FlowFormerCovEIVAAllBV2NegGapsProd10-25-02-52-11/
#           MACVO_FrontendCov_EIVAProd_gaps_fast_b1_v2.onnx (static batch 1)
# CACHE     default: that directory's ..._fast_o5_b1_v2.timing.cache. It is
#           copied; the original is never written. A cache from another
#           ONNX / TensorRT version / GPU model makes the build fail the miss
#           check, which is the point.
# REF_PLAN  optional reference plan built from the same ONNX and cache
# GPU       default 1: the index (PCI bus order) of an idle GPU
# CPP       default MACVO_TRT_BuildCpp/build/Release/macvo_trt_build[.exe]
# OUT       default MACVO_TRT_BuildCpp/verify_out
# NO_WS     1 builds without weight streaming
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$REPO/Model/FlowFormerCovEIVAAllBV2NegGapsProd10-25-02-52-11"
ONNX="${1:-${ONNX:-$MODEL_DIR/MACVO_FrontendCov_EIVAProd_gaps_fast_b1_v2.onnx}}"
CACHE="${2:-${CACHE:-$MODEL_DIR/MACVO_FrontendCov_EIVAProd_gaps_fast_o5_b1_v2.timing.cache}}"
REF_PLAN="${3:-${REF_PLAN:-}}"
GPU="${GPU:-1}"
NO_WS="${NO_WS:-0}"
OUT="${OUT:-$REPO/MACVO_TRT_BuildCpp/verify_out}"
CPP="${CPP:-$REPO/MACVO_TRT_BuildCpp/build/Release/macvo_trt_build.exe}"
[ -x "$CPP" ] || CPP="${CPP%.exe}"

for f in "$ONNX" "$CACHE" "$CPP"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }
done
[ -z "$REF_PLAN" ] || [ -f "$REF_PLAN" ] || { echo "missing REF_PLAN: $REF_PLAN" >&2; exit 2; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

ARGS=(--onnx "$ONNX" --opt-level 5 --avg-timing 8 --tiling-level 0 --error-on-timing-cache-miss)
[ "$NO_WS" = "1" ] && ARGS+=(--no-weight-streaming)

mkdir -p "$OUT"
cp "$CACHE" "$OUT/cache_cpp.bin"
cp "$CACHE" "$OUT/cache_py.bin"
rm -f "$OUT"/cpp.plan* "$OUT"/py.plan* "$OUT"/miss_*

now() { date +%s; }

echo "== C++ build"
t0=$(now)
"$CPP" "${ARGS[@]}" --out "$OUT/cpp.plan" --timing-cache "$OUT/cache_cpp.bin" || echo "C++ build FAILED"
echo "C++ build seconds: $(( $(now) - t0 ))"

echo "== Python build"
t0=$(now)
# Plain `python`, not `python -I`: the TensorRT wheel is in the user site.
python "$REPO/Build.py" "${ARGS[@]}" --out "$OUT/py.plan" --timing-cache "$OUT/cache_py.bin" || echo "Python build FAILED"
echo "Python build seconds: $(( $(now) - t0 ))"

cat > "$OUT/layers.py" <<'PY'
"""Dump (layer Name, TacticName) pairs of a plan as JSON via the engine inspector."""
import json
import sys

import tensorrt as trt

rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
with open(sys.argv[1], "rb") as f:
    eng = rt.deserialize_cuda_engine(f.read())
info = json.loads(eng.create_engine_inspector().get_engine_information(
    trt.LayerInformationFormat.JSON))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump([[l["Name"], l.get("TacticName")] for l in info["Layers"]], f)
PY

[ -n "$REF_PLAN" ] && python "$OUT/layers.py" "$REF_PLAN" "$OUT/ref.layers.json"
for k in cpp py; do
    [ -f "$OUT/$k.plan" ] && python "$OUT/layers.py" "$OUT/$k.plan" "$OUT/$k.layers.json"
done

python - "$OUT" <<'PY'
import json
import os
import sys
from collections import Counter

out = sys.argv[1]
ok = True


def load(k):
    p = f"{out}/{k}.layers.json"
    return [tuple(x) for x in json.load(open(p, encoding="utf-8"))] if os.path.exists(p) else None


L = {k: load(k) for k in ("ref", "cpp", "py")}
for k, v in L.items():
    print(f"{k}: {'missing' if v is None else str(len(v)) + ' layers'}")
for a, b in (("cpp", "ref"), ("py", "ref"), ("cpp", "py")):
    if "ref" in (a, b) and L["ref"] is None:
        continue    # no REF_PLAN given
    if L[a] is None or L[b] is None:
        ok = False
        continue
    da, db = dict(L[a]), dict(L[b])
    pair_diff = [n for n in sorted(set(da) | set(db)) if da.get(n) != db.get(n)]
    ms = (Counter(t for _, t in L[a]) - Counter(t for _, t in L[b])
          + Counter(t for _, t in L[b]) - Counter(t for _, t in L[a]))
    print(f"{a} vs {b}: (layer, TacticName) differences={len(pair_diff)}, "
          f"TacticName multiset differences={sum(ms.values())}")
    for n in pair_diff[:20]:
        print(f"    {n}\n      {a}: {da.get(n)}\n      {b}: {db.get(n)}")
    ok = ok and not pair_diff

if os.path.exists(f"{out}/cpp.plan.json") and os.path.exists(f"{out}/py.plan.json"):
    sc = json.load(open(f"{out}/cpp.plan.json", encoding="utf-8"))
    sp = json.load(open(f"{out}/py.plan.json", encoding="utf-8"))
    # Windows path separators may differ between std::filesystem and os.path.
    for s in (sc, sp):
        for k in ("onnx_path", "timing_cache"):
            if s.get(k):
                s[k] = s[k].replace("\\", "/")
    # timing_cache differs by design (each build has its own copy).
    expected = {"timing_cache"}
    print("sidecar fields")
    for k in list(sc) + [k for k in sp if k not in sc]:
        x, y = sc.get(k, "<missing>"), sp.get(k, "<missing>")
        tag = "same" if x == y else ("expected-diff" if k in expected else "DIFF")
        ok = ok and tag != "DIFF"
        print(f"  {tag:14s}{k}: cpp={x!r} py={y!r}")
else:
    print("sidecar(s) missing")
    ok = False
print("RESULT:", "PASS" if ok else "FAIL")
PY

echo "== Plan bytes"
if [ -f "$OUT/cpp.plan" ] && [ -f "$OUT/py.plan" ]; then
    cmp -s "$OUT/cpp.plan" "$OUT/py.plan" && echo "cpp vs py: bytes identical" || echo "cpp vs py: bytes differ"
fi
if [ -n "$REF_PLAN" ]; then
    for k in cpp py; do
        [ -f "$OUT/$k.plan" ] && { cmp -s "$OUT/$k.plan" "$REF_PLAN" && echo "$k vs ref: bytes identical" || echo "$k vs ref: bytes differ"; }
    done
fi

echo "== Deliberate miss: empty cache, --error-on-timing-cache-miss, expect failure and no plan"
: > "$OUT/miss_cpp.cache"
: > "$OUT/miss_py.cache"
"$CPP" "${ARGS[@]}" --out "$OUT/miss_cpp.plan" --timing-cache "$OUT/miss_cpp.cache"
echo "C++ exit code: $?  plan exists: $([ -f "$OUT/miss_cpp.plan" ] && echo yes || echo no)"
python "$REPO/Build.py" "${ARGS[@]}" --out "$OUT/miss_py.plan" --timing-cache "$OUT/miss_py.cache"
echo "Python exit code: $?  plan exists: $([ -f "$OUT/miss_py.plan" ] && echo yes || echo no)"
