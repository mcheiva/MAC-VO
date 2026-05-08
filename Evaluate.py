"""Evaluation harness for MACVO_FrontendCov TensorRT plans.

Pipelines the polygraphy-based numerical checks plus an optional plane-nose
trajectory run, and writes a leaderboard row so multiple plans accumulate
in `runs/leaderboard.md`. Pair with Build.py to validate any new build.

Stages (individually skippable):
  1. ONNX validity      - polygraphy run --validate (NaN/Inf scan).
  2. ORT-CPU reference  - cached per (onnx, sample-pair) hash.
  3. TRT vs ORT parity  - mean abs / max abs diff per output.
  4. Trajectory (opt)   - macvo.exe replay + evo APE/RPE vs ground truth.
  5. Leaderboard append - one markdown row to runs/leaderboard.md.

Defaults resolution (first hit wins, all overridable on the command line):
  1. CLI args
  2. Environment vars: MACVO_EVAL_SAMPLE_LEFT, MACVO_EVAL_SAMPLE_RIGHT,
                       MACVO_EVAL_MACVO_EXE, MACVO_EVAL_DATASET
  3. Config file: Evaluate.config.json next to this script (override with
                  --config or env MACVO_EVAL_CONFIG)
  4. Auto-discovery in common locations.

Label is auto-derived from the plan filename (strip MACVO_FrontendCov_ prefix
and .plan suffix) if --label is not given.

Batch mode: --plan accepts multiple paths or a glob; each plan gets its own
trajectory + leaderboard row.

Examples:
  python Evaluate.py --plan MACVO_FrontendCov_strongly_fast.plan      # numerical only
  python Evaluate.py --plan "MACVO_FrontendCov_*.plan" --macvo-exe ... --dataset ...
  echo '{"sample_left": "left_t1.npy", "sample_right": "right_t1.npy", \
         "macvo_exe": "../MACVO_TRT/build/bin/Release/macvo.exe", \
         "dataset": "C:/data/plane_nose"}' > Evaluate.config.json
  python Evaluate.py --plan X.plan   # all paths auto-resolved
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

CONFIG_NAME = "Evaluate.config.json"
ENV_PREFIX = "MACVO_EVAL_"
DEFAULT_INPUT_SHAPE = (1, 3, 704, 704)

AUTO_SEARCH_SAMPLE_LEFT = [
    "left_t1.npy",
    "samples/left_t1.npy",
    "../MAC-VO/left_t1.npy",
]
AUTO_SEARCH_SAMPLE_RIGHT = [
    "right_t1.npy",
    "samples/right_t1.npy",
    "../MAC-VO/right_t1.npy",
]
AUTO_SEARCH_MACVO_EXE = [
    "../MACVO_TRT/build/bin/Release/macvo.exe",
    "../MACVO_TRT/build/bin/RelWithDebInfo/macvo.exe",
    "../MACVO_TRT/build/bin/Debug/macvo.exe",
]
AUTO_SEARCH_DATASET = [
    "C:/src/git/AI/Photogrammetry/TestingUtils/performancedata/plane_nose",
]


def first_existing(candidates: list[str], cwd: Path) -> Path | None:
    for c in candidates:
        p = (cwd / c).resolve() if not Path(c).is_absolute() else Path(c)
        if p.exists():
            return p
    return None


def load_config(explicit: Path | None) -> dict:
    if explicit is not None:
        return json.loads(Path(explicit).read_text())
    env_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if env_path and Path(env_path).exists():
        return json.loads(Path(env_path).read_text())
    next_to_script = THIS_DIR / CONFIG_NAME
    if next_to_script.exists():
        return json.loads(next_to_script.read_text())
    return {}


def resolve_default(cli_value, config: dict, key: str, env_suffix: str,
                    auto_candidates: list[str]) -> Path | None:
    if cli_value is not None:
        return Path(cli_value)
    env_v = os.environ.get(f"{ENV_PREFIX}{env_suffix}")
    if env_v:
        return Path(env_v)
    cfg_v = config.get(key)
    if cfg_v:
        return Path(cfg_v)
    return first_existing(auto_candidates, THIS_DIR)


def label_from_plan(plan_path: Path) -> str:
    stem = plan_path.stem
    for prefix in ("MACVO_FrontendCov_", "MACVO_FrontendCov"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):].lstrip("_")
            break
    return stem or plan_path.stem


def expand_plan_arg(plan_args: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in plan_args:
        if any(c in raw for c in "*?["):
            out.extend(Path(p) for p in sorted(glob.glob(raw)))
        else:
            out.append(Path(raw))
    deduped = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            deduped.append(p)
            seen.add(rp)
    return deduped


def hash_sample(sample_left: Path, sample_right: Path, onnx: Path) -> str:
    h = hashlib.sha256()
    for p in (sample_left, sample_right, onnx):
        h.update(p.resolve().as_posix().encode())
        h.update(b"\0")
        h.update(str(p.stat().st_mtime_ns).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def run_cmd(cmd: list[str], desc: str) -> str:
    print(f"\n>>> {desc}", flush=True)
    print(f"    $ {' '.join(map(str, cmd))}", flush=True)
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout, flush=True)
        print(p.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"{desc} failed (exit {p.returncode})")
    return p.stdout


def make_loader_script(left_npy: Path, right_npy: Path, out_path: Path) -> None:
    out_path.write_text(
        f"""# Auto-generated by Evaluate.py
import numpy as np


def load_data():
    left = np.load(r"{left_npy}").astype(np.float32)
    right = np.load(r"{right_npy}").astype(np.float32)
    if left.ndim == 3:
        left = left[None, ...]
        right = right[None, ...]
    if left.max() <= 1.5:
        left = left * 255.0
        right = right * 255.0
    yield {{
        "image_1": np.ascontiguousarray(left),
        "image_2": np.ascontiguousarray(right),
    }}
"""
    )


def stage_validate_onnx(onnx_path: Path, loader_path: Path) -> dict:
    out = run_cmd(
        ["polygraphy", "run", onnx_path,
         "--onnxrt", "--providers", "cpu",
         "--data-loader-script", loader_path,
         "--validate"],
        desc="Stage 1: ONNX validity (NaN/Inf scan)",
    )
    return {"validate_pass": "PASSED" in out}


def stage_ort_reference(onnx_path: Path, loader_path: Path,
                        ort_outs_path: Path) -> dict:
    if ort_outs_path.exists():
        print(f"\n>>> Stage 2: ORT reference (cached at {ort_outs_path})",
              flush=True)
        return {"ort_cached": True}
    run_cmd(
        ["polygraphy", "run", onnx_path,
         "--onnxrt", "--providers", "cpu",
         "--data-loader-script", loader_path,
         "--save-outputs", ort_outs_path],
        desc="Stage 2: ORT-CPU reference (saving outputs)",
    )
    return {"ort_cached": False}


def stage_trt_parity(plan_path: Path, loader_path: Path, ort_outs_path: Path,
                     input_shape: tuple) -> dict:
    """Run polygraphy run with tolerances so loose nothing fails; parse the
    `Minimum Required Tolerance` lines for actual mean abs diff. We treat
    the diff numbers as data, not pass/fail signal -- fp16-vs-fp32 will
    always exceed any tight default tolerance."""
    shape_str = f"image_1:[{','.join(map(str, input_shape))}]"
    shape_str2 = f"image_2:[{','.join(map(str, input_shape))}]"
    cmd = ["polygraphy", "run", plan_path,
           "--model-type=engine", "--trt",
           "--data-loader-script", loader_path,
           "--load-outputs", ort_outs_path,
           "--check-error-stat", "mean",
           "--rtol", "1e9", "--atol", "1e9",
           "--input-shapes", shape_str, shape_str2]
    print(f"\n>>> Stage 3: TRT vs ORT tensor parity", flush=True)
    print(f"    $ {' '.join(map(str, cmd))}", flush=True)
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    # Don't treat non-zero exit as fatal: parity-fail is normal for fp16.
    out = p.stdout + p.stderr
    metrics = {"parity_outputs": []}
    cur_output = None
    for line in out.splitlines():
        if "Comparing Output:" in line:
            try:
                cur_output = line.split("Comparing Output:")[1].split("'")[1]
            except IndexError:
                cur_output = None
        elif cur_output and "Minimum Required Tolerance" in line:
            try:
                tail = line.split("[abs=")[1].split("]")[0]
                metrics["parity_outputs"].append(
                    {"output": cur_output, "mean_abs_diff": float(tail)})
            except (IndexError, ValueError):
                pass
            cur_output = None
    return metrics


def stage_trajectory(plan_path: Path, label: str, macvo_exe: Path,
                     dataset_dir: Path, runs_dir: Path) -> dict:
    runs_dir.mkdir(parents=True, exist_ok=True)
    tum_out = runs_dir / f"plane_nose_{label}_tum.txt"
    tum_gt_out = runs_dir / f"plane_nose_{label}_gt_tum.txt"
    run_cmd(
        [macvo_exe, "--no-viz",
         "--engine", plan_path,
         "--tum-output", tum_out,
         "--tum-gt-output", tum_gt_out,
         "dataset", "--dataset", dataset_dir],
        desc=f"Stage 4: macvo.exe trajectory ({label})",
    )
    return {"tum_run": str(tum_out), "tum_gt": str(tum_gt_out)}


def stage_evo_compare(tum_run: Path, tum_gt: Path) -> dict:
    """Compute APE + RPE via evo. Same metric as scripts/compare_trajectories.py."""
    import copy
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.tools import file_interface

    ref = file_interface.read_tum_trajectory_file(str(tum_gt))
    est = file_interface.read_tum_trajectory_file(str(tum_run))
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=0.05)
    est_aligned = copy.deepcopy(est_s)
    est_aligned.align(ref_s, correct_scale=False, correct_only_scale=False)

    ape = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
    ape.process_data((ref_s, est_aligned))
    rpe = evo_metrics.RPE(evo_metrics.PoseRelation.translation_part,
                          delta=1.0, delta_unit=evo_metrics.Unit.frames,
                          rel_delta_tol=0.1, all_pairs=False)
    rpe.process_data((ref_s, est_aligned))
    a = ape.get_all_statistics()
    r = rpe.get_all_statistics()
    return {
        "APE_rmse": a["rmse"], "APE_mean": a["mean"], "APE_max": a["max"],
        "RPE_rmse": r["rmse"], "RPE_mean": r["mean"],
    }


def stage_leaderboard_append(runs_dir: Path, label: str, plan_path: Path,
                             metrics: dict) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = runs_dir / "leaderboard.md"
    if not leaderboard.exists():
        leaderboard.write_text(
            "| label | plan size MB | APE_rmse | APE_mean | APE_max | "
            "RPE_rmse | RPE_mean |\n"
            "|-------|-------------:|---------:|---------:|--------:|"
            "---------:|---------:|\n"
        )
    size_mb = plan_path.stat().st_size / (1024 * 1024)
    nan = float('nan')
    row = (
        f"| {label} | {size_mb:.1f} "
        f"| {metrics.get('APE_rmse', nan):.4f} "
        f"| {metrics.get('APE_mean', nan):.4f} "
        f"| {metrics.get('APE_max', nan):.4f} "
        f"| {metrics.get('RPE_rmse', nan):.4f} "
        f"| {metrics.get('RPE_mean', nan):.4f} |\n"
    )
    with leaderboard.open("a") as f:
        f.write(row)
    print(f"\n>>> Appended leaderboard row to {leaderboard}", flush=True)
    print(row.rstrip(), flush=True)


def evaluate_one(plan_path: Path, label: str, onnx_path: Path,
                 sample_left: Path | None, sample_right: Path | None,
                 macvo_exe: Path | None, dataset_dir: Path | None,
                 runs_dir: Path, input_shape: tuple,
                 skip: set[int]) -> dict:
    print(f"\n========================================")
    print(f" Plan : {plan_path}")
    print(f" Label: {label}")
    print(f"========================================")
    metrics: dict = {"label": label, "plan": str(plan_path)}

    have_samples = sample_left is not None and sample_right is not None
    have_runtime = macvo_exe is not None and dataset_dir is not None

    do_numerical = have_samples and ({1, 2, 3} - skip)
    if have_samples:
        runs_dir.mkdir(parents=True, exist_ok=True)
        loader_path = runs_dir / "_pg_loader_auto.py"
        make_loader_script(sample_left, sample_right, loader_path)

        cache_key = hash_sample(sample_left, sample_right, onnx_path)
        ort_outs_path = onnx_path.with_suffix(f".ort_outs_{cache_key}.json")
        if 1 not in skip:
            metrics.update(stage_validate_onnx(onnx_path, loader_path))
        if 2 not in skip:
            metrics.update(stage_ort_reference(onnx_path, loader_path, ort_outs_path))
        if 3 not in skip:
            metrics.update(stage_trt_parity(plan_path, loader_path,
                                            ort_outs_path, input_shape))
    elif do_numerical:
        print("WARN: stages 1-3 skipped (no sample-left/sample-right resolved)",
              file=sys.stderr)

    if have_runtime and 4 not in skip:
        traj = stage_trajectory(plan_path, label, macvo_exe,
                                dataset_dir, runs_dir)
        metrics.update(stage_evo_compare(Path(traj["tum_run"]),
                                         Path(traj["tum_gt"])))
        if 5 not in skip:
            stage_leaderboard_append(runs_dir, label, plan_path, metrics)
    elif 4 not in skip and (macvo_exe is None or dataset_dir is None):
        print("Stage 4 skipped (need both --macvo-exe and --dataset)",
              flush=True)

    print(f"\n>>> Metrics for {label}:")
    print(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                      for k, v in metrics.items()}, indent=2))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(
        description="Numerical + trajectory evaluation for MACVO TRT plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--plan", required=True, nargs="+",
                   help="One or more plan paths (globs supported, e.g. '*.plan')")
    p.add_argument("--label",
                   help="Override auto-derived label (single-plan mode only)")
    p.add_argument("--onnx", type=Path,
                   default=THIS_DIR / "MACVO_FrontendCov.onnx",
                   help="Source ONNX (for ORT reference)")
    p.add_argument("--config", type=Path,
                   help="JSON config with sample/macvo/dataset paths")
    p.add_argument("--sample-left", type=Path)
    p.add_argument("--sample-right", type=Path)
    p.add_argument("--macvo-exe", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--runs-dir", type=Path, default=THIS_DIR / "runs")
    p.add_argument("--input-shape", default="1,3,704,704")
    p.add_argument("--skip", type=int, nargs="*", default=[],
                   choices=[1, 2, 3, 4, 5],
                   help="Skip stages by number, e.g. --skip 1 2 3 to run "
                        "trajectory only")
    args = p.parse_args()

    config = load_config(args.config)
    sample_left = resolve_default(args.sample_left, config, "sample_left",
                                  "SAMPLE_LEFT", AUTO_SEARCH_SAMPLE_LEFT)
    sample_right = resolve_default(args.sample_right, config, "sample_right",
                                   "SAMPLE_RIGHT", AUTO_SEARCH_SAMPLE_RIGHT)
    macvo_exe = resolve_default(args.macvo_exe, config, "macvo_exe",
                                "MACVO_EXE", AUTO_SEARCH_MACVO_EXE)
    dataset = resolve_default(args.dataset, config, "dataset",
                              "DATASET", AUTO_SEARCH_DATASET)

    print("Resolved paths:")
    print(f"  sample_left  = {sample_left}")
    print(f"  sample_right = {sample_right}")
    print(f"  macvo_exe    = {macvo_exe}")
    print(f"  dataset      = {dataset}")

    plans = expand_plan_arg(args.plan)
    if not plans:
        print(f"ERROR: --plan {args.plan} matched no files", file=sys.stderr)
        return 2
    if args.label and len(plans) > 1:
        print("ERROR: --label only valid with a single plan", file=sys.stderr)
        return 2

    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    skip = set(args.skip)

    results = []
    for plan in plans:
        if not plan.exists():
            print(f"WARN: plan not found: {plan}", file=sys.stderr)
            continue
        label = args.label or label_from_plan(plan)
        results.append(evaluate_one(
            plan_path=plan, label=label, onnx_path=args.onnx,
            sample_left=sample_left, sample_right=sample_right,
            macvo_exe=macvo_exe, dataset_dir=dataset,
            runs_dir=args.runs_dir, input_shape=input_shape, skip=skip,
        ))

    print(f"\n========================================")
    print(f" Done: {len(results)} plan(s) evaluated")
    print(f"========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
