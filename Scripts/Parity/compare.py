"""Compare a candidate (PyTorch module / ONNX / TRT plan) against runs/ref.

    python Scripts/Parity/compare.py --torch fp32            # exact rewrite check (gate: EPE max < 1e-3)
    python Scripts/Parity/compare.py --torch fast            # bf16/fp16 eager
    python Scripts/Parity/compare.py --trt MACVO_FrontendCov.plan [--pairing b2]
    python Scripts/Parity/compare.py --onnx MACVO_FrontendCov.onnx

--pairing b2 runs stereo+temporal of a frame as one batch-2 call in the production layout
(img1=[L_t, L_{t-1}], img2=[R_t, L_t]); kf stays batch-1. Default b1 runs every pair alone.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PAIR_TYPES, ROOT, build_model, run_model  # noqa: E402

OUT_KEYS = ("flow", "flow_raw", "cov", "cov_raw")
KEYS = ["epe_mean", "epe_p99", "epe_max", "epe_frac_gt_0.5", "dlogcov_mean", "dlogcov_p99",
        "raw_epe_mean", "disp_epe_mean"]


class TorchCand:
    def __init__(self, mode, depth):
        self.m = build_model(mode, decoder_depth=depth)

    def __call__(self, i1: np.ndarray, i2: np.ndarray) -> dict:   # [B,3,H,W] float32 in [0,1]
        o = run_model(self.m, torch.from_numpy(i1).cuda(), torch.from_numpy(i2).cuda())
        return {k: v.numpy() for k, v in o.items()}


class OnnxCand:
    def __init__(self, path):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()]
        self.s = ort.InferenceSession(str(path), providers=prov)
        self.names = [o.name for o in self.s.get_outputs()]

    def __call__(self, i1, i2):
        return dict(zip(self.names, self.s.run(None, {"image_1": i1, "image_2": i2})))


class TrtCand:
    """Direct TensorRT runner (same calls as macvo.cpp: set_input_shape + execute_async_v3).
    Fresh output tensors per call; polygraphy's TrtRunner reuses host buffers across infer()."""

    def __init__(self, path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, f"failed to deserialize {path}"
        self.ctx = self.engine.create_execution_context()

    def __call__(self, i1, i2):
        t1, t2 = torch.from_numpy(i1).cuda().contiguous(), torch.from_numpy(i2).cuda().contiguous()
        self.ctx.set_input_shape("image_1", tuple(t1.shape))
        self.ctx.set_input_shape("image_2", tuple(t2.shape))
        self.ctx.set_tensor_address("image_1", t1.data_ptr())
        self.ctx.set_tensor_address("image_2", t2.data_ptr())
        outs = {}
        for k in OUT_KEYS:
            outs[k] = torch.empty(tuple(self.ctx.get_tensor_shape(k)), device="cuda", dtype=torch.float32)
            self.ctx.set_tensor_address(k, outs[k].data_ptr())
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        assert ok, "execute_async_v3 failed"
        torch.cuda.synchronize()
        return {k: v.cpu().numpy() for k, v in outs.items()}


def metrics(c: dict, r: dict) -> dict:
    epe = np.linalg.norm(c["flow"] - r["flow"], axis=0)
    dcov = np.abs(c["cov"] - r["cov"])
    return {"epe_mean": float(epe.mean()), "epe_p99": float(np.quantile(epe, .99)),
            "epe_max": float(epe.max()), "epe_frac_gt_0.5": float((epe > .5).mean()),
            "dlogcov_mean": float(dcov.mean()), "dlogcov_p99": float(np.quantile(dcov, .99)),
            "raw_epe_mean": float(np.linalg.norm(c["flow_raw"] - r["flow_raw"], axis=0).mean()),
            "disp_epe_mean": float(np.abs(c["flow"][0] - r["flow"][0]).mean())}


def to_in(u8: np.ndarray) -> np.ndarray:
    return u8[None].astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--torch", choices=["fp32", "fast"])
    g.add_argument("--onnx", type=Path)
    g.add_argument("--trt", type=Path)
    ap.add_argument("--ref", type=Path, default=ROOT / "runs" / "ref")
    ap.add_argument("--pairing", choices=["b1", "b2"], default="b1")
    ap.add_argument("--depth", type=int, default=12, help="(--torch) decoder iterations")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (debug)")
    args = ap.parse_args()

    cand = (TorchCand(args.torch, args.depth) if args.torch else
            OnnxCand(args.onnx) if args.onnx else TrtCand(args.trt))

    files = sorted(args.ref.glob("*/*_stereo.npz"))
    if args.limit:
        files = files[:args.limit]
    rows = []
    for f in files:
        refs = {p: dict(np.load(str(f).replace("_stereo", f"_{p}"))) for p in PAIR_TYPES}
        outs = {}
        if args.pairing == "b2":
            i1 = np.concatenate([to_in(refs["stereo"]["img1"]), to_in(refs["temporal"]["img1"])])
            i2 = np.concatenate([to_in(refs["stereo"]["img2"]), to_in(refs["temporal"]["img2"])])
            o = cand(i1, i2)
            outs["stereo"] = {k: o[k][0] for k in OUT_KEYS}
            outs["temporal"] = {k: o[k][1] for k in OUT_KEYS}
            o = cand(to_in(refs["kf"]["img1"]), to_in(refs["kf"]["img2"]))
            outs["kf"] = {k: o[k][0] for k in OUT_KEYS}
        else:
            for p in PAIR_TYPES:
                o = cand(to_in(refs[p]["img1"]), to_in(refs[p]["img2"]))
                outs[p] = {k: o[k][0] for k in OUT_KEYS}
        line = []
        for p in PAIR_TYPES:
            m = metrics(outs[p], refs[p])
            m.update(seq=str(refs[p]["seq"]), frame=int(refs[p]["frame"]), pair=p)
            rows.append(m)
            line.append(f"{p}: epe {m['epe_mean']:.2e}")
        print(f"{rows[-1]['seq']:12s} {rows[-1]['frame']:5d}  " + "  ".join(line), flush=True)

    print(f"\n{'pair':9s}" + "".join(f"{k:>17s}" for k in KEYS))
    summary = {}
    for p in PAIR_TYPES + ("all",):
        sel = [r for r in rows if p == "all" or r["pair"] == p]
        agg = {k: (max(r[k] for r in sel) if k == "epe_max" else float(np.mean([r[k] for r in sel])))
               for k in KEYS}
        summary[p] = agg
        print(f"{p:9s}" + "".join(f"{agg[k]:17.3e}" for k in KEYS))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
        print("->", args.json)


if __name__ == "__main__":
    main()
