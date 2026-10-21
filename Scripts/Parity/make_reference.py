"""Generate fp32 PyTorch reference outputs (batch 1) for the parity harness.

    python Scripts/Parity/make_reference.py                 # 3 seqs x 8 frames x 3 pair types
    python Scripts/Parity/make_reference.py --ckpt Model/<run>/25000.pth --out runs/ref_chol

Each pair -> runs/ref/<seq>/<frame>_<pair>.npz with the uint8 704x704 inputs, fp32 outputs in the
engine layout (cov is log-sigma for the shipped head, not exponentiated; [sigma_uu, sigma_vv,
sigma_uv] for any other head) and the per-iteration mean |delta_flow| of the decoder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, DATA_ROOT, PAIR_TYPES, ROOT, build_model, list_frames,  # noqa: E402
                    load_image_u8, run_model, u8_to_input)

KF_LAG = 5


def pick_frames(n: int, k: int) -> list[int]:
    return [int(round(v)) for v in np.linspace(KF_LAG, n - 1, k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "ref")
    ap.add_argument("--seqs", nargs="+", default=["plane_nose", "plane_tail1", "Welland"])
    ap.add_argument("--frames-per-seq", type=int, default=8)
    ap.add_argument("--mode", default="fp32", choices=["fp32", "fast"])
    ap.add_argument("--ckpt", type=Path, default=CKPT, help="frontend checkpoint (default: the shipped head)")
    args = ap.parse_args()

    model = build_model(args.mode, ckpt=args.ckpt)
    dec = model.memory_decoder
    for seq in args.seqs:
        frames = list_frames(args.root / seq)
        (args.out / seq).mkdir(parents=True, exist_ok=True)
        for t in pick_frames(len(frames), args.frames_per_seq):
            L_t = load_image_u8(frames[t][0]); R_t = load_image_u8(frames[t][1])
            L_p = load_image_u8(frames[t - 1][0]); L_k = load_image_u8(frames[t - KF_LAG][0])
            pairs = {"stereo": (L_t, R_t), "temporal": (L_p, L_t), "kf": (L_k, L_t)}
            outs: dict[str, dict] = {}
            logs: dict[str, list] = {}
            for name, (i1, i2) in pairs.items():
                dec.iter_log = []
                outs[name] = run_model(model, u8_to_input(i1), u8_to_input(i2))
                logs[name] = dec.iter_log
            dec.iter_log = None
            for name in PAIR_TYPES:
                i1, i2 = pairs[name]
                np.savez_compressed(
                    args.out / seq / f"{t:05d}_{name}.npz", img1=i1, img2=i2,
                    iter_delta=np.asarray(logs[name], np.float32),
                    **{k: v[0].numpy() for k, v in outs[name].items()},
                    seq=seq, frame=t, pair=name, left=frames[t][0].name)
            print(f"{seq} frame {t}: " + " ".join(f"{n}:{logs[n][-1]:.3f}px" for n in PAIR_TYPES), flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
