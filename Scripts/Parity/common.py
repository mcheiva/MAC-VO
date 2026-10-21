"""Shared helpers for the parity harness. Imports the network package directly: `import
Module` pulls in the whole odometry pipeline (DataLoader, scipy), which the harness does not need."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "Module") not in sys.path:
    sys.path.insert(0, str(ROOT / "Module"))

from Network.FlowFormer.configs.submission import get_cfg          # noqa: E402
from Network.FlowFormerCov import build_flowformer                 # noqa: E402
from Network.FlowFormerCov.covhead import cov_entries              # noqa: E402

torch.backends.cudnn.allow_tf32 = False          # the harness means fp32 when it says fp32
torch.backends.cuda.matmul.allow_tf32 = False

CKPT = ROOT / "Model" / "MACVO_FrontendCov.pth"
RES = 704
DTYPES = {"fp32": (torch.float32, torch.float32), "fast": (torch.float16, torch.bfloat16)}
PAIR_TYPES = ("stereo", "temporal", "kf")
DATA_ROOT = Path("C:/src/git/AI/Photogrammetry/TestingUtils/performancedata")


def build_model(mode: str = "fp32", decoder_depth: int = 12, device: str = "cuda",
                ckpt: Path = CKPT) -> torch.nn.Module:
    cfg = get_cfg()
    cfg.latentcostformer.decoder_depth = decoder_depth
    enc, dec = DTYPES[mode]
    model = build_flowformer(cfg, enc, dec)
    model.load_ddp_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.eval().to(device)


@torch.inference_mode()
def run_model(model, img1: torch.Tensor, img2: torch.Tensor) -> dict[str, torch.Tensor]:
    """img*: [B,3,H,W] float in [0,1] on the model device. Returns fp32 outputs on CPU in the engine layout
    (Export.py): cov is log sigma for the shipped head, [sigma_uu, sigma_vv, sigma_uv] for any other."""
    (flow, flow_raw), (cov, cov_raw) = model(img1, img2)
    if not model.memory_decoder.log_sigma_output:
        cov, cov_raw = cov_entries(cov.float()), cov_entries(cov_raw.float())
    return {k: v.float().cpu() for k, v in
            dict(flow=flow, flow_raw=flow_raw, cov=cov, cov_raw=cov_raw).items()}


def list_frames(seq_dir: Path) -> list[tuple[Path, Path]]:
    left = sorted(p for p in (seq_dir / "processed" / "left").glob("*.jpg"))
    right = sorted(p for p in (seq_dir / "processed" / "right").glob("*.jpg"))
    assert len(left) == len(right), (seq_dir, len(left), len(right))
    return list(zip(left, right))


def load_image_u8(path: Path) -> np.ndarray:
    """-> uint8 [3,704,704] RGB. 4x downscale with INTER_AREA; candidates see these exact bytes."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert im is not None, path
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im.transpose(2, 0, 1))


def u8_to_input(u8: np.ndarray, device="cuda") -> torch.Tensor:
    return torch.from_numpy(u8).to(device).float().div_(255.0).unsqueeze(0)
