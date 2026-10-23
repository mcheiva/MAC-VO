import torch
import torch.nn.functional as F
import warnings
from typing import get_args, get_origin, Literal, TypeGuard, TypeVar

from DataLoader import StereoData


T_DataType  = Literal["fp32", "fp16", "bf16"]
T_TrainType = Literal["flow", "cov", "flow+cov", "finalcov"]
T_Scheduler = Literal["OneCycleLR"]
T_Optimizer = Literal["AdamW"]

T_Type = TypeVar("T_Type")

def AssertLiteralType(value: str | float | int | bool, t: T_Type) -> TypeGuard[T_Type]:
    if get_origin(t) is Literal:
        if value in get_args(t): return True
        else: raise ValueError(f"AssertLiteralType does not match - get {value}, expect to be one of {get_args(t)}")
    else:
        raise Exception(f"AssertLiteralType can only check for literal types.")


def get_datatype(type_string: T_DataType) -> torch.dtype:
    AssertLiteralType(type_string, T_DataType)
    match type_string:
        case "fp32":
            return torch.float32
        case "fp16":
            warnings.warn(f"Training with fp16 may cause loss to become nan. Consider use bf16/fp32 for better stability.")
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _: raise Exception("Impossible")


def get_scheduler(scheduler_string: T_Scheduler) -> type[torch.optim.lr_scheduler.LRScheduler]:
    AssertLiteralType(scheduler_string, T_Scheduler)
    
    match scheduler_string:
        case "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR
        case _: raise Exception("Impossible")


def stereo_target(data: StereoData) -> torch.Tensor:
    """Left -> right flow of a rectified pair, (-fx * b / depth, 0); NaN where depth is not finite and positive."""
    assert data.gt_depth is not None, "stereo supervision needs gtDepth: true in the sequence config"
    depth = data.gt_depth
    fxb   = (data.K[:, 0, 0] * data.baseline.view(-1)).view(-1, 1, 1, 1)
    valid = torch.isfinite(depth) & (depth > 0)
    u = torch.where(valid, -fxb / depth, float("nan"))
    return torch.cat([u, torch.where(valid, 0., float("nan"))], dim=1)


def water_mask(depth: torch.Tensor, margin: int = 24, min_coverage: float = 0.05) -> torch.Tensor:
    """
    Pixels known to have no correspondence, from a sparse GT depth [B, 1, H, W] (EIVA photogrammetry): farther than
    `margin` px (square window) from any pixel with depth, i.e. open water and haze; EIVA_NoIMU's water_margin.
    Samples with a dense depth (synthetic data) have none, and samples with less than `min_coverage` GT (gated
    frames, whose depth is withheld) get none either.
    """
    has = (torch.isfinite(depth) & (depth > 0)).float()
    near = F.max_pool2d(has, kernel_size=2 * margin + 1, stride=1, padding=margin) > 0
    usable = has.flatten(1).mean(1).view(-1, 1, 1, 1) >= min_coverage
    return ~near & usable


class IndexedDataset(torch.utils.data.Dataset):
    """Map-style view of a SequenceBase (an IterableDataset), which ConcatDataset and shuffling reject."""
    def __init__(self, sequence) -> None: self.sequence = sequence
    def __len__(self) -> int: return len(self.sequence)
    def __getitem__(self, index: int): return self.sequence[index]


def get_optimizer(optim_string: T_Optimizer) -> type[torch.optim.Optimizer]:
    AssertLiteralType(optim_string, T_Optimizer)
    
    match optim_string:
        case "AdamW":
            return torch.optim.AdamW
        case _: raise Exception("Impossible")
