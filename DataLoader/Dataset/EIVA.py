import cv2
import json
import torch
import numpy as np
import pypose as pp
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..SequenceBase import SequenceBase
from ..Interface    import StereoFrame, StereoData


class EIVA_StereoSequence(SequenceBase[StereoFrame]):
    """
    An EIVA stereo survey sequence (`<root>/processed/{left,right}/*.jpg`, rectified) with the depth and optical
    flow ground truth rendered from its photogrammetry by Scripts/Dataset/make_eiva_gt.py (`gt` directory).
    Frames are the posed frames of the GT index, at the GT's resolution (the images are downscaled with
    INTER_AREA, as the MACVO_TRT runtime feeds the engine); gt_flow of a frame goes to the next indexed frame.
    Pixels without GT have depth 0 and NaN flow, so stereo_target and the covariance loss skip them.

    Where the reconstruction is locally wrong, a frame's rendered depth disagrees with its own stereo pair by a
    few percent everywhere, and a wrong relative pose shifts a pair's whole flow field (Scripts/Dataset/
    check_eiva_gt.py stores the per-frame `disparity_ratio` and `flow_error`). A frame beyond `max_scale_error`
    (default 0.025), or in a pair beyond `max_flow_error` (default 0.75 px), has no stereo target; a pair
    touching such a frame has no flow target.

    `teacher` (Scripts/Dataset/make_eiva_teacher.py output) fills the pixels without a target with the original
    network's flow and depth, for matcher fine-tunes: open water has no target and the stereo drifts there otherwise.
    `water_disparity` instead gives open water (farther than `water_margin` px from any GT pixel) a far stereo
    target, the convention synthetic data has for the sky.
    """
    @classmethod
    def name(cls) -> str: return "EIVA_NoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.root    = Path(cfg.root)
        self.gt_root = Path(cfg.gt)
        index = json.loads((self.gt_root / "index.json").read_text())
        self.frames   = list(index["frames"])
        self.height, self.width = index["hw"]
        self.K        = torch.tensor(index["K"], dtype=torch.float32).unsqueeze(0)
        self.baseline = float(index["baseline"])
        self.T_BS     = pp.identity_SE3(1)
        ratio = np.array(index.get("disparity_ratio", [1.] * len(self.frames)), dtype=np.float64)
        flow_error = np.array(index.get("flow_error", [0.] * len(self.frames)), dtype=np.float64)
        # pair i -> i + 1 consistent (the last frame has no pair: NaN); a frame keeps its stereo target if its scale
        # is consistent and neither of its pairs is off (one of the two poses is wrong, which one is unknown)
        self.pair_ok = ~(flow_error > getattr(cfg, "max_flow_error", 0.75))
        self.consistent = (np.abs(ratio - 1) <= getattr(cfg, "max_scale_error", 0.025)) & self.pair_ok \
                          & np.concatenate([[True], self.pair_ok[:-1]])                       # NaN ratio (no GT) -> False
        # use_filled: false keeps only pixels with a point of their own (make_eiva_gt's `filled` mask)
        self.use_filled = bool(getattr(cfg, "use_filled", True))
        # teacher: Scripts/Dataset/make_eiva_teacher.py output. Pixels without a target take the original network's
        # flow and depth (from its disparity), so a flow fine-tune keeps its behaviour in open water. Matcher
        # fine-tunes only: the covariance head has to learn real errors.
        self.teacher = Path(cfg.teacher) if getattr(cfg, "teacher", None) else None
        # water_disparity: open water (no GT pixel within water_margin px) gets this stereo target (px), i.e. far, as
        # synthetic data has for the sky; MAC-VO's selector skips points beyond fx b / 1 px. Matcher fine-tunes only.
        self.water_disparity = getattr(cfg, "water_disparity", None)
        self.water_margin = float(getattr(cfg, "water_margin", 24))
        super().__init__(len(self.frames))

    def load_image(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Failed to read image from {path}")
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.width, self.height), interpolation=cv2.INTER_AREA)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        name  = self.frames[index]
        gt    = np.load(self.gt_root / f"{Path(name).stem}.npz")
        depth = torch.tensor(gt["depth"], dtype=torch.float32)[None, None]
        if "flow" in gt and self.pair_ok[index] and self.consistent[index] and self.consistent[index + 1]:
            flow = torch.tensor(gt["flow"], dtype=torch.float32).unsqueeze(0)
        else:       # last frame (no next frame) or inconsistent GT
            flow = torch.full((1, 2, self.height, self.width), float("nan"))
        if not self.consistent[index]:
            depth = torch.zeros_like(depth)
        if not self.use_filled and "filled" in gt:
            hole = torch.tensor(gt["filled"].astype(bool))[None, None]
            depth = torch.where(hole, 0., depth)
            flow = torch.where(hole, float("nan"), flow)
        if self.teacher is not None:
            t = np.load(self.teacher / f"{Path(name).stem}.npz")
            disparity = torch.tensor(t["disparity"], dtype=torch.float32)[None, None]
            depth = torch.where((depth <= 0) & (disparity > 0.5), self.K[0, 0, 0] * self.baseline / disparity.clamp(min=0.5), depth)
            if "flow" in t:
                flow = torch.where(torch.isfinite(flow), flow, torch.tensor(t["flow"], dtype=torch.float32).unsqueeze(0))
        if self.water_disparity is not None:
            water = torch.from_numpy(distance_transform_edt(gt["depth"] <= 0) > self.water_margin)[None, None]
            depth = torch.where(water, self.K[0, 0, 0] * self.baseline / float(self.water_disparity), depth)
        return StereoFrame(
            idx     = [local_index],
            time_ns = [index * 1000],       # FIXME: a fake timestamp (as GeneralStereo).
            stereo  = StereoData(
                T_BS      = self.T_BS,
                K         = self.K,
                baseline  = torch.tensor([self.baseline]),
                time_ns   = [index * 1000],
                height    = self.height,
                width     = self.width,
                imageL    = self.load_image(self.root / "processed" / "left" / name),
                imageR    = self.load_image(self.root / "processed" / "right" / name.replace("image_left", "image_right")),
                gt_depth  = depth,
                gt_flow   = flow,
                flow_mask = torch.isfinite(flow).all(dim=1, keepdim=True).float(),
            )
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root": lambda s: isinstance(s, str),
            "gt"  : lambda s: isinstance(s, str),
        }, allow_excessive_cfg=True)    # optional: max_scale_error, max_flow_error, use_filled, teacher, water_disparity, water_margin
