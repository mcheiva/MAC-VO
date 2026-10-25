import cv2
import functools
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

    A pose can also be off in a way that keeps the scale: Scripts/Dataset/check_eiva_photo.py stores, per frame, the
    share of textured GT pixels its own images confirm (`photo_stereo`: the right image at the GT disparity;
    `photo_temporal`: the next frame at the GT flow). With `photo_rel` set, a frame below max(`photo_floor`,
    `photo_rel` x the dive's median) loses its stereo target, a pair below it its flow target.

    `frame_stride` k (with `frame_offset`) makes the sequence every k-th frame, so consecutive items are k frames
    apart: larger displacements from the same footage. Their flow target is computed from the first frame's GT depth
    and the two GT poses (pose_gt.txt), kept where the second frame's GT depth confirms the point within 3%
    (occlusion, out of view), and only when every pair in between and both frames pass the gates.

    `water_disparity` gives open water (farther than `water_margin` px from any GT pixel) a far stereo target, the
    convention synthetic data has for the sky; for matcher fine-tunes, whose stereo otherwise drifts to near surfaces
    there (open water has no GT).
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
        # photo gate: share of confirmed GT pixels against the dive's median (NaN, too little GT to judge: passes)
        self.stereo_ok, self.temporal_ok = self.consistent.copy(), np.ones(len(self.frames), bool)
        if (photo_rel := getattr(cfg, "photo_rel", None)) is not None:
            if "photo_stereo" not in index:
                raise KeyError(f"photo_rel needs Scripts/Dataset/check_eiva_photo.py's scores in {self.gt_root / 'index.json'}")
            floor = float(getattr(cfg, "photo_floor", 0.5))
            for key, ok in (("photo_stereo", self.stereo_ok), ("photo_temporal", self.temporal_ok)):
                share = np.array(index[key], dtype=np.float64)
                ok &= ~(share < max(floor, float(photo_rel) * np.nanmedian(share)))
        # frame_range [start, stop): only this contiguous block of GT frames, e.g. to fine-tune the matcher and train the
        # covariance head on disjoint frames of a site. The gates keep one frame past the block for its last pair.
        if (frame_range := getattr(cfg, "frame_range", None)) is not None:
            start, stop = int(frame_range[0]), min(int(frame_range[1]), len(self.frames))
            self.frames = self.frames[start:stop]
            self.pair_ok, self.consistent = self.pair_ok[start:stop + 1], self.consistent[start:stop + 1]
            self.stereo_ok, self.temporal_ok = self.stereo_ok[start:stop + 1], self.temporal_ok[start:stop + 1]
        # frame_stride / frame_offset: items are frames offset, offset + k, ... of the (block of) GT frames
        self.stride = int(getattr(cfg, "frame_stride", 1))
        self.items = np.arange(int(getattr(cfg, "frame_offset", 0)), len(self.frames), self.stride)
        # this block's poses only, as arrays: the training script builds sequences in a process pool and pickles each
        # back on its own, so a shared per-dive dict would be copied into every one of thousands of sequences
        if self.stride > 1:
            poses = load_poses(self.root / "pose_gt.txt")
            self.pose_R = np.stack([poses[n][0] for n in self.frames])
            self.pose_t = np.stack([poses[n][1] for n in self.frames])
        # water_disparity: open water (no GT pixel within water_margin px) gets this stereo target (px), i.e. far, as
        # synthetic data has for the sky; MAC-VO's selector skips points beyond fx b / 1 px. Matcher fine-tunes only.
        self.water_disparity = getattr(cfg, "water_disparity", None)
        self.water_margin = float(getattr(cfg, "water_margin", 24))
        super().__init__(len(self.items))

    def load_image(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Failed to read image from {path}")
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.width, self.height), interpolation=cv2.INTER_AREA)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.

    def strided_flow(self, depth: np.ndarray, a: int, b: int) -> torch.Tensor:
        """Flow [1, 2, H, W] from frame a to frame b (indices into self.frames): `depth` back-projected at the pixel
        centres, moved by the relative GT pose, kept where b's GT depth confirms the point (within 3%); NaN elsewhere."""
        R1, t1, R2, t2, nxt = self.pose_R[a], self.pose_t[a], self.pose_R[b], self.pose_t[b], self.frames[b]
        R = R2 @ R1.T                                                       # camera 1 -> camera 2 (X_c = R X_w + t)
        K = self.K[0].double().numpy()
        v, u = np.nonzero(depth > 0)
        z = depth[v, u].astype(np.float64)
        X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1) @ R.T + (t2 - R @ t1)
        u2, v2 = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
        iu, iv = np.round(u2).astype(np.int64), np.round(v2).astype(np.int64)
        seen = (X[:, 2] > 0.05) & (iu >= 0) & (iu < self.width) & (iv >= 0) & (iv < self.height)
        d2 = np.zeros(len(z))
        d2[seen] = np.load(self.gt_root / f"{Path(nxt).stem}.npz")["depth"].astype(np.float64)[iv[seen], iu[seen]]
        seen &= (d2 > 0) & (np.abs(X[:, 2] - d2) < 0.03 * d2)
        flow = np.full((2, self.height, self.width), np.nan, np.float32)
        flow[0, v[seen], u[seen]] = u2[seen] - u[seen]
        flow[1, v[seen], u[seen]] = v2[seen] - v[seen]
        return torch.from_numpy(flow).unsqueeze(0)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = int(self.items[self.get_index(local_index)])    # into self.frames and the gates
        name  = self.frames[index]
        gt    = np.load(self.gt_root / f"{Path(name).stem}.npz")
        depth = torch.tensor(gt["depth"], dtype=torch.float32)[None, None]
        nxt   = index + self.stride
        gates = slice(index, nxt)
        if nxt < len(self.pair_ok) and self.pair_ok[gates].all() and self.temporal_ok[gates].all() \
                and self.consistent[index] and self.consistent[nxt] \
                and (self.stride == 1 or (self.stereo_ok[index] and self.stereo_ok[nxt] and nxt < len(self.frames))):
            if self.stride == 1:
                flow = torch.tensor(gt["flow"], dtype=torch.float32).unsqueeze(0) if "flow" in gt \
                       else torch.full((1, 2, self.height, self.width), float("nan"))
            else:
                flow = self.strided_flow(gt["depth"].astype(np.float32), index, nxt)
        else:       # last frame (no next frame) or inconsistent GT
            flow = torch.full((1, 2, self.height, self.width), float("nan"))
        if not self.stereo_ok[index]:
            depth = torch.zeros_like(depth)
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
        }, allow_excessive_cfg=True)    # optional: max_scale_error, max_flow_error, photo_rel, photo_floor,
                                        # water_disparity, water_margin, frame_range, frame_stride, frame_offset


@functools.lru_cache(maxsize=None)
def load_poses(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """pose_gt.txt rows `<left image> x y z r00 .. r22`, world -> camera (X_c = R X_w + t), as make_eiva_gt.py reads them."""
    poses = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 13:
            poses[p[0]] = (np.array(p[4:13], np.float64).reshape(3, 3), np.array(p[1:4], np.float64))
    return poses
