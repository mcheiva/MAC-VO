"""
Synthetic camera rotation for temporal training pairs: the second camera turns about its centre by a random R, which
moves every pixel by the homography H = K R K^-1 whatever the scene depth, so the GT flow stays exact:

    image 2'(x') = image 2(H^-1 x')        flow'(x1) = H (x1 + flow(x1)) - x1

kept where H (x1 + flow) lands inside the image. Image 2' is black where it shows nothing of image 2. A diver's sharp
turn is exactly this (plane_tail1 #330-#332: 11-29 deg between frames, 240-290 px of flow), and EIVA's slow pairs,
even with frame gaps, rarely move that far. FlowNet-style relative augmentations do the same with in-plane transforms;
the homography of a 3D rotation is the one a real camera produces.
"""
import math

import torch
import torch.nn.functional as F


def random_rotations(n: int, max_deg: float, p: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """-> R [n, 3, 3] and the mask of rotated samples: a share p gets a rotation by U(0, max_deg) about a random axis."""
    chosen = torch.rand(n, device=device) < p
    axis = F.normalize(torch.randn(n, 3, device=device), dim=1)
    angle = (torch.rand(n, device=device) * math.radians(max_deg) * chosen)[:, None, None]
    x, y, z = axis.unbind(1)
    o = torch.zeros_like(x)
    skew = torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=1).view(n, 3, 3)
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    return eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * skew @ skew, chosen


def rotate_second(img2: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor | None, K: torch.Tensor,
                  R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """img2 [B, C, H, W], flow [B, 2, H, W] (NaN where no target), mask [B, 1, H, W] or None, K [B, 3, 3] of the
    (cropped) images, R [B, 3, 3] -> img2', flow', mask' as if camera 2 had turned by R."""
    B, _, Hh, Ww = img2.shape
    Hm = K.float() @ R @ torch.linalg.inv(K.float())
    ys, xs = torch.meshgrid(torch.arange(Hh, device=img2.device, dtype=torch.float32),
                            torch.arange(Ww, device=img2.device, dtype=torch.float32), indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)]).view(1, 3, -1)
    src = torch.linalg.inv(Hm) @ pts
    uv = src[:, :2] / src[:, 2:].clamp(min=1e-6)
    grid = torch.stack([uv[:, 0] / (Ww - 1) * 2 - 1, uv[:, 1] / (Hh - 1) * 2 - 1], dim=-1).view(B, Hh, Ww, 2)
    # sources behind the camera (src z <= 0) sample nothing
    grid = torch.where((src[:, 2] > 0).view(B, Hh, Ww, 1), grid, torch.full_like(grid, 2.))
    img2 = F.grid_sample(img2.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True).to(img2.dtype)
    x2 = (pts[:, :2].view(1, 2, Hh, Ww) + flow).flatten(2)
    p = Hm @ torch.cat([x2, torch.ones_like(x2[:, :1])], dim=1)
    x2r = p[:, :2] / p[:, 2:].clamp(min=1e-6)
    new = (x2r - pts[:, :2]).view(B, 2, Hh, Ww)
    valid = torch.isfinite(new).all(1, keepdim=True) & (p[:, 2:] > 0).view(B, 1, Hh, Ww) \
        & (x2r[:, :1] >= 0).view(B, 1, Hh, Ww) & (x2r[:, :1] <= Ww - 1).view(B, 1, Hh, Ww) \
        & (x2r[:, 1:] >= 0).view(B, 1, Hh, Ww) & (x2r[:, 1:] <= Hh - 1).view(B, 1, Hh, Ww)
    flow = torch.where(valid, new, float("nan")).to(flow.dtype)
    return img2, flow, (mask * valid) if mask is not None else None
