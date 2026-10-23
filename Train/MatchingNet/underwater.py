"""
Synthetic underwater appearance for in-air training pairs (TartanAir), rendered from their GT depth.

The revised underwater image formation model (Akkaynak & Treibitz, CVPR 2018), the usual way to synthesise
underwater training data from in-air RGB-D (e.g. UWCNN, Li et al. 2020, the Jerlov water types on NYU-v2):

    I_c = J_c exp(-beta_c^D z) + B_c (1 - exp(-beta_c^B z))

J is the in-air image, z the depth (standing in for the range), beta^D the attenuation, beta^B the backscatter
coefficient and B the veiling light, per colour channel. One set of water parameters is drawn per sample and shared
by both images of the pair, so the correspondence does not change. The depth target does not change either: fully
attenuated pixels keep their true, far depth, so the matcher learns that open water is far, across water colours and
turbidities it has not seen (a matcher fine-tuned on one site's water read another site's water as a near surface).
"""
import math

import torch
import torch.nn.functional as F


def right_view_depth(depth: torch.Tensor, fxb: torch.Tensor) -> torch.Tensor:
    """Left-view depth [B, 1, H, W] -> right-view depth, forward-warped by the disparity fx b / z (the nearest surface
    wins); pixels no left pixel lands on take the farthest valid depth within 32 px along the row, else the left depth."""
    B, _, H, W = depth.shape
    disp = fxb.view(-1, 1, 1, 1) / depth.clamp(min=1e-3)
    x = torch.arange(W, device=depth.device).view(1, 1, 1, W)
    xr = (x - disp).round().long()
    ok = (xr >= 0) & (xr < W)
    base = (torch.arange(B * H, device=depth.device) * W).view(B, 1, H, 1)
    idx = (base + xr)[ok]
    right = torch.zeros(B * H * W, device=depth.device, dtype=disp.dtype)
    right.scatter_reduce_(0, idx, disp[ok], reduce="amax")
    right = right.view(B, 1, H, W)
    hole = right <= 0
    far = -F.max_pool2d(-torch.where(hole, torch.full_like(right, float("inf")), right), (1, 65), stride=1, padding=(0, 32))
    right = torch.where(hole & torch.isfinite(far), far, right)
    right = torch.where(right <= 0, disp, right)
    return fxb.view(-1, 1, 1, 1) / right


def water(n: int, device: torch.device, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Water parameters for n samples ([n, 3, 1, 1] each, RGB): attenuation from clear to turbid (beta^D of the green
    channel log-uniform in 0.03-0.6 /m, red 1.5-4x and blue 0.4-1.5x of it), backscatter coefficient within 30% of the
    attenuation, and a green-to-blue veiling light from dark to bright."""
    u = lambda *shape: torch.rand(*shape, device=device, generator=generator)
    g = torch.exp(math.log(0.03) + u(n) * (math.log(0.6) - math.log(0.03)))
    beta_d = torch.stack([g * (1.5 + 2.5 * u(n)), g, g * (0.4 + 1.1 * u(n))], dim=1)
    beta_b = beta_d * (0.7 + 0.6 * u(n, 1))
    green = 0.15 + 0.6 * u(n)
    veil = torch.stack([green * (0.05 + 0.45 * u(n)), green, green * (0.6 + 0.8 * u(n))], dim=1).clamp(max=1.)
    return {"beta_d": beta_d.view(n, 3, 1, 1), "beta_b": beta_b.view(n, 3, 1, 1), "veil": veil.view(n, 3, 1, 1)}


def underwater(image: torch.Tensor, depth: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """image [B, 3, H, W] in [0, 1] seen through water w at depth [B, 1, H, W]."""
    z = depth.clamp(min=0.)
    return (image * torch.exp(-w["beta_d"] * z) + w["veil"] * (1 - torch.exp(-w["beta_b"] * z))).clamp(0., 1.)


def underwater_pair(img1: torch.Tensor, img2: torch.Tensor, depth1: torch.Tensor, depth2: torch.Tensor,
                    p: float, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """With probability p per sample, both images through the same water. Only samples with a dense depth (in-air
    synthetic data; EIVA's sparse photogrammetry depth is skipped, its images are underwater already)."""
    dense = ((depth1 > 0).flatten(1).float().mean(1) > 0.99) & ((depth2 > 0).flatten(1).float().mean(1) > 0.99)
    pick = dense & (torch.rand(img1.size(0), device=img1.device, generator=generator) < p)
    if not pick.any():
        return img1, img2
    w = water(img1.size(0), img1.device, generator)
    m = pick.view(-1, 1, 1, 1)
    return torch.where(m, underwater(img1, depth1, w), img1), torch.where(m, underwater(img2, depth2, w), img2)
