import torch
from Utility.Extensions import OnCallCompiler

@OnCallCompiler()
def flow_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    flow_loss = torch.tensor(0.0, device=gt.device)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (preds[i] - gt).abs()
        flow_loss += i_weight * (mask * i_loss).nanmean()

    return flow_loss


def cov_nll(pred: torch.Tensor, gt: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-pixel Gaussian NLL (without the constant) and Mahalanobis term r^T Sigma^-1 r, both [B, 1, H, W].

    c is the covariance head output: [B, 2, H, W] log sigma per axis (shipped head), or [B, 3, H, W]
    Cholesky parameters (L11 = e^c1, L22 = e^c2, L21 = c3; Sigma = L L^T). The 2-channel head is the
    Cholesky form with c3 = 0. r^T Sigma^-1 r = |L^-1 r|^2 by forward substitution, log det Sigma = 2 (c1 + c2).
    r is zeroed outside `mask` (e.g. undefined stereo targets) so no NaN reaches the gradient.
    """
    r  = torch.where(mask, pred.detach() - gt, 0.)
    c1, c2 = c[:, 0:1], c[:, 1:2]
    c3 = c[:, 2:3] if c.size(1) == 3 else torch.zeros_like(c1)
    y1 = r[:, 0:1] * torch.exp(-c1)
    y2 = (r[:, 1:2] - c3 * y1) * torch.exp(-c2)
    maha = y1 * y1 + y2 * y2
    return maha + 2 * (c1 + c2), maha


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, x, 0.).sum() / mask.sum().clamp(min=1)


@OnCallCompiler()
def cov_loss(gamma: float, preds: list[torch.Tensor], gt: torch.Tensor, cov_preds: list[torch.Tensor],
             mask: torch.Tensor, beta: float = 0.) -> torch.Tensor:
    """
    gamma-weighted sum over decoder iterations of the mean per-pixel NLL over `mask` ([B, 1, H, W] bool).
    beta > 0 is beta-NLL (Seitzer et al. 2022): each pixel is scaled by stopgrad(det Sigma)^(beta / 2)
    = stopgrad(exp(beta (c1 + c2))). beta = 0 is the plain NLL.
    """
    n_predictions = len(preds)
    loss = torch.zeros((), device=gt.device)
    for i in range(n_predictions):
        nll, _ = cov_nll(preds[i], gt, cov_preds[i], mask)
        if beta > 0:
            nll = nll * torch.exp(beta * (cov_preds[i][:, 0:1] + cov_preds[i][:, 1:2])).detach()
        loss = loss + gamma ** (n_predictions - i - 1) * masked_mean(nll, mask)
    return loss
    

@torch.no_grad()
def cov_metrics(pred: torch.Tensor, gt: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Last-iteration NLL, mean r^T Sigma^-1 r (2 when calibrated), EPE and mean |c3| over `mask`."""
    nll, maha = cov_nll(pred, gt, c, mask)
    metrics = {
        "nll"  : masked_mean(nll, mask).item(),
        "calib": masked_mean(maha, mask).item(),
        "epe"  : masked_mean(torch.where(mask, pred - gt, 0.).norm(dim=1, keepdim=True), mask).item(),
    }
    if c.size(1) == 3:
        metrics["abs_c3"] = masked_mean(c[:, 2:3].abs(), mask).item()
    return metrics

@OnCallCompiler()
def depth_loss(gamma: float, preds: torch.Tensor, gt_depth: torch.Tensor, Ks: torch.Tensor, bls: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    fxs = Ks[:, 0, 0]
    gt_disparity  = (fxs * bls).unsqueeze(-1).unsqueeze(-1) / gt_depth 
    est_disparity = preds[..., 0]
    
    depth_loss = torch.tensor(0.0, device=gt_disparity.device)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss   = (est_disparity[i] + gt_disparity)
        depth_loss = i_weight * i_loss.mean()
    
    return depth_loss

def sequence_loss(cfg, preds: torch.Tensor, gt: torch.Tensor, flow_mask: torch.Tensor | None, cov_preds: list[torch.Tensor] | None):    
    """
    Covariance modes supervise every pixel with a finite target (NaN marks undefined targets, e.g. stereo
    pixels without valid depth). Occluded and out-of-view pixels stay in on purpose: that is where the
    head must learn a large sigma. `cov_mask: True` restricts the loss to flow_mask-valid pixels with
    |gt| < max_flow (an ablation, not the default).
    """
    gt_mag = gt.norm(dim=1, keepdim=True)
    mask = gt_mag < cfg.max_flow
    if flow_mask is not None: mask &= flow_mask.bool()
    
    metrics = dict()
    
    match cfg.training_mode:
        case "flow":
            loss = flow_loss(cfg.gamma, preds, gt, mask)
        
        case "cov" | "finalcov":
            assert cov_preds is not None
            cov_mask = torch.isfinite(gt).all(dim=1, keepdim=True)
            if cfg.get("cov_mask", False): cov_mask &= mask
            if cfg.training_mode == "finalcov":
                loss = cov_loss(1.0, preds[-1:], gt, cov_preds[-1:], cov_mask, cfg.get("beta", 0.))
            else:
                loss = cov_loss(cfg.gamma, preds, gt, cov_preds, cov_mask, cfg.get("beta", 0.))
            metrics = cov_metrics(preds[-1], gt, cov_preds[-1], cov_mask)
            
        case default:
            raise ValueError(f"Unavailable training mode {default}")      
    return loss, metrics


@torch.no_grad()
def sequence_metric(cfg, preds: torch.Tensor, cov_preds: list[torch.Tensor] | None, gt: torch.Tensor, flow_mask: torch.Tensor | None) -> dict[str, float]:
    """EPE statistics (and covariance metrics, if any) on flow_mask-valid pixels with |gt| < max_flow."""
    epe = (preds[-1] - gt).norm(dim=1, keepdim=True)
    mask = gt.norm(dim=1, keepdim=True) < cfg.max_flow
    if flow_mask is not None: mask &= flow_mask.bool()
    masked_epe = epe[mask]
    
    metrics = {
        'epe': masked_epe.mean().item(),
        '1px': (masked_epe < 1).float().mean().item(),
        '3px': (masked_epe < 3).float().mean().item(),
        '5px': (masked_epe < 5).float().mean().item(),
    }
    
    flow_gt_thresholds = [5, 10, 20]
    gt_mag = gt.norm(dim=1, keepdim=True)[mask]
    for t in flow_gt_thresholds:
        e = masked_epe[gt_mag < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    if cov_preds is not None:
        metrics.update({f"valid_{k}": v for k, v in cov_metrics(preds[-1], gt, cov_preds[-1], mask).items()})
    return metrics
