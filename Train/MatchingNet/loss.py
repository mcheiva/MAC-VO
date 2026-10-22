import math
import torch
import torch.nn.functional as F
from Utility.Extensions import OnCallCompiler

@OnCallCompiler()
def flow_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """gamma-weighted L1 over iterations: the mean over elements with a finite target of mask * |pred - gt| (the
    value `(mask * |pred - gt|).nanmean()` had). Undefined targets (NaN: stereo pixels without depth, EIVA pixels
    without GT) are zeroed before the subtraction - 0 * NaN in the backward pass would poison every gradient."""
    n_predictions = len(preds)
    finite = torch.isfinite(gt)
    use = finite & mask.bool()
    gt = torch.where(finite, gt, 0.)
    count = finite.sum().clamp(min=1)

    flow_loss = torch.tensor(0.0, device=gt.device)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = torch.where(use, (preds[i] - gt).abs(), 0.)
        flow_loss += i_weight * i_loss.sum() / count

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
    c3 = c[:, 2:3] if c.size(1) >= 3 else torch.zeros_like(c1)
    y1 = r[:, 0:1] * torch.exp(-c1)
    y2 = (r[:, 1:2] - c3 * y1) * torch.exp(-c2)
    maha = y1 * y1 + y2 * y2
    return maha + 2 * (c1 + c2), maha


def cov_mixture_nll(pred: torch.Tensor, gt: torch.Tensor, c: torch.Tensor, mask: torch.Tensor,
                    outlier_var: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Two-component NLL for the single-shot inlier head, c = [c1, c2, c3, logit p]: the error is drawn from
    N(0, Sigma) with probability p and from N(0, outlier_var I) otherwise. Same units as cov_nll (-2 log
    density without the 2 log 2 pi constant); returns (nll, inlier r^T Sigma^-1 r).
    """
    nll_in, maha = cov_nll(pred, gt, c[:, :3], mask)
    r = torch.where(mask, pred.detach() - gt, 0.)
    nll_out = (r * r).sum(dim=1, keepdim=True) / outlier_var + 2 * math.log(outlier_var)
    logit = c[:, 3:4]
    return -2 * torch.logaddexp(F.logsigmoid(logit) - 0.5 * nll_in, F.logsigmoid(-logit) - 0.5 * nll_out), maha


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, x, 0.).sum() / mask.sum().clamp(min=1)


@OnCallCompiler()
def cov_loss(gamma: float, preds: list[torch.Tensor], gt: torch.Tensor, cov_preds: list[torch.Tensor],
             mask: torch.Tensor, beta: float = 0., outlier_var: float = 32. ** 2) -> torch.Tensor:
    """
    gamma-weighted sum over decoder iterations of the mean per-pixel NLL over `mask` ([B, 1, H, W] bool).
    beta > 0 is beta-NLL (Seitzer et al. 2022): each pixel is scaled by stopgrad(det Sigma)^(beta / 2)
    = stopgrad(exp(beta (c1 + c2))). beta = 0 is the plain NLL. 4-channel predictions (single-shot inlier
    head) use the mixture NLL; the single-shot head has one prediction, so gamma does not apply.
    """
    n_predictions = len(preds)
    loss = torch.zeros((), device=gt.device)
    for i in range(n_predictions):
        if cov_preds[i].size(1) == 4:
            nll, _ = cov_mixture_nll(preds[i], gt, cov_preds[i], mask, outlier_var)
        else:
            nll, _ = cov_nll(preds[i], gt, cov_preds[i], mask)
        if beta > 0:
            nll = nll * torch.exp(beta * (cov_preds[i][:, 0:1] + cov_preds[i][:, 1:2])).detach()
        loss = loss + gamma ** (n_predictions - i - 1) * masked_mean(nll, mask)
    return loss


@torch.no_grad()
def cov_metrics(pred: torch.Tensor, gt: torch.Tensor, c: torch.Tensor, mask: torch.Tensor,
                outlier_var: float = 32. ** 2) -> dict[str, float]:
    """Last-iteration NLL, mean r^T Sigma^-1 r (2 when calibrated; inlier Sigma for the mixture), EPE,
    mean |c3| and, for the inlier head, the mean inlier probability over `mask`."""
    if c.size(1) == 4:
        nll, maha = cov_mixture_nll(pred, gt, c, mask, outlier_var)
    else:
        nll, maha = cov_nll(pred, gt, c, mask)
    metrics = {
        "nll"  : masked_mean(nll, mask).item(),
        "calib": masked_mean(maha, mask).item(),
        "epe"  : masked_mean(torch.where(mask, pred - gt, 0.).norm(dim=1, keepdim=True), mask).item(),
    }
    if c.size(1) >= 3:
        metrics["abs_c3"] = masked_mean(c[:, 2:3].abs(), mask).item()
    if c.size(1) == 4:
        metrics["inlier"] = masked_mean(torch.sigmoid(c[:, 3:4]), mask).item()
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
            outlier_var = cfg.latentcostformer.get("outlier_sigma", 32.) ** 2
            if cfg.training_mode == "finalcov":
                loss = cov_loss(1.0, preds[-1:], gt, cov_preds[-1:], cov_mask, cfg.get("beta", 0.), outlier_var)
            else:
                loss = cov_loss(cfg.gamma, preds, gt, cov_preds, cov_mask, cfg.get("beta", 0.), outlier_var)
            metrics = cov_metrics(preds[-1], gt, cov_preds[-1], cov_mask, outlier_var)
            
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
