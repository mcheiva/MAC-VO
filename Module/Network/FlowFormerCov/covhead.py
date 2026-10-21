import torch
import torch.nn as nn

from ..FlowFormer.core.gru import SepConvGRU, FusedGRUPair
from ..FlowFormer.core.decoder import MemoryDecoder, initialize_flow


def cov_entries(c: torch.Tensor) -> torch.Tensor:
    """
    Decoder output c [B, C, H, W] -> covariance entries.
    C = 2 (shipped head): log sigma per axis -> [sigma_uu, sigma_vv].
    C = 3 (Cholesky head): L11 = exp(c1), L22 = exp(c2), L21 = c3 -> [sigma_uu, sigma_vv, sigma_uv] of
    Sigma = L L^T. With c3 = 0 this is the C = 2 mapping plus sigma_uv = 0.
    """
    if c.size(1) == 2:
        return torch.exp(2 * c)
    assert c.size(1) == 3, "cov_entries takes 2 (log sigma) or 3 (Cholesky) channels"
    l11, l22, l21 = torch.exp(c[:, 0:1]), torch.exp(c[:, 1:2]), c[:, 2:3]
    return torch.cat([l11 * l11, l21 * l21 + l22 * l22, l11 * l21], dim=1)


def correlation_peak_stats(corr: torch.Tensor) -> torch.Tensor:
    """Matching-confidence cues of a correlation window [B, K, H, W] (fp32): best score, its margin over the
    second best, and the entropy of the softmax over the window (Hu & Mordohai 2012 survey such measures)."""
    corr = corr.float()
    top2 = corr.topk(2, dim=1).values
    logp = corr.log_softmax(dim=1)
    return torch.cat([top2[:, :1], top2[:, :1] - top2[:, 1:], -(logp.exp() * logp).sum(dim=1, keepdim=True)], dim=1)


CUE_DIM = 5     # recurrent head cues per iteration: correlation peak stats (3) + this iteration's flow update (2)


class CovHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, out_dim=2):
        super(CovHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 4, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def set_out_dim(self, out_dim: int) -> None:
        """Replace conv4 by a fresh conv with `out_dim` outputs (same dtype/device); caller loads its weights."""
        old = self.conv4
        self.conv4 = nn.Conv2d(old.in_channels, out_dim, 3, padding=1).to(device=old.weight.device, dtype=old.weight.dtype)

    def set_in_dim(self, in_dim: int) -> None:
        """Replace conv1 by a fresh conv with `in_dim` inputs (same dtype/device); caller loads its weights."""
        old = self.conv1
        self.conv1 = nn.Conv2d(in_dim, old.out_channels, 3, padding=1).to(device=old.weight.device, dtype=old.weight.dtype)

    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(x)))
        x = self.conv4(self.relu(self.conv3(x)))
        return x


class CovUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
        )
        # cov_channels: 2 = per-axis log sigma (shipped), 3 = Cholesky factor of the full 2x2 covariance.
        # cov_cues: CovHead also reads CUE_DIM matching-confidence cues per iteration (see MemoryCovDecoder).
        cues = CUE_DIM if getattr(args, "cov_cues", False) else 0
        self.cov_head = CovHead(hidden_dim + cues, hidden_dim=256, out_dim=getattr(args, "cov_channels", 2))
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward_state(self, covs_net, inp_cat):
        covs_net   = self.gru(covs_net, inp_cat)
        delta_covs = self.cov_head(covs_net)
        return covs_net, delta_covs

    def forward_mask(self, covs_net):
        return 0.25 * self.mask(covs_net)

    def forward(self, covs_net, inp_cat):
        covs_net, delta_covs = self.forward_state(covs_net, inp_cat)
        return covs_net, delta_covs, self.forward_mask(covs_net)


class SingleShotCovHead(nn.Module):
    """
    Covariance head that runs once, after the decoder loop, on what the flow decoder has computed: the final
    flow-GRU state (it carries the iteration history), the last motion features, the context, the 9x9
    correlation window at the final flow with its peak statistics, the flow and the last two flow updates.
    Replaces the recurrent branch (a second GRU and CovHead on every iteration, ~13% of the engine).

    Outputs Cholesky parameters (c1, c2, c3) at 1/8 resolution, upsampled with the flow's convex mask. With
    `inlier`, a fourth channel is the logit of p, the probability that the error is drawn from N(0, Sigma)
    rather than from the broad outlier term N(0, outlier_sigma^2 I); training uses the mixture likelihood
    and consumers get the moment-matched covariance p Sigma + (1 - p) outlier_sigma^2 I (`effective`).
    Runs in fp32 whatever the decoder dtype.
    """
    def __init__(self, corr_dim: int = 81, inlier: bool = False, outlier_sigma: float = 32.):
        super().__init__()
        in_dim = 128 + 256 + 128 + corr_dim + 3 + 2 + 2 + 2
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, 192, 1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, 4 if inlier else 3, 3, padding=1)
        # Start from a constant prediction: sigma = 0.5 px, no correlation, p = 0.95.
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([-0.7, -0.7, 0., 3.])[:self.out.out_channels])
        self.register_buffer("outlier_var", torch.tensor(outlier_sigma ** 2))
        # What consumers get from the inlier head (see `effective`): 0 = the moment-matched covariance,
        # tau > 0 = the inlier Sigma where p >= tau and outlier_var I elsewhere.
        self.register_buffer("gate_tau", torch.tensor(0.))

    @property
    def inlier(self) -> bool: return self.out.out_channels == 4

    def forward(self, net, motion, context, corr, flow, delta, prev_delta) -> torch.Tensor:
        corr = corr.float()
        x = torch.cat([net.float(), motion.float(), context.float(), corr, correlation_peak_stats(corr),
                       flow / 8, delta, prev_delta], dim=1)
        return self.out(self.body(x))

    def effective(self, c: torch.Tensor) -> torch.Tensor:
        """
        Raw output -> Cholesky parameters of the covariance consumers use. For the inlier head either the
        moment-matched p Sigma + (1 - p) outlier_var I (gate_tau = 0), or gated: the inlier Sigma where
        p >= gate_tau and outlier_var I elsewhere, so the pose graph sees Sigma uninflated and the selector
        drops likely outliers. Both are computed and selected with tensor ops (no host sync, graph-capturable).
        """
        if not self.inlier:
            return c
        e, p = cov_entries(c[:, :3]), torch.sigmoid(c[:, 3:4])
        uu = p * e[:, 0:1] + (1 - p) * self.outlier_var
        vv = p * e[:, 1:2] + (1 - p) * self.outlier_var
        l11 = uu.sqrt()
        l21 = p * e[:, 2:3] / l11
        l22 = (vv - l21 * l21).clamp(min=1e-12).sqrt()
        moment = torch.cat([l11.log(), l22.log(), l21], dim=1)
        log_sigma_out = 0.5 * self.outlier_var.log() * torch.ones_like(p)
        gated = torch.where(p >= self.gate_tau, c[:, :3], torch.cat([log_sigma_out, log_sigma_out, torch.zeros_like(p)], dim=1))
        return torch.where(self.gate_tau > 0, gated, moment)


class MemoryCovDecoder(MemoryDecoder):
    def __init__(self, cfg, decoder_dtype: torch.dtype):
        super(MemoryCovDecoder, self).__init__(cfg)
        # cov_head: "recurrent" (shipped: second GRU + CovHead every iteration) or "single_shot"
        if getattr(self.cfg, "cov_head", "recurrent") == "single_shot":
            self.ss_head = SingleShotCovHead(inlier=getattr(self.cfg, "inlier", False),
                                             outlier_sigma=getattr(self.cfg, "outlier_sigma", 32.))
        else:
            self.cov_update = CovUpdateBlock(self.cfg, hidden_dim=128)

        self.decoder_dtype = decoder_dtype

        self.delta              = self.delta.to(dtype=self.decoder_dtype)
        self.att                = self.att.to(dtype=self.decoder_dtype)
        self.decoder_layer      = self.decoder_layer.to(dtype=self.decoder_dtype)
        self.flow_token_encoder = self.flow_token_encoder.to(dtype=self.decoder_dtype)
        self.update_block       = self.update_block.to(dtype=self.decoder_dtype)
        # cov_fp32: the recurrent covariance branch keeps fp32 weights while the rest of the decoder runs in
        # decoder_dtype - deployment-precision training: the frozen flow path computes as deployed, the trained
        # branch keeps fp32 master weights (FusedGRUPair casts them to the activation dtype).
        if self.recurrent and not getattr(self.cfg, "cov_fp32", False):
            self.cov_update     = self.cov_update.to(dtype=self.decoder_dtype)
        # Diagnostics: when set to a list, forward appends the mean |delta_flow| of every
        # iteration. Leave None for export.
        self.iter_log: list | None = None

    @property
    def recurrent(self) -> bool: return hasattr(self, "cov_update")

    @property
    def cov_dtype(self) -> torch.dtype:
        """Weight dtype of the recurrent covariance branch (decoder_dtype unless cov_fp32)."""
        return self.cov_update.cov_head.conv1.weight.dtype

    @property
    def cov_cues(self) -> bool:
        """Recurrent head reads matching-confidence cues (follows conv1's width, so it tracks the checkpoint)."""
        return self.recurrent and self.cov_update.cov_head.conv1.in_channels > self.cov_update.gru.convz1.out_channels

    @property
    def log_sigma_output(self) -> bool:
        """The shipped layout, 2 channels of log sigma per axis; every other head emits Cholesky parameters (cov_entries)."""
        return self.recurrent and self.cov_update.cov_head.conv4.out_channels == 2

    def cov_parameters(self):
        """Parameters of the covariance branch (what --training_mode cov trains)."""
        return (self.cov_update if self.recurrent else self.ss_head).parameters()

    def use_single_shot(self, inlier: bool) -> None:
        """Swap the recurrent branch for a fresh SingleShotCovHead on the same device; caller loads its weights."""
        device = self.proj.weight.device
        del self.cov_update
        self.ss_head = SingleShotCovHead(inlier=inlier).to(device)

    def forward(self, cost_memory, context, cost_maps):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_memory = cost_memory.to(dtype=self.decoder_dtype)

        flow_coords0, flow_coords1 = initialize_flow(context)
        # Recurrent head: c accumulates the per-iteration updates. The 2-channel head keeps its original form
        # (offsets on the pixel grid, differenced at the end) so existing engines and parity references stay exact.
        if self.recurrent:
            cov_dim = self.cov_update.cov_head.conv4.out_channels
            if cov_dim == 2:
                cov_coords0, cov_coords1 = flow_coords0, flow_coords1.clone()
            else:
                cov_coords0 = torch.zeros_like(flow_coords0[:, :1]).repeat(1, cov_dim, 1, 1)
                cov_coords1 = cov_coords0
        last_delta = torch.zeros_like(flow_coords0)

        flow_predictions = []
        cov_predictions = []

        context   = self.proj(context)
        flow_net, flow_inp  = torch.split(context, [128, 128], dim=1)
        flow_net = flow_net.tanh().to(dtype=self.decoder_dtype)
        fcov_net = flow_net.clone().to(dtype=self.decoder_dtype) if self.recurrent else None
        flow_inp = flow_inp.relu()

        flow_inp  = flow_inp.to(dtype=self.decoder_dtype)
        attention = self.att(flow_inp)

        size = flow_net.shape
        key, value = None, None
        # Built per forward from this module's own GRUs: nn.DataParallel replicas copy submodules,
        # not plain attributes, so a pair stored in __init__ would keep pointing at GPU 0's weights.
        fused_gru = FusedGRUPair(self.update_block.gru, self.cov_update.gru if self.recurrent else None)
        gru_pre = fused_gru.precompute(flow_inp)

        for i in range(self.depth):
            # Only the last iteration's upsampled flow / cov are returned in
            # eval mode (see the `else` branch below). Compute mask + upsample
            # only for that iteration; state-propagation ops (gru, flow_head,
            # cov_head, coord updates) still run every iteration unchanged.
            # The single-shot head is supervised on the final iteration only, also in training.
            needs_upsample = (self.training and self.recurrent) or (i == self.depth - 1)

            flow_coords1      = flow_coords1.detach()
            bf16_flow_coords1 = flow_coords1.to(dtype=self.decoder_dtype)
            flow              = (flow_coords1 - flow_coords0).to(dtype=self.decoder_dtype)

            cost_forward = self.encode_flow_token(cost_maps, flow_coords1)
            cost_forward = cost_forward.to(dtype=self.decoder_dtype)

            query = self.flow_token_encoder(cost_forward)
            query = query.permute(0, 2, 3, 1).view(size[0] * size[2] * size[3], 1, self.dim)

            cost_global, key, value = self.decoder_layer(
                query, key, value, cost_memory, bf16_flow_coords1, size, self.cfg.query_latent_dim
            )
            corr = torch.cat([cost_global, cost_forward], dim=1)

            motion_feat        = self.update_block.encoder(flow, corr)
            motion_feat_global = self.update_block.aggregator(attention, motion_feat)

            x_ab = torch.cat([motion_feat, motion_feat_global], dim=1)
            flow_net, fcov_net = fused_gru.step(flow_net, fcov_net, x_ab, gru_pre)
            delta_flow = self.update_block.flow_head(flow_net)
            if self.cov_cues:
                # Ambiguity of this iteration's 9x9 correlation window and the size of its flow update
                cues = torch.cat([correlation_peak_stats(cost_forward), delta_flow.float()], dim=1).to(fcov_net.dtype)
                delta_cov = self.cov_update.cov_head(torch.cat([fcov_net, cues], dim=1).to(self.cov_dtype))
            elif self.recurrent:
                delta_cov = self.cov_update.cov_head(fcov_net.to(self.cov_dtype))

            # State propagation must happen every iteration so the next iter
            # consumes the updated coords (matches the original code).
            delta_flow_fp32 = delta_flow.to(dtype=torch.float32)
            flow_coords1    = flow_coords1 + delta_flow_fp32
            prev_delta, last_delta = last_delta, delta_flow_fp32
            if self.iter_log is not None:
                self.iter_log.append(delta_flow_fp32.norm(dim=1).mean().item())

            if self.recurrent:
                delta_cov_fp32 = delta_cov.to(dtype=torch.float32)
                cov_coords1    = cov_coords1 + delta_cov_fp32

            if needs_upsample:
                up_mask = self.update_block.mask(flow_net)
                up_mask = 0.25 * up_mask.to(dtype=torch.float32)
                flow_up = self.upsample_flow(flow_coords1 - flow_coords0, up_mask)
                flow_predictions.append(flow_up)

                if self.recurrent:
                    cov_mask = self.cov_update.forward_mask(fcov_net.to(self.cov_dtype)).to(dtype=torch.float32)
                    cov_up   = self.upsample_flow(cov_coords1 - cov_coords0, cov_mask)
                    cov_predictions.append(cov_up)

        if not self.recurrent:
            flow_low = flow_coords1 - flow_coords0
            corr     = self.encode_flow_token(cost_maps, flow_coords1)          # 9x9 window at the final flow
            cov_low  = self.ss_head(flow_net, x_ab, flow_inp, corr, flow_low, last_delta, prev_delta)
            cov_up   = self.upsample_flow(cov_low, up_mask, scale=1.)
            if self.training:
                return flow_predictions, [cov_up]
            return (flow_predictions[-1], flow_low), (self.ss_head.effective(cov_up), self.ss_head.effective(cov_low))

        if self.training:
            return flow_predictions, cov_predictions
        else:
            return (flow_predictions[-1], flow_coords1 - flow_coords0), (cov_predictions[-1], cov_coords1 - cov_coords0)
