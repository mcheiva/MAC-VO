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


def marginal_log_sigma(c: torch.Tensor) -> torch.Tensor:
    """
    Decoder output c [B, C, H, W] -> log sigma per axis [B, 2, H, W], the shipped head's layout (the consumer applies
    exp(2 * cov)). C = 2 is that already; for C = 3, log sigma_u = c1 and log sigma_v = log sqrt(L21^2 + L22^2)
    = c2 + 0.5 log1p((c3 e^-c2)^2). sigma_uv is dropped, so an engine of any head drops into the shipped runtime.
    """
    if c.size(1) == 2:
        return c
    assert c.size(1) == 3, "marginal_log_sigma takes 2 (log sigma) or 3 (Cholesky) channels"
    c1, c2, c3 = c[:, 0:1], c[:, 1:2], c[:, 2:3]
    return torch.cat([c1, c2 + 0.5 * torch.log1p((c3 * torch.exp(-c2)) ** 2)], dim=1)


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
        # cov_channels: 2 = per-axis log sigma (shipped), 3 = Cholesky factor of the full 2x2 covariance
        self.cov_head = CovHead(hidden_dim, hidden_dim=256, out_dim=getattr(args, "cov_channels", 2))
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


class MemoryCovDecoder(MemoryDecoder):
    def __init__(self, cfg, decoder_dtype: torch.dtype):
        super(MemoryCovDecoder, self).__init__(cfg)
        self.cov_update = CovUpdateBlock(self.cfg, hidden_dim=128)

        self.decoder_dtype = decoder_dtype

        self.delta              = self.delta.to(dtype=self.decoder_dtype)
        self.att                = self.att.to(dtype=self.decoder_dtype)
        self.decoder_layer      = self.decoder_layer.to(dtype=self.decoder_dtype)
        self.flow_token_encoder = self.flow_token_encoder.to(dtype=self.decoder_dtype)
        self.update_block       = self.update_block.to(dtype=self.decoder_dtype)
        # cov_fp32: the covariance branch keeps fp32 weights while the rest of the decoder runs in decoder_dtype -
        # deployment-precision training: the frozen flow path computes as deployed, the trained branch keeps fp32
        # master weights (FusedGRUPair casts them to the activation dtype).
        if not getattr(self.cfg, "cov_fp32", False):
            self.cov_update     = self.cov_update.to(dtype=self.decoder_dtype)
        # Diagnostics: when set to a list, forward appends the mean |delta_flow| of every
        # iteration. Leave None for export.
        self.iter_log: list | None = None

    @property
    def cov_dtype(self) -> torch.dtype:
        """Weight dtype of the covariance branch (decoder_dtype unless cov_fp32)."""
        return self.cov_update.cov_head.conv1.weight.dtype

    @property
    def log_sigma_output(self) -> bool:
        """The shipped layout, 2 channels of log sigma per axis; the Cholesky head emits its parameters (cov_entries)."""
        return self.cov_update.cov_head.conv4.out_channels == 2

    def cov_parameters(self):
        """Parameters of the covariance branch (what --training_mode cov trains)."""
        return self.cov_update.parameters()

    def forward(self, cost_memory, context, cost_maps):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_memory = cost_memory.to(dtype=self.decoder_dtype)

        flow_coords0, flow_coords1 = initialize_flow(context)
        # c accumulates the per-iteration updates. The 2-channel head keeps its original form (offsets on the pixel
        # grid, differenced at the end) so existing engines and parity references stay exact.
        if self.log_sigma_output:
            cov_coords0, cov_coords1 = flow_coords0, flow_coords1.clone()
        else:
            cov_coords0 = torch.zeros_like(flow_coords0[:, :1]).repeat(1, self.cov_update.cov_head.conv4.out_channels, 1, 1)
            cov_coords1 = cov_coords0

        flow_predictions = []
        cov_predictions = []
        # "covsum": the per-iteration cov deltas are summed once after the loop instead of running-added. Nothing in
        # the loop reads cov_coords1, so TRT would otherwise defer the 12 running adds into the final upsample kernel
        # and recompute the 12-term sum for each of the 9 taps of every output pixel (-0.14 ms with `upsample`
        # on top of its own gain, -0.44 ms together). Gated: eval only (training upsamples every iteration, so
        # the running value is needed), and only for the 3-channel head, whose accumulator starts at 0, so the
        # one-shot sum is exact in real arithmetic (fp32 rounding <= 5e-7). The 2-channel head accumulates on the pixel grid
        # (start value flow_coords1), where a separate sum would round differently: it keeps the running add.
        defer_cov = (not self.training) and not self.log_sigma_output
        cov_deltas = []

        context   = self.proj(context)
        flow_net, flow_inp  = torch.split(context, [128, 128], dim=1)
        flow_net = flow_net.tanh().to(dtype=self.decoder_dtype)
        fcov_net = flow_net.clone().to(dtype=self.decoder_dtype)
        flow_inp = flow_inp.relu()

        flow_inp  = flow_inp.to(dtype=self.decoder_dtype)
        attention = self.att(flow_inp)

        size = flow_net.shape
        key, value = None, None
        # Built per forward from this module's own GRUs: nn.DataParallel replicas copy submodules,
        # not plain attributes, so a pair stored in __init__ would keep pointing at GPU 0's weights.
        fused_gru = FusedGRUPair(self.update_block.gru, self.cov_update.gru)
        gru_pre = fused_gru.precompute(flow_inp)

        for i in range(self.depth):
            # Only the last iteration's upsampled flow / cov are returned in
            # eval mode (see the `else` branch below). Compute mask + upsample
            # only for that iteration; state-propagation ops (gru, flow_head,
            # cov_head, coord updates) still run every iteration unchanged.
            needs_upsample = self.training or (i == self.depth - 1)

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
            delta_cov  = self.cov_update.cov_head(fcov_net.to(self.cov_dtype))

            # State propagation must happen every iteration so the next iter
            # consumes the updated coords (matches the original code).
            delta_flow_fp32 = delta_flow.to(dtype=torch.float32)
            flow_coords1    = flow_coords1 + delta_flow_fp32
            if self.iter_log is not None:
                self.iter_log.append(delta_flow_fp32.norm(dim=1).mean().item())

            delta_cov_fp32 = delta_cov.to(dtype=torch.float32)
            if defer_cov:
                cov_deltas.append(delta_cov_fp32)
            else:
                cov_coords1 = cov_coords1 + delta_cov_fp32

            if needs_upsample:
                if defer_cov:
                    cov_coords1 = cov_coords0 + torch.stack(cov_deltas, 0).sum(0)
                up_mask = self.update_block.mask(flow_net)
                up_mask = 0.25 * up_mask.to(dtype=torch.float32)
                flow_up = self.upsample_flow(flow_coords1 - flow_coords0, up_mask)
                flow_predictions.append(flow_up)

                cov_mask = self.cov_update.forward_mask(fcov_net.to(self.cov_dtype)).to(dtype=torch.float32)
                cov_up   = self.upsample_flow(cov_coords1 - cov_coords0, cov_mask)
                cov_predictions.append(cov_up)

        if self.training:
            return flow_predictions, cov_predictions
        return (flow_predictions[-1], flow_coords1 - flow_coords0), (cov_predictions[-1], cov_coords1 - cov_coords0)
