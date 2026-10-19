import torch
import torch.nn as nn
import torch.nn.functional as F
from .gma import Aggregate

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class FusedGRUPair:
    """
    Exact re-parametrisation of two SepConvGRUs (flow and cov) that consume the same input
    x = cat[inp, a, b] (3 x 128 ch) with separate hidden states. Not an nn.Module: it only
    holds references to the two GRUs and slices their parameters at call time, so checkpoint
    keys are unchanged and training code keeps using SepConvGRU.

    conv(cat[h, x]) == conv_h(h) + conv_inp(inp) + conv_ab(cat[a, b]). `inp` is constant over
    the decoder iterations, so its term (plus the bias) is computed once per forward in
    `precompute`. Per iteration: one conv over cat[a, b] for z, r, q of both GRUs (768 out),
    and per GRU a fused z/r conv on h (256 out) and a q conv on r*h. Input-channel layout of
    every original weight [128, 512, k]: [0:128]=h, [128:256]=inp, [256:384]=a, [384:512]=b.
    Fused output layout (768): [a_z, a_r, a_q, b_z, b_r, b_q].
    Partial sums are rounded to the working dtype before the add, so reduced-precision
    results are not bit-identical to the single-conv form (fp32 agrees to rounding).
    """
    H, X0, A0 = 128, 128, 256   # channel offsets inside the 512-wide input

    def __init__(self, gru_a: SepConvGRU, gru_b: SepConvGRU):
        self.grus = (gru_a, gru_b)
        self.dirs = (("convz1", "convr1", "convq1"), ("convz2", "convr2", "convq2"))

    def _convs(self, d: int) -> list[nn.Conv2d]:
        return [getattr(g, n) for g in self.grus for n in self.dirs[d]]

    def precompute(self, inp: torch.Tensor) -> list[torch.Tensor]:
        """Loop-invariant term per direction: conv(inp, W[:, inp]) + bias -> [B, 768, H, W]."""
        out = []
        for d in range(2):
            convs = self._convs(d)
            w = torch.cat([c.weight[:, self.X0:self.A0] for c in convs], 0)
            b = torch.cat([c.bias for c in convs], 0)
            out.append(F.conv2d(inp, w, b, padding=convs[0].padding))
        return out

    def _direction(self, d: int, ha: torch.Tensor, hb: torch.Tensor, x_ab: torch.Tensor,
                   p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        convs = self._convs(d)
        pad = convs[0].padding
        w_ab = torch.cat([c.weight[:, self.A0:] for c in convs], 0)
        x = F.conv2d(x_ab, w_ab, None, padding=pad) + p
        new_h = []
        for g, (gru, h) in enumerate(zip(self.grus, (ha, hb))):
            cz, cr, cq = convs[3 * g: 3 * g + 3]
            o = 384 * g
            zr = F.conv2d(h, torch.cat([cz.weight[:, :self.H], cr.weight[:, :self.H]], 0), None, padding=pad)
            z = torch.sigmoid(zr[:, :self.H] + x[:, o:o + 128])
            r = torch.sigmoid(zr[:, self.H:] + x[:, o + 128:o + 256])
            q = torch.tanh(F.conv2d(r * h, cq.weight[:, :self.H], None, padding=pad) + x[:, o + 256:o + 384])
            new_h.append((1 - z) * h + z * q)
        return new_h[0], new_h[1]

    def step(self, ha: torch.Tensor, hb: torch.Tensor, x_ab: torch.Tensor,
             pre: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU update of both hidden states. x_ab = cat[a, b]; pre from `precompute`."""
        ha, hb = self._direction(0, ha, hb, x_ab, pre[0])   # horizontal (1x5)
        ha, hb = self._direction(1, ha, hb, x_ab, pre[1])   # vertical   (5x1), same x
        return ha, hb


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = 81 + args.query_latent_dim
        
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
