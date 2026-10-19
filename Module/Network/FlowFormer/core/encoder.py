import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .twins import Block
from .utils import coords_grid
from .twins_svt import TwinsSVTLarge
from .attention import BroadMultiHeadAttention, MultiHeadAttention, LinearPositionEmbeddingSine
from .pad_helper import pad_constant_zero


def s2d_weight(conv: nn.Conv2d) -> torch.Tensor:
    """
    Weights of a 6x6 / stride-2 / pad-2 conv, [O, C, 6, 6], rearranged for the equivalent
    3x3 / stride-1 / pad-1 conv over the pixel_unshuffle(2) input, [O, C*4, 3, 3] with input
    channel c*4 + dy*2 + dx.

    out[o] = sum_t W[t] x[2o - 2 + t]; writing the input index as 2i + dy (cell i, phase dy)
    gives t = 2*ky + dy and i = o - 1 + ky, i.e. a 3-tap conv over cells. The stride-2
    1-channel form maps to very slow conv kernels in TensorRT.
    """
    w = conv.weight
    O, C = w.shape[:2]
    return w.view(O, C, 3, 2, 3, 2).permute(0, 1, 3, 5, 2, 4).reshape(O, C * 4, 3, 3)


def conv3x3_cells_to_s2d(x: torch.Tensor, w3: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    3x3/s1/p1 conv over a cell grid (h, w even) whose output is produced directly in
    pixel_unshuffle(2) order: [N, O*4, h/2, w/2] with channel o*4 + dy*2 + dx. Implemented as a
    4x4/s2/p1 conv: output cell j, phase dy reads input cells 2j+dy-1 .. 2j+dy+1, so tap
    ty = dy + ky. 1.78x the MACs of the 3x3 form (zero taps) but no transpose of the large
    intermediate activation between PatchEmbed stages.
    """
    O, Cin = w3.shape[:2]
    parts = [pad_constant_zero(w3, (dx, 1 - dx, dy, 1 - dy)) for dy in (0, 1) for dx in (0, 1)]
    w4 = torch.stack(parts, 1).reshape(O * 4, Cin, 4, 4)
    b4 = bias.repeat_interleave(4)
    return F.conv2d(x, w4, b4, stride=2, padding=1)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64):
        super().__init__()
        self.patch_size = 8
        self.dim        = embed_dim
        self.proj       = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//4, kernel_size=6, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=6, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=6, stride=2, padding=2),
        )

        self.ffn_with_coord = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1)
        )
        self.norm = nn.LayerNorm(embed_dim*2)

    def forward(self, x) -> tuple[torch.Tensor, list[int]]:
        """Reference (token-major) implementation; forward_cf is the one used at inference."""
        B, C, H, W = x.shape    # C == 1

        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = pad_constant_zero(x, (pad_l, pad_r, pad_t, pad_b))

        x = self.proj(x)
        out_size = x.shape[2:]

        # The encoding is identical for every one of the B (= batch*H1*W1) cost maps:
        # build it once and broadcast. Static H/W folds the sin/cos into constants in ONNX.
        patch_coord = coords_grid(
            1, out_size[0], out_size[1], x.device, x.dtype
        ) * self.patch_size + (self.patch_size / 2) # in feature coordinate space
        patch_coord = patch_coord.view(1, 2, -1).permute(0, 2, 1)
        patch_coord_enc = LinearPositionEmbeddingSine(patch_coord, dim=self.dim)
        patch_coord_enc = patch_coord_enc.permute(0, 2, 1).view(1, -1, out_size[0], out_size[1])

        x_pe = torch.cat([x, patch_coord_enc.expand(B, -1, -1, -1)], dim=1)
        x = self.ffn_with_coord(x_pe)
        x = self.norm(x.flatten(2).transpose(1, 2))

        return x, out_size

    def forward_cf(self, x) -> tuple[torch.Tensor, list[int]]:
        """
        Same function as forward(), but returns the normalised tokens channels-first,
        [B*H1*W1, 2*dim, h*w], so the large token tensor is never transposed. LayerNorm is
        written out over dim 1.
        """
        B, C, H, W = x.shape
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = pad_constant_zero(x, (0, pad_r, 0, pad_b))
        # == relu(proj0) / relu(proj2) / proj4, each 6x6/s2/p2. One pixel_unshuffle on the input,
        # then every stage emits its output already in the next stage's unshuffled order.
        x = F.pixel_unshuffle(x, 2)                                                   # [N, 4, H/2, W/2]
        x = F.relu(conv3x3_cells_to_s2d(x, s2d_weight(self.proj[0]), self.proj[0].bias))   # [N, 64, H/4, W/4]
        x = F.relu(conv3x3_cells_to_s2d(x, s2d_weight(self.proj[2]), self.proj[2].bias))   # [N, 128, H/8, W/8]
        x = F.conv2d(x, s2d_weight(self.proj[4]), self.proj[4].bias, padding=1)            # [N, 64, H/8, W/8]
        out_size = x.shape[2:]

        patch_coord = coords_grid(1, out_size[0], out_size[1], x.device, x.dtype) * self.patch_size + (self.patch_size / 2)
        patch_coord_enc = LinearPositionEmbeddingSine(patch_coord.view(1, 2, -1).permute(0, 2, 1), dim=self.dim)
        patch_coord_enc = patch_coord_enc.permute(0, 2, 1).view(1, -1, out_size[0], out_size[1])
        # ffn_with_coord[0](cat[x, pe]) == W[:, :dim] x + (W[:, dim:] pe + b). The second term is a
        # constant [2*dim, h*w] that varies over position, so it cannot be a conv bias; as a GEMM
        # over channels-first tokens the broadcast add and the ReLU fuse into the epilogue.
        w0, b0 = self.ffn_with_coord[0].weight, self.ffn_with_coord[0].bias
        w2, b2 = self.ffn_with_coord[2].weight, self.ffn_with_coord[2].bias
        x = x.flatten(2)                                                 # [B, dim, h*w]
        pe_term = F.conv2d(patch_coord_enc, w0[:, self.dim:], b0).flatten(2)   # [1, 2*dim, h*w]
        x = F.relu(torch.matmul(w0[:, :self.dim, 0, 0], x) + pe_term)    # [B, 2*dim, h*w]
        x = torch.matmul(w2[:, :, 0, 0], x) + b2[None, :, None]
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).square().mean(dim=1, keepdim=True)
        x = (x - mu) * torch.rsqrt(var + self.norm.eps)
        x = x * self.norm.weight[None, :, None] + self.norm.bias[None, :, None]
        return x, out_size


class VerticalSelfAttentionLayer(nn.Module):
    def __init__(self, dim, cfg, num_heads=8, dropout=0.):
        super(VerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        drop_rate = dropout
        attn_drop_rate=0.

        self.local_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, sr_ratio=sr_ratio, ws=ws, vert_c_dim=cfg.vert_c_dim
        )
        self.global_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, sr_ratio=sr_ratio, ws=1, vert_c_dim=cfg.vert_c_dim
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int], context=None):
        x = self.local_block(x, size, context)
        x = self.global_block(x, size, context)

        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num +=  np.prod(param.size())

        return num


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, proj_drop=0., dropout=0.):
        super(SelfAttentionLayer, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(dim, num_heads)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = x
        x = self.norm1(x)

        q, k, v = self.q(x), self.k(x), self.v(x)

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num +=  np.prod(param.size())

        return num


class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, proj_drop=0., dropout=0.):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token):
        """
            x: [BH1W1, H3W3, D]
            Reference (token-major) implementation; forward_cf is the one used at inference.
        """
        short_cut = query
        query = self.norm1(query)
        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        x = self.multi_head_attn(q, k, v)
        x = short_cut + self.proj_drop(self.proj(x))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

    def forward_cf(self, query: torch.Tensor, tgt_cf: torch.Tensor) -> torch.Tensor:
        """
        Same function as forward() for a constant query (the learned latent tokens, [1, Q, C])
        and channels-first targets tgt_cf [N, D, T]. Re-associated so the K and V projections
        of the N*T target tokens are never materialised:
            scores_qh = (q_h Wk_h) . xn            (q_h bk_h is constant over keys: softmax-invariant)
            out_qh    = (A_qh . xn) Wv_h^T + bv_h   (rows of A sum to 1)
        Exact in real arithmetic; fp rounding differs from forward().
        """
        N = tgt_cf.shape[0]
        heads = self.multi_head_attn.heads
        Q = query.shape[1]
        qk_dim = self.q.out_features
        hd = qk_dim // heads
        q = self.q(self.norm1(query))                                   # [1, Q, qk_dim]
        q = q.view(Q, heads, hd).permute(1, 0, 2)                       # [h, Q, hd]
        wk = self.k.weight.view(heads, hd, -1)                          # [h, hd, D]
        qp = torch.matmul(q, wk).reshape(heads * Q, -1)                 # [h*Q, D]
        scale = hd ** -0.5
        scores = torch.matmul(qp * scale, tgt_cf)                       # [N, h*Q, T]
        attn = scores.softmax(dim=-1)
        z = torch.matmul(attn, tgt_cf.transpose(1, 2))                  # [N, h*Q, D]
        z = z.view(N, heads, Q, -1)
        wv = self.v.weight.view(heads, hd, -1)                          # [h, hd, D]
        out = torch.matmul(z, wv.transpose(1, 2))                       # [N, h, Q, hd]
        out = out + self.v.bias.view(1, heads, 1, hd)
        out = out.permute(0, 2, 1, 3).reshape(N, Q, heads * hd)         # heads concatenated as in forward()
        x = query + self.proj_drop(self.proj(out))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class CostPerceiverEncoder(nn.Module):
    def __init__(self, cfg):
        super(CostPerceiverEncoder, self).__init__()
        self.cfg = cfg
        self.patch_size: int = 8
        self.cost_heads_num: int = cfg.cost_heads_num
        self.cost_latent_token_num: int = cfg.cost_latent_token_num
        
        self.patch_embed = PatchEmbed(in_chans=self.cfg.cost_heads_num, embed_dim=cfg.cost_latent_input_dim)

        self.depth = cfg.encoder_depth

        self.latent_tokens = nn.Parameter(torch.randn(1, cfg.cost_latent_token_num, cfg.cost_latent_dim))

        query_token_dim, tgt_token_dim = cfg.cost_latent_dim, cfg.cost_latent_input_dim*2
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.input_layer = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout)
        
        self.encoder_layers = nn.ModuleList([
            SelfAttentionLayer(cfg.cost_latent_dim, dropout=cfg.dropout)
            for idx in range(self.depth)
        ])
        self.vertical_encoder_layers = nn.ModuleList([
            VerticalSelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout)
            for idx in range(self.depth)
        ])

    def forward(self, cost_volume: torch.Tensor, context=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, heads, H1, W1, H2, W2 = cost_volume.shape
        cost_maps = cost_volume.permute(0, 2, 3, 1, 4, 5).contiguous().view(B*H1*W1, self.cost_heads_num, H2, W2)
        
        x, size = self.patch_embed.forward_cf(cost_maps)        # B*H1*W1, C, size[0]*size[1]
        x = self.input_layer.forward_cf(self.latent_tokens, x)  # B*H1*W1, tokens, C

        short_cut = x

        for layer, vert_layer in zip(self.encoder_layers, self.vertical_encoder_layers):
            
            #with torch.cuda.nvtx.range("SelfAttentionLayer"):
            x = layer(x)
            x = x.view(B, H1*W1, self.cost_latent_token_num, -1).permute(0, 2, 1, 3).reshape(B*self.cost_latent_token_num, H1*W1, -1)
            
            #with torch.cuda.nvtx.range("Vertical SelfAttnLayer"):
            x = vert_layer(x, (H1, W1), context)
            x = x.view(B, self.cost_latent_token_num, H1*W1, -1).permute(0, 2, 1, 3).reshape(B*H1*W1, self.cost_latent_token_num, -1)

        x = x + short_cut
        return x, cost_maps


class MemoryEncoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg

        self.feat_encoder = TwinsSVTLarge(pretrained=self.cfg.pretrain)
        self.channel_convertor = nn.Conv2d(cfg.encoder_latent_dim, cfg.encoder_latent_dim, 1, padding=0, bias=False)
        self.cost_perceiver_encoder = CostPerceiverEncoder(cfg)

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        heads = self.cfg.cost_heads_num
        d = dim // heads  # each head's channel dim
        
        # Simplified feature map reshaping - combine operations
        fmap1 = fmap1.reshape(batch, heads, d, ht*wd).permute(0, 1, 3, 2)  # [b, heads, (ht*wd), d]
        fmap2 = fmap2.reshape(batch, heads, d, ht*wd).permute(0, 1, 3, 2)  # [b, heads, (ht*wd), d]
        
        # Reshape for bmm (keeping the efficient batch matrix multiplication)
        fmap1_bmm = fmap1.reshape(batch*heads, ht*wd, d)
        fmap2_bmm = fmap2.reshape(batch*heads, ht*wd, d)
        
        # Use bmm as in the original code
        corr_bmm = torch.bmm(fmap1_bmm, fmap2_bmm.transpose(1, 2))  # [b*heads, (ht*wd), (ht*wd)]
        
        # Simplified reshaping to final 6D format - direct reshape from bmm result
        corr = corr_bmm.view(batch, heads, ht*wd, ht*wd).reshape(batch, heads, ht, wd, ht, wd)
        
        return corr

    def forward(self, img1, img2, context):
        imgs = torch.cat([img1, img2], dim=0)
        
        #with torch.cuda.nvtx.range("Feature Encoder"):
        feats = self.feat_encoder(imgs)
        
        #with torch.cuda.nvtx.range("Channel Convertor"):
        feats = self.channel_convertor(feats)
        
        B = feats.shape[0] // 2
        feat_s, feat_t = feats[:B], feats[B:]
        
        #with torch.cuda.nvtx.range("Build Corr Volume"):
        cost_volume = self.corr(feat_s, feat_t).to(feats)
        
        #with torch.cuda.nvtx.range("Cost Perceiver Enc"):
        x, cost_maps = self.cost_perceiver_encoder(cost_volume, context)

        return x, cost_maps
