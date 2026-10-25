import torch
from collections import OrderedDict

from ..FlowFormer.core.utils import InputPadder
from ..FlowFormer.core.transformer import FlowFormer
from .covhead import MemoryCovDecoder, cov_entries


class FlowFormerCov(FlowFormer):
    def __init__(self, cfg, encoder_dtype: torch.dtype=torch.float32, decoder_dtype: torch.dtype=torch.float32):
        super(FlowFormerCov, self).__init__(cfg)
        self.memory_decoder = MemoryCovDecoder(self.cfg, decoder_dtype)
        
        self.enc_dtype = encoder_dtype
        self.context_encoder = self.context_encoder.to(dtype=self.enc_dtype) 
        self.memory_encoder  = self.memory_encoder.to(dtype=self.enc_dtype)

    def forward(self, image1, image2):
        image1 = ((2 * image1) - 1.0).to(dtype=self.enc_dtype)
        image2 = ((2 * image2) - 1.0).to(dtype=self.enc_dtype)
        # if not torch.compiler.is_compiling():
        #     torch.cuda.nvtx.range_push("Context Encoder")
        context = self.context_encoder(image1)
        #if not torch.compiler.is_compiling():
        #    torch.cuda.nvtx.range_pop()

        #if not torch.compiler.is_compiling():
        #    torch.cuda.nvtx.range_push("Memory Encoder")
        cost_memory, cost_maps = self.memory_encoder(image1, image2, context)
        cost_maps = cost_maps.float()
        context   = context.float()

        #if not torch.compiler.is_compiling():
        #    torch.cuda.nvtx.range_pop()

        #if not torch.compiler.is_compiling():
        #    torch.cuda.nvtx.range_push("Memory Decoder")
        flow_predictions, cov_predictions = self.memory_decoder(cost_memory, context, cost_maps)

        #if not torch.compiler.is_compiling():
        #    torch.cuda.nvtx.range_pop()

        return flow_predictions, cov_predictions
    
    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image1: torch.Tensor, image2: torch.Tensor):
        padder = InputPadder(image1.shape)
        image1, image2    = padder.pad(image1, image2)
        flow_pre, cov_pre = self.forward(image1, image2)
        flow_pre = padder.unpad(flow_pre[0])
        cov_pre = padder.unpad(cov_pre[0])
        
        # [sigma_uu, sigma_vv] for the shipped head, [sigma_uu, sigma_vv, sigma_uv] for the Cholesky head
        return flow_pre, cov_entries(cov_pre)

    # Checkpoint tensors the inference graph does not consume: the Twins final norms and the
    # GMA relative position embedding. Everything else loads strictly, so a restructured
    # module cannot silently keep random weights.
    UNUSED_CKPT_KEYS = (
        "context_encoder.svt.norm.weight", "context_encoder.svt.norm.bias",
        "memory_encoder.feat_encoder.svt.norm.weight", "memory_encoder.feat_encoder.svt.norm.bias",
        "memory_decoder.att.pos_emb.rel_ind",
        "memory_decoder.att.pos_emb.rel_height.weight", "memory_decoder.att.pos_emb.rel_width.weight",
    )

    COV_CONV4  = "memory_decoder.cov_update.cov_head.conv4"

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        """
        Strict load, except that the covariance head follows the checkpoint's channel count (2: the shipped per-axis
        log sigma, 3: the Cholesky head):
        * A 2-channel checkpoint into a 3-channel model: its rows are copied and c3 starts at 0 (L21 = 0, the
          shipped diagonal solution), so training starts from the shipped behaviour.
        * A 3-channel checkpoint into a 2-channel model (e.g. a frontend built from the default config): the head
          is switched to 3 channels.
        """
        cvt_ckpt = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in ckpt.items())
        for k in self.UNUSED_CKPT_KEYS:
            cvt_ckpt.pop(k, None)

        head = self.memory_decoder.cov_update.cov_head
        w, b = cvt_ckpt[self.COV_CONV4 + ".weight"], cvt_ckpt[self.COV_CONV4 + ".bias"]
        n_ckpt, n_model = w.size(0), head.conv4.out_channels
        if n_ckpt > n_model:
            head.set_out_dim(n_ckpt)
            print(f"[FlowFormerCov] covariance head switched to {n_ckpt} channels to match the checkpoint")
        elif n_ckpt < n_model:
            cvt_ckpt[self.COV_CONV4 + ".weight"] = torch.cat([w, w.new_zeros(n_model - n_ckpt, *w.shape[1:])], dim=0)
            cvt_ckpt[self.COV_CONV4 + ".bias"]   = torch.cat([b, b.new_zeros(n_model - n_ckpt)], dim=0)
            print(f"[FlowFormerCov] {n_model}-channel covariance head initialised from a {n_ckpt}-channel checkpoint")
        self.load_state_dict(cvt_ckpt, strict=True)

