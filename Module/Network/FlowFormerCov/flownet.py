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

    COV_CONV1  = "memory_decoder.cov_update.cov_head.conv1"
    COV_CONV4  = "memory_decoder.cov_update.cov_head.conv4"
    COV_UPDATE = "memory_decoder.cov_update."
    SS_HEAD    = "memory_decoder.ss_head."

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        """
        Strict load, except that the covariance head follows the checkpoint:
        * 2-channel checkpoint into a 3-channel (Cholesky) model: rows c1, c2 are copied and the c3 row is
          zero, so the model starts exactly at the shipped diagonal solution (L21 = 0).
        * 3-channel checkpoint into a 2-channel model (e.g. a frontend built from the default config): the
          head is switched to 3 channels.
        * Single-shot checkpoint into a recurrent model: the decoder switches to the single-shot head.
        * Recurrent checkpoint into a single-shot model: the recurrent branch is dropped and the single-shot
          head starts fresh; everything else still loads strictly.
        * Checkpoint without cov_cues into a cue model: the cue inputs of conv1 start at zero (exactly the
          checkpoint's head); a cue checkpoint into a model without cues switches conv1 to the cue width.
        """
        cvt_ckpt = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in ckpt.items())
        for k in self.UNUSED_CKPT_KEYS:
            cvt_ckpt.pop(k, None)

        decoder = self.memory_decoder
        if decoder.recurrent and any(k.startswith(self.SS_HEAD) for k in cvt_ckpt):
            decoder.use_single_shot(inlier=cvt_ckpt[self.SS_HEAD + "out.weight"].size(0) == 4)
            print("[FlowFormerCov] covariance head switched to single-shot to match the checkpoint")
        if not decoder.recurrent:
            cvt_ckpt = OrderedDict((k, v) for k, v in cvt_ckpt.items() if not k.startswith(self.COV_UPDATE))
            missing, unexpected = self.load_state_dict(cvt_ckpt, strict=False)
            assert not unexpected and all(k.startswith(self.SS_HEAD) for k in missing), (missing, unexpected)
            if self.SS_HEAD + "out.weight" in missing:
                print("[FlowFormerCov] single-shot covariance head initialised fresh (not in the checkpoint)")
            return

        head = decoder.cov_update.cov_head
        w1 = cvt_ckpt[self.COV_CONV1 + ".weight"]
        if w1.size(1) > head.conv1.in_channels:
            head.set_in_dim(w1.size(1))
            print("[FlowFormerCov] covariance head switched to matching-confidence cues to match the checkpoint")
        elif w1.size(1) < head.conv1.in_channels:
            extra = w1.new_zeros(w1.size(0), head.conv1.in_channels - w1.size(1), *w1.shape[2:])
            cvt_ckpt[self.COV_CONV1 + ".weight"] = torch.cat([w1, extra], dim=1)
            print("[FlowFormerCov] covariance-head cue inputs initialised at zero (checkpoint has no cues)")
        w, b = cvt_ckpt[self.COV_CONV4 + ".weight"], cvt_ckpt[self.COV_CONV4 + ".bias"]
        if w.size(0) == 3 and head.conv4.out_channels == 2:
            head.set_out_dim(3)
            print("[FlowFormerCov] covariance head switched to 3 channels (Cholesky) to match the checkpoint")
        elif w.size(0) == 2 and head.conv4.out_channels == 3:
            cvt_ckpt[self.COV_CONV4 + ".weight"] = torch.cat([w, torch.zeros_like(w[:1])], dim=0)
            cvt_ckpt[self.COV_CONV4 + ".bias"]   = torch.cat([b, torch.zeros_like(b[:1])], dim=0)
            print("[FlowFormerCov] Cholesky covariance head initialised from a 2-channel checkpoint (c3 = 0)")
        self.load_state_dict(cvt_ckpt, strict=True)

