import torch
import torch_tensorrt as trt
from Module.Network import FlowFormerCov
from Module.Network.FlowFormer.configs.submission import get_cfg
from Module.Network.FlowFormerCov import build_flowformer
    
if __name__ == "__main__":
    from Module.Network.FlowFormer.configs.submission import get_cfg
    from Module.Network.FlowFormerCov import build_flowformer
        
    cfg = get_cfg()
    cfg.latentcostformer.decoder_depth = 12
    model = build_flowformer(cfg, torch.float16, torch.bfloat16)
    device = torch.device("cuda")

    ckpt  = torch.load("./Model/MACVO_FrontendCov.pth", map_location=device, weights_only=True)
    
    model.eval()
    model.to(device)
    model.load_ddp_state_dict(ckpt)

    torch.backends.cuda.matmul.allow_tf32 = True    # Allow tensor cores
    torch.backends.cudnn.allow_tf32 = True          # Allow tensor cores
    torch.set_float32_matmul_precision("medium")    # Reduced precision for higher throughput
    torch.backends.cuda.preferred_linalg_library = "cusolver"   # For faster linalg ops
    
    
    H = W = 704
    batch_size = 2  # Use 2 for export, but allow dynamic batch
    channels = 3
    #inputs = [torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda(),
    #          torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()]
    inpA = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()
    inpB = torch.randn((batch_size, channels, H, W), dtype=torch.float32).cuda()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inpA, inpB),
            opset_version=18,
            f="MACVO_FrontendCov.onnx",
            input_names=["image_1", "image_2"], 
            output_names=["flow", "flow_raw", "cov", "cov_raw"],
            #dynamic_axes=None,
            #dynamic_axes={
            #"image_1": {0: "batch"},
            #"image_2": {0: "batch"},
            #"flow": {0: "batch"},
            #"flow_raw": {0: "batch"},
            #"cov": {0: "batch"},
            #"cov_raw": {0: "batch"}
            #},
            export_params=True,
        )
  # trtexec --onnx=MACVO_FrontendCov.onnx --saveEngine=MACVO_FrontendCov.plan --useCudaGraph --useSpinWait --fp16 --minShapes=image_1:1x3x704x704,image_2:1x3x704x704 --maxShapes=image_1:2x3x704x704,image_2:2x3x704x704 --optShapes=image_1:2x3x704x704,image_2:2x3x704x704