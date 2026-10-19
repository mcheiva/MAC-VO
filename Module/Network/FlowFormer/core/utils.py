import torch
import torch.nn.functional as F

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        if sum(self._pad) == 0: return inputs
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def bilinear_sampler(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for grid_sample, uses pixel coordinates (..., 2) = (x, y).
    Normalisation is written as whole-tensor elementwise ops (same operation order as the
    per-channel form, so bit-identical); in-place slice assignment exports as ONNX ScatterND.
    """
    H, W = img.shape[-2:]
    wh = torch.tensor([W - 1, H - 1], device=coords.device, dtype=coords.dtype)
    return F.grid_sample(img, (2 * coords) / wh - 1, align_corners=True)

_GRID_CACHE: dict = {}


def coords_grid(batch: int, ht: int, wd: int, device: torch.device, dtype: torch.dtype):
    """
    [batch, 2, ht, wd] pixel coordinates (x, y). Built once per (ht, wd) with numpy and
    cached: inside torch.onnx tracing the cached tensor is a constant, so the sine position
    embeddings derived from it fold into initializers instead of Range/Sin/Cos at runtime.
    """
    import numpy as np
    key = (int(ht), int(wd), str(device), dtype)
    g = _GRID_CACHE.get(key)
    if g is None:
        ys, xs = np.meshgrid(np.arange(int(ht), dtype=np.float32), np.arange(int(wd), dtype=np.float32), indexing="ij")
        # torch.tensor(..., device=) creates the constant on the device directly; a traced .to() on a
        # CPU constant makes torch.onnx's constant folder mix cpu/cuda tensors and fail.
        g = torch.tensor(np.stack((xs, ys), axis=0), device=device, dtype=dtype)
        _GRID_CACHE[key] = g
    return g.unsqueeze(0).repeat(batch, 1, 1, 1)
