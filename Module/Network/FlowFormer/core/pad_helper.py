"""ONNX-friendly constant-zero padding that propagates input dtype.

`F.pad(x, pad, mode='constant', value=0)` exports as ONNX Pad with an empty
constant_value input that defaults to fp32. Under TRT mixed-precision builds
(polygraphy `debug precision`, OBEY/PREFER constraints), the myelin codegen
demands fill_value match the runtime layer dtype and aborts with
`Slice operation "...Pad_slice" has incorrect fill value type`.

This helper rebuilds the same operation as `torch.cat`-with-typed-zeros so
ONNX bakes a Constant of `x.dtype`, dodging the myelin issue.
"""
import torch


def pad_constant_zero(x: torch.Tensor, pad) -> torch.Tensor:
    """Equivalent to `F.pad(x, pad, mode='constant', value=0)` but the zero
    fill is a `torch.zeros(dtype=x.dtype)` Constant in the exported graph.

    Args:
        x:   input tensor.
        pad: even-length sequence of ints; same convention as F.pad
             (pairs for trailing dims, last dim first).
    """
    n_pad_dims = len(pad) // 2
    for i in range(n_pad_dims):
        dim = -1 - i
        left = int(pad[2 * i])
        right = int(pad[2 * i + 1])
        if left == 0 and right == 0:
            continue
        pieces = []
        if left > 0:
            shape = list(x.shape)
            shape[dim] = left
            pieces.append(torch.zeros(shape, dtype=x.dtype, device=x.device))
        pieces.append(x)
        if right > 0:
            shape = list(x.shape)
            shape[dim] = right
            pieces.append(torch.zeros(shape, dtype=x.dtype, device=x.device))
        x = torch.cat(pieces, dim=dim)
    return x
