"""ONNX-friendly constant-zero padding that propagates input dtype.

`F.pad(x, pad, mode='constant', value=0)` exports as ONNX Pad with an empty
constant_value input that defaults to fp32. Under TRT mixed-precision builds
(polygraphy `debug precision`, OBEY/PREFER constraints), the myelin codegen
demands fill_value match the runtime layer dtype and aborts with
`Slice operation "...Pad_slice" has incorrect fill value type`.

This helper rebuilds the same operation as `torch.cat`-with-typed-zeros so
the fill is a tensor of `x.dtype`, dodging the myelin issue.

The zeros are a one-element tensor EXPANDED to the pad shape, not a dense
`torch.zeros(shape)`. A dense zeros exports (in a plain static trace) as a
literal all-zero ONNX Constant, and Myelin then generates a ~3.7x slower kernel
for the decoder's convex upsampler (it cannot fuse a dense constant as an
in-kernel broadcast). The expand exports as a runtime Expand / ConstantOfShape
that TensorRT keeps as a zero-stride broadcast, which is what a dynamic-axis
trace gave before. Values and dtype are identical.
"""
import torch


def _zeros(x: torch.Tensor, shape) -> torch.Tensor:
    """Zeros of `shape`, dtype/device of `x`, as a broadcast of one element."""
    return x.new_zeros(1).expand(shape)


def pad_constant_zero(x: torch.Tensor, pad) -> torch.Tensor:
    """Equivalent to `F.pad(x, pad, mode='constant', value=0)` but the zero
    fill is a one-element `x.dtype` zero expanded to the pad shape.

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
            pieces.append(_zeros(x, shape))
        pieces.append(x)
        if right > 0:
            shape = list(x.shape)
            shape[dim] = right
            pieces.append(_zeros(x, shape))
        x = torch.cat(pieces, dim=dim)
    return x
