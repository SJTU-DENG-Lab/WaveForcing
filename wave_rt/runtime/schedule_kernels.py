"""RF pointwise fusion; preserve eager precision, layout and RNG inputs."""
import torch
import triton
import triton.language as tl


@triton.jit
def _offset(index, SHAPE: tl.constexpr, STRIDES: tl.constexpr, OUT_STRIDES: tl.constexpr):
    offset = tl.full(index.shape, 0, tl.int32)
    for axis in tl.static_range(4):
        offset += (index // OUT_STRIDES[axis] % SHAPE[axis]) * STRIDES[axis]
    return offset


@triton.jit
def _x0(Prediction, Noisy, Times, Sigmas, Out,
        SIZE: tl.constexpr, SHAPE: tl.constexpr, PS: tl.constexpr,
        NS: tl.constexpr, OS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = index < SIZE
    frame = index // OS[0] % SHAPE[0]
    step = tl.load(Times+frame, valid, other=0)
    sigma = tl.load(Sigmas+step, valid, other=0).to(tl.float64)
    prediction = tl.load(Prediction+_offset(index, SHAPE, PS, OS), valid, other=0).to(tl.float64)
    noisy = tl.load(Noisy+_offset(index, SHAPE, NS, OS), valid, other=0).to(tl.float64)
    # Eager uses separate FP64 multiply/subtract, then a cast to the output dtype.
    value = noisy - sigma*prediction
    tl.store(Out+index, value.to(tl.float32).to(Out.dtype.element_ty), valid)


@triton.jit
def _add_noise(Original, Noise, Sigma, Out,
               SIZE: tl.constexpr, SHAPE: tl.constexpr, XS: tl.constexpr,
               NS: tl.constexpr, OS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = index < SIZE
    sigma = tl.load(Sigma).to(tl.float32)
    original = tl.load(Original+_offset(index, SHAPE, XS, OS), valid, other=0).to(tl.float32)
    noise = tl.load(Noise+_offset(index, SHAPE, NS, OS), valid, other=0).to(tl.float32)
    # Do not contract the two FP32 products and sum, or change the supplied RNG.
    value = (1.-sigma)*original + sigma*noise
    tl.store(Out+index, value, valid)


def x0(prediction, noisy, timestep, sigmas):
    out = torch.empty_like(noisy, dtype=prediction.dtype)
    _x0[(triton.cdiv(out.numel(), 256),)](
        prediction, noisy, timestep, sigmas, out, out.numel(), tuple(out.shape),
        prediction.stride(), noisy.stride(), out.stride(), 256,
        num_warps=4, enable_fp_fusion=False)
    return out


def add_noise(original, noise, sigma):
    out = torch.empty_like(original, dtype=noise.dtype)
    _add_noise[(triton.cdiv(out.numel(), 256),)](
        original, noise, sigma, out, out.numel(), tuple(out.shape),
        original.stride(), noise.stride(), out.stride(), 256,
        num_warps=4, enable_fp_fusion=False)
    return out
