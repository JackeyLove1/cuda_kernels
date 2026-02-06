import time
import argparse

import torch
from torch.utils.cpp_extension import load

import os

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
_DIR = os.path.dirname(__file__)
print("DIR: ", _DIR)

lib = load(
    name="rms_norm_lib",
    sources=[os.path.join(_DIR, "94-torch-rmsnorm.cu")],
    extra_include_paths=[
        # Needed for headers included as <include/...>
        _DIR,
        "include",
        os.path.join(_DIR, "include"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    verbose=True,
)


# un-fused naive rms norm
# x [batch, d], weight [d]
@torch.compile
def naive_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # y = x * rsqrt(mean(x^2) + eps) * weight
    s_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x * s_rms) * weight


## triton implement
import triton
import triton.language as tl

import torch

# x [batch, d], y [batch, d] weight [d]

@triton.jit
def _rms_norm_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        stride_x,
        stride_y,
        eps,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptrs = x_ptr + row * stride_x
    y_ptrs = y_ptr + row * stride_y

    # RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in tl.static_range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptrs + cols, mask=(cols < N), other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + eps)

    for off in tl.static_range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptrs + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(y_ptrs + cols, y, mask=mask)

def triton_rms_norm(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [batch, d], got shape={tuple(x.shape)}")
    if weight.dim() != 1:
        raise ValueError(f"weight must be 1D [d], got shape={tuple(weight.shape)}")
    if x.size(1) != weight.numel():
        raise ValueError(
            f"weight.numel() must equal x.size(1); got {weight.numel()} vs {x.size(1)}"
        )
    if x.stride(1) != 1:
        raise ValueError("x must be contiguous in the last dimension (stride(1)==1)")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    y = torch.empty_like(x)
    batch, N = x.shape
    grid = lambda meta: (batch,)
    _rms_norm_kernel[grid](
        x,
        y,
        weight,
        x.stride(0),
        y.stride(0),
        eps,
        N=N,
        BLOCK_SIZE=1024,
    )
    return y

def ext_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Prefer the canonical binding, fallback to the older alias name.
    if hasattr(lib, "rms_norm"):
        return lib.rms_norm(x, weight, eps)
    return lib.rms_norm_lib(x, weight, eps)


def bench(fn: callable, warmup: int, iters: int):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    return out, ms


def _fmt_out(out: torch.Tensor) -> str:
    vals = out.flatten()[:3].detach().float().cpu().tolist()
    vals = [round(v, 8) for v in vals]
    vals = [f"{v:<12}" for v in vals]
    return str(vals)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-5)
    # Keep CLI backward-compatible, but the extension is fp32-only in this test build.
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.dtype != "fp32":
        print(f"[note] extension is fp32-only; overriding --dtype {args.dtype} -> fp32")
        args.dtype = "fp32"
    dtype = torch.float32

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn((args.N, args.K), device="cuda", dtype=dtype)
    w = torch.randn((args.K,), device="cuda", dtype=dtype)

    print("-" * 85)
    print(f"N={args.N} K={args.K} dtype={args.dtype} eps={args.eps} iters={args.iters} warmup={args.warmup}")

    # Correctness (fp32 baseline)
    y_ref_fp32 = naive_rms_norm(x, w, eps=float(args.eps))
    y_ext = ext_rms_norm(x, w, eps=float(args.eps))
    max_abs = (y_ref_fp32 - y_ext).abs().max().item()
    denom = y_ref_fp32.abs().max().item()
    rel = max_abs / (denom + 1e-12)
    print(f"correctness: max_abs={max_abs:.3e}, rel={rel:.3e}")

    y_triton = triton_rms_norm(x, w, eps=float(args.eps))
    max_abs_t = (y_ref_fp32 - y_triton).abs().max().item()
    denom_t = y_ref_fp32.abs().max().item()
    rel_t = max_abs_t / (denom_t + 1e-12)
    print(f"correctness(triton): max_abs={max_abs_t:.3e}, rel={rel_t:.3e}")

    if args.show_all:
        print("ref(fp32) =", y_ref_fp32)
        print("ext(fp32) =", y_ext)
        if y_triton is not None:
            print("triton(fp32) =", y_triton)

    # Benchmark
    if args.compile_baseline:
        try:
            baseline = torch.compile(naive_rms_norm, fullgraph=True)
            baseline_name = "naive_compile"
        except Exception:
            baseline = naive_rms_norm
            baseline_name = "torch_compile_rms_norm (compile failed)"
    else:
        baseline = naive_rms_norm
        baseline_name = "torch_compile_rms_norm"

    out_base, ms_base = bench(lambda: baseline(x, w, float(args.eps)), warmup=args.warmup, iters=args.iters)
    out_ext, ms_ext = bench(lambda: ext_rms_norm(x, w, float(args.eps)), warmup=args.warmup, iters=args.iters)
    out_tri, ms_tri = bench(lambda: triton_rms_norm(x, w, float(args.eps)), warmup=args.warmup, iters=args.iters)

    print(f"{baseline_name:>16}: out={_fmt_out(out_base)}, time={ms_base:.6f} ms")
    print(f"{'cuda_rms_norm':>16}: out={_fmt_out(out_ext)}, time={ms_ext:.6f} ms")
    print(f"{'triton_rms_norm':>16}: out={_fmt_out(out_tri)}, time={ms_tri:.6f} ms")
    if ms_ext > 0:
        print(f"{'speedup':>16}: {ms_base / ms_ext:.3f}x")
    if ms_tri > 0:
        print(f"{'speedup(triton)':>16}: {ms_base / ms_tri:.3f}x")
    if ms_ext > 0 and ms_tri > 0:
        print(f"{'ext_vs_triton':>16}: {ms_tri / ms_ext:.3f}x (triton/ext)")


if __name__ == "__main__":
    main()
