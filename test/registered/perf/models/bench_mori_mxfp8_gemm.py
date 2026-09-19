#!/usr/bin/env python3
"""mori's mxfp8 GEMM against the native linear it would replace.

Single process: this path has no collective. What it replaces is
`mxfp8_native_blockscaled_linear`, whose own route changes with M --
`native_route_plan` picks a GEMV shape at small M, `dot_scaled` in the middle
and `hipblaslt_bf16` at the top -- so the baseline is not one kernel and the
curve is not smooth. That is the curve `_MIN_M` has to be read off.

One M per process. That is not fussiness: in the fused path's threshold work a
multi-M sweep reported 697us and 1083us for two M values that pad to the same
size and must therefore cost the same, and isolated runs gave 697 for both.

    python bench_mori_mxfp8_gemm.py --shape wq_b -m 1024
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

MXFP8_BK = 32
SHAPES = {"wq_b": (8192, 1280), "wo_b": (5120, 2048)}


class _Layer:
    def __init__(self, weight, weight_scale_mx_e8m0, weight_bf16):
        self.weight = weight
        self.weight_scale_mx_e8m0 = weight_scale_mx_e8m0
        self.weight_bf16 = weight_bf16
        self.mxfp8_native_ready = True


def build(n, k, seed=1234):
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        prepare_mxfp8_native_weight,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    eb = torch.randint(
        120,
        123,
        (n // MXFP8_BK, k // MXFP8_BK),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    shuffled, scale_e8m0, weight_bf16 = prepare_mxfp8_native_weight(
        w, torch.exp2(eb.float() - 127.0), (32, 32)
    )
    return _Layer(shuffled.view(torch.float8_e4m3fn), scale_e8m0, weight_bf16)


def median_us(fn, warmup=10, iters=51):
    """Median over a CUDA-graph replay.

    Graph-captured rather than eager, because an eager loop measures per-launch
    host work the server does not pay -- prefill replays a graph. Measuring it
    eagerly is how an earlier round of this work concluded a path was not worth
    enabling when it was.
    """
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(SHAPES), required=True)
    p.add_argument("-m", type=int, required=True)
    p.add_argument("--floor", type=int, default=64)
    args = p.parse_args()

    n, k = SHAPES[args.shape]
    import sglang.srt.layers.mori_mxfp8_gemm as mori_gemm
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        native_route_plan,
    )
    from sglang.srt.environ import envs

    mori_gemm._MIN_M = args.floor
    layer = build(n, k)
    m = args.m
    x = (torch.randn(m, k, device="cuda") / 8).to(torch.bfloat16)

    def base():
        return mxfp8_native_blockscaled_linear(
            x,
            layer.weight.view(torch.uint8),
            layer.weight_scale_mx_e8m0,
            weight_bf16=layer.weight_bf16,
        )

    with envs.SGLANG_OPT_MORI_MXFP8_GEMM.override(True):

        def mori():
            return mori_gemm.mori_mxfp8_linear(layer, x, None, None, False)

        served = mori() is not None
        t_base = median_us(base)
        t_mori = median_us(mori) if served else None

    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "shape": args.shape,
                "n": n,
                "k": k,
                "m": m,
                "served": served,
                "ref_route": native_route_plan(
                    m, n, k, layer.weight_bf16 is not None, False
                ),
                "base_us": t_base,
                "mori_us": t_mori,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
