# SPDX-License-Identifier: Apache-2.0
"""mori's mxfp8 GEMM in place of the native mxfp8 linear, on ROCm gfx950.

Separate from `mori_gemm_ar`, which fuses the GEMM with `wo_b`'s all-reduce.
This is the same GEMM with nothing fused onto it, and it exists because most of
the win is the multiply rather than the overlap. bf16 in and bf16 out, the whole
pipeline including quantisation, at `N=5120 K=2048 M=16384` on MI355X:

    bf16 (fake_quant + hipBLASLt)   281.3 us
    sglang mxfp8 (tl.dot_scaled)    274.9
    mori mxfp8                      196.0     -30.3%

**The two are different optimisations with different reach, and their gains do
not add.** At `wo_b` the fused path already runs this GEMM, so this one only
catches the calls fusing declined. At `wq_b` it is the whole story: a
`ColumnParallelLinear` has no all-reduce, so there is nothing to fuse and the
fused op is structurally inapplicable. Per-rank at TP4:

    wq_b   N=8192 K=1280   ColumnParallel -- no collective
    wo_b   N=5120 K=2048   RowParallel

The ordering falls out for free and needs no coordination: the model calls
`fused_wo_b` first and only reaches the linear when it declines, so this hook
sees exactly the remainder.

Hooked at `Fp8LinearMethod._apply_gfx95_native`, which is the one place every
`mxfp8_native_ready` layer passes through, and where the input has already been
normalised to `(x, input_scale, input_on_fp8_grid)`.

Both operand forms the hook can hand over are served: a bf16 activation, and
the fp8-plus-row-major-scale one a fused producer emits. The second needs the
scale converting, which is not free, and is still a win -- see `_a_operands`.

Returning None means "use the normal path" and is not a failure: an unsupported
shape or an M below the floor both land there.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.mori_mxfp8_common import (
    mori_weight,
    mxfp8_ready,
    mxfp8_shape,
    quantize_packed,
)

logger = logging.getLogger(__name__)

#: Smallest M worth handing to mori rather than the native route.
#:
#: Against `mxfp8_native_blockscaled_linear` at V4.1-Flash's two attention
#: shapes, TP4, graph-replayed, one M per process, two runs:
#:
#:     M       wq_b (8192x1280)   wo_b (5120x2048)
#:       64         +45.8%             +66.5%
#:      256         +36.6%             +36.5%
#:      512          +2.0%             +22.1%
#:     1024          -4.9%              -1.2%
#:     2048         -24.0%             -20.4%
#:    16384         -34.1%             -28.9%
#:
#: mori's time is almost flat from M=64 to 512 (29-39us on both shapes) because
#: the 256x256 tile is mostly idle there, while the native route drops to a
#: GEMV-shaped kernel that suits it. 2048 rather than 1024: at 1024 both shapes
#: are inside run-to-run noise -- 1.6us and 0.5us of absolute difference -- so
#: there is nothing there to collect, and from 2048 both win by at least 20%.
_MIN_M = 2048

_ops: dict[tuple[int, int], object] = {}
_disabled = False
_warned_reject = False

#: Two of these run per layer (wq_b and wo_b), so 40 layers is 80 calls a
#: forward; this logs roughly every ten forwards.
_SHAPE_EVERY = 800
_SHAPE_LOG = os.environ.get("SGLANG_OPT_MORI_MXFP8_GEMM_SHAPE_LOG") == "1"
_shape_hist: dict = {}
_shape_calls = 0


#: The env read, done once. Not `.get()` per call: this runs on every linear of
#: every layer, including decode's, where it does nothing but decline. Reading
#: and parsing os.environ each time measured 1.4us against 0.29 -- 0.39ms per
#: decode step across 7 shapes and 40 layers, which is 8-11% of ITL spent by a
#: path that is not doing anything.
_enabled: bool | None = None


def mori_mxfp8_available() -> bool:
    """Static gate, cheap enough to call per layer -- see `_enabled`."""
    global _enabled
    if _enabled is None:
        _enabled = envs.SGLANG_OPT_MORI_MXFP8_GEMM.get()
    return _enabled and not _disabled


def _op_for(n: int, k: int):
    """The op for this shape, or False if mori cannot express it.

    One per (N, K) and no per-M cache behind it: `c_m` is a runtime argument, so
    one compile serves every token count. That matters here more than anywhere
    -- a server's prefill batch size changes constantly, and a per-M cache would
    pay a multi-second compile on nearly every batch.
    """
    hit = _ops.get((n, k))
    if hit is not None:
        return hit
    from mori.ops.gemm_ar import Mxfp8GemmOp, supports_gemm

    if not supports_gemm(n, k):
        _ops[(n, k)] = False
        return False
    _ops[(n, k)] = Mxfp8GemmOp(n=n, k=k)
    return _ops[(n, k)]


def _a_operands(op, x, input_scale, m):
    """A's fp8 values and mori's packed A scale, plus the padded M.

    Two input forms reach here and both are served.

    A **bf16** activation is quantised straight into mori's layout, which costs
    nothing extra: the packed store stays inside the 64 bytes one quantiser
    program already owns.

    An activation a fused producer already quantised -- what `wq_b` gets from
    `rmsnorm_fake_quant_fp8` at large M -- arrives as fp8 plus a *row-major*
    ``[M, K/32]`` scale, which mori cannot index, so it needs
    `preshuffle_a_scale`. That is a real extra pass, 22-24us at wq_b's shape,
    and it is still worth paying: against the native route, which takes this
    form with no conversion at all, mori is -38.1% at M=2048 and -22.2% at
    M=16384 *including* the conversion.

    Declining it was tempting and would have been wrong -- `wq_b` is the layer
    this path exists for, being column-parallel with nothing to fuse, and at the
    M where it matters its operand is always this form.
    """
    from mori.ops.gemm_ar import preshuffle_a_scale

    m_pad = op.padded_m(m)
    if input_scale is None:
        x_in = x if m_pad == m else op.pad_rows(x, m_pad)
        return quantize_packed(x_in), m_pad

    # Already fp8. Pad the values with zeros -- a padded row contributes zero
    # whatever its scale says -- and the scale with anything the layout accepts.
    if m_pad != m:
        x = op.pad_rows(x, m_pad)
        input_scale = torch.nn.functional.pad(
            input_scale, (0, 0, 0, m_pad - m)
        )
    return (x, preshuffle_a_scale(input_scale.to(torch.int32))), m_pad


def mori_mxfp8_linear(
    layer,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor | None,
    input_on_fp8_grid: bool,
) -> torch.Tensor | None:
    """The native mxfp8 linear, through mori. ``None`` means "use the normal path".

    ``input_on_fp8_grid`` needs no special handling: it says the bf16 input is
    already rounded onto the fp8 grid, which makes the quantisation below
    lossless rather than a second rounding.
    """
    global _disabled, _warned_reject

    if not mori_mxfp8_available():
        return None

    # The M test first and on x's own shape: it rejects every decode call, and
    # anything above it here runs on all of them.
    m = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]
    if m < _MIN_M:
        if _SHAPE_LOG:
            _record_shape(*mxfp8_shape(layer), m, False)
        return None

    x_2d = x.view(-1, x.shape[-1])
    n, k = mxfp8_shape(layer)
    if k != x_2d.shape[1]:
        if _SHAPE_LOG:
            _record_shape(n, k, m, False)
        return None

    try:
        if not mxfp8_ready(layer):
            return None
        op = _op_for(n, k)
        if op is False:
            if _SHAPE_LOG:
                _record_shape(n, k, m, False)
            return None
        operands = _a_operands(op, x_2d, input_scale, m)
        if operands is None:
            if _SHAPE_LOG:
                _record_shape(n, k, m, False)
            return None
        (q_input, a_scale), m_pad = operands
        weight, b_scale = mori_weight(layer)
        out = op(q_input, weight, a_scale, b_scale)[:m]
        if _SHAPE_LOG:
            _record_shape(n, k, m, True)
    except ValueError as err:
        # A shape or contract rejection is about *this call*. Disabling the
        # process on one would be a standing hazard: a server's M changes with
        # every batch, so one unlucky shape would switch the path off for good.
        if not _warned_reject:
            _warned_reject = True
            logger.warning(
                "mori mxfp8 GEMM declined a call and fell back for it; further "
                "declines are silent: %s",
                err,
            )
        return None
    except Exception as err:  # noqa: BLE001 - anything else is not per-call
        _disabled = True
        logger.warning(
            "mori mxfp8 GEMM failed and is disabled for this process; falling "
            "back to the native linear: %s",
            err,
        )
        return None

    if bias is not None:
        out = out + bias
    if envs.SGLANG_DEBUG_MORI_MXFP8_GEMM.get():
        _cross_check(layer, x_2d, out, m, n, m_pad)
    return out.view(*x.shape[:-1], n)


def _cross_check(layer, x_2d, out, m, n, m_pad):
    """Log relL2 against the route this replaced, on the same inputs."""
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
    )

    ref = mxfp8_native_blockscaled_linear(
        x_2d,
        layer.weight.view(torch.uint8),
        layer.weight_scale_mx_e8m0,
        weight_bf16=layer.weight_bf16,
    )
    rel = (
        torch.linalg.vector_norm(out.float() - ref.float())
        / torch.linalg.vector_norm(ref.float())
    ).item()
    logger.info(
        "mori mxfp8 GEMM check: N=%d K=%d M=%d(pad %d) relL2=%.3e",
        n,
        layer.weight_scale_mx_e8m0.shape[1] * 32,
        m,
        m_pad,
        rel,
    )


def _record_shape(n: int, k: int, m: int, served: bool) -> None:
    """Log what M each shape is handed, as a periodic histogram.

    A path that declined every call looks exactly like one that ran and was not
    worth it: the model stays correct, the profile stays plausible, and a
    benchmark reports the baseline under the optimisation's name. This is what
    tells the two apart. Set SGLANG_OPT_MORI_MXFP8_GEMM_SHAPE_LOG=1.
    """
    global _shape_calls
    key = (n, k, m, bool(served))
    _shape_hist[key] = _shape_hist.get(key, 0) + 1
    _shape_calls += 1
    if _shape_calls >= _SHAPE_EVERY:
        _shape_calls = 0
        items = sorted(_shape_hist.items(), key=lambda kv: -kv[1])
        logger.info(
            "mori mxfp8 shapes: %s",
            " ".join(
                f"{n}x{k}/M={m}{'+' if ok else '-'}x{c}"
                for (n, k, m, ok), c in items
            ),
        )
        _shape_hist.clear()
