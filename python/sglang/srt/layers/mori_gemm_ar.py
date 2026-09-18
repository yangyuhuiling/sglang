# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DeepSeek-V4.1-Flash ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``, so today it runs a GEMM and then an NCCL
all-reduce of the ``[M, hidden]`` result. The two can overlap: mori's fused
kernel writes each destination's row band straight into a symmetric window and,
as soon as a band's tiles are done, hands it to the SDMA copy engines; the
reduce and all-gather then run as their own kernels.

**Most of the win here is not the overlap**, and reading it as such would set
the thresholds wrong. At V4.1-Flash's TP4 shape (N=5120 K=2048) three separate
things are on offer, measured on an idle MI355X:

* **The GEMM.** wo_b's tuning table picks ``hipblaslt_bf16`` at this shape.
  mori's mxfp8 GEMM does the whole bf16-in/bf16-out pipeline at M=16384 in
  196.0us against 281.3, **-30.3%** -- before any fusion.
* **The fp8 all-gather wire.** Worth -15.6% on its own, i.e. without fusing at
  all, at a cost in accuracy: relL2 2.37e-03 -> 2.31e-02.
* **The fusion.** +1.7% at M=4096 to +5.9% at M=16384 on a bf16 wire. Not the
  -19.4% V4-Pro sees, and the reason is arithmetic rather than a regression:
  the GEMM is 178us against 1414us of communication, so the ceiling (hiding the
  GEMM entirely) is 11-15% and the fusion collects about half of it. Making the
  GEMM faster lowered this number.

Composed, the best configuration is 1192.1us against 1592.5 for split/bf16,
**-25.1%**: fused-sdma with ``gather_dtype=fp8`` over the LSA pull.

**Check that mori was built with `BUILD_CCO_SDMA=ON` before believing any
measurement of this path.** With it off every put silently does nothing: the
all-reduce returns mostly the local slice, the model still answers, and the
fused path looks *faster* than it is because it is not moving any data. Measured
that way it reads -6.8% instead of -2.3%, and the fp8 pull -- the only leg that
does not go through SDMA -- looks like the slowest of the three instead of the
fastest. Perplexity is what catches it: 862511 against 3.26 on the same text.

Prerequisites, all of which this module checks rather than assumes:

* a mori built with ``BUILD_CCO_SDMA=ON``. Setting ``BUILD_CCO_SDMA=1`` in the
  environment only rebuilds the *device* bitcode; if the host library was built
  without it there are no SDMA queues, every put silently does nothing, and the
  all-reduce quietly produces zeros. Point ``PYTHONPATH`` at a mori built with
  the flag on.
* ``MORI_ENABLE_SDMA=1`` at process start.
* a layer sglang's native mxfp8 route already prepared
  (``layer.mxfp8_native_ready``) -- see ``_fused_weight``.

The kernel itself is ``mori.ops.gemm_ar.GemmAllReduceOp``, which owns the
symmetric window, the per-M compile cache and the row padding. What stays here
is the part that is sglang's: the TP communicator, how the window is sized, the
operand conversions, and whether fusing is worth it at this shape.

The fused epilogue pushes a whole ``BLOCK_M`` row band to one destination, so a
destination's row slice has to be a whole number of bands: M must be a multiple
of ``tp_size * 256``. That granule is mxfp8's, not a choice -- see ``_BLOCK_M``.
Ragged M is zero-padded up to it, which costs GEMM rows and buys the overlap --
see ``_MIN_PAD_FILL``. Decode and small batches still fall back to the ordinary
path, and that is expected rather than a failure.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: Smallest padded M worth fusing on a **bf16** wire.
#:
#: 16384 rather than V4-Pro's 4096, and the honest reading of the measurements
#: is that this wire is not worth enabling here at all -- 16384 is where it
#: first clears the noise, not where it starts paying. Against what runs today
#: (``mxfp8_native_blockscaled_linear`` + NCCL) at TP4 N=5120 K=2048, by m_pad:
#:
#:     m_pad  5120   +11.0%      m_pad  9216   +10.9% .. -8.5%
#:     m_pad  8192   -0.4% .. -9.5%    m_pad 13312   -2.2% .. -2.4%
#:     m_pad 16384  -12.2%
#:
#: The spread within one m_pad is the *baseline* moving, not the fused path:
#: today's route retunes per M bucket and swings ~20% between them (M=8200
#: costs 943us, M=8800 costs 1133). The fusion's own contribution does not
#: clear that, which is the arithmetic in the module docstring -- the GEMM is
#: 178us against 1414us of communication, so the ceiling is 11-15%.
_MIN_FUSED_M = 16384
#: The same for the **fp8** wire, which is a different question and gets a
#: different answer: it is consistently 15-25% below the bf16 wire, so it does
#: clear the baseline's bucket noise. Every m_pad >= 8192 measured wins, worst
#: case -7.4%:
#:
#:     m_pad  8192  -16.7% .. -24.4%   m_pad 13312  -18.6% .. -18.7%
#:     m_pad  9216   -7.4% .. -23.6%   m_pad 16384  -28.8%
#:
#: Below that it is not monotonic. m_pad 3072 and 4096 win (-9.8%, -13.0%) but
#: **m_pad 5120 loses** (+18.7%, +8.8%, -0.5%): the fused cost jumps 510 ->
#: 600us across that step while the baseline only goes 587 -> 603, because both
#: are in one dot_scaled bucket and it is not compute-bound there. A single
#: floor cannot keep 4096 and drop 5120, so this gives up the small win below
#: rather than take an 18% regression on a band of M a server will actually hit.
_MIN_FUSED_M_FP8_GATHER = 8192
#: Padding buys the overlap and costs GEMM rows, so a ragged M has to be full
#: enough that the gain at the padded size still clears it.
#:
#: Note this does not bind at the floors above: the granule is
#: ``tp_size * 256``, so at m_pad >= 8192 the fill cannot fall below
#: (8192-1023)/8192 = 0.875 in the first place. It is a guard for a lowered
#: floor rather than an active rule. The lowest fill measured above the floor,
#: 0.879 at M=7200, wins -16.7% on the fp8 wire.
_MIN_PAD_FILL = 0.85
_state: Optional[_FusedWoB] = None
_disabled = False
_warned_layout = False
_warned_reject = False

#: Log what M this layer is actually handed, as a periodic histogram.
#:
#: M is the token count of one forward, not of one request, so whether the fused
#: path engages depends on how the scheduler batches -- which is not something to
#: infer from the client's concurrency. Set
#: SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG=1 to find out.
_SHAPE_LOG = os.environ.get("SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG") == "1"
_shape_hist: dict = {}
_shape_calls = 0
#: 61 wo_b calls make one forward, so this logs roughly every ten of them.
_SHAPE_EVERY = 610


def _record_shape(m, eligible, world_size):
    global _shape_calls
    key = (m, _padded_m(m, world_size), bool(eligible))
    _shape_hist[key] = _shape_hist.get(key, 0) + 1
    _shape_calls += 1
    if _shape_calls >= _SHAPE_EVERY:
        _shape_calls = 0
        items = sorted(_shape_hist.items(), key=lambda kv: -kv[1])
        logger.info(
            "wo_b shapes: %s",
            " ".join(f"M={m}/pad{p}{'+' if e else '-'}x{c}" for (m, p, e), c in items),
        )
        _shape_hist.clear()


#: mori's mxfp8 kernel packs a lane's four M tiles into one scale dword and
#: picks the byte with the MFMA's opsel, which is four tiles only at BLOCK_M 256
#: -- so a destination's row band is 256 rows here, not blockscale's 128, and M
#: pads to a multiple of ``tp_size * 256``.
_BLOCK_M = 256
#: The ue8m0 group along K. Also the rows one quantiser program owns, which is
#: what lets it write mori's packed scale layout without extra traffic.
_QUANT_BLOCK_M = 64


def _padded_m(m: int, world_size: int) -> int:
    from mori.ops.gemm_ar import padded_m

    return padded_m(m, world_size, _BLOCK_M)


@triton.jit
def _mxfp8_quant_packed_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    BLOCK_M: tl.constexpr,
):
    """sglang's ``_mxfp8_quant_kernel`` writing mori's scale layout directly.

    A variant rather than a stride argument on the original, because the layout
    is not expressible as strides: mori wants element ``(m, kb)`` at
    ``kb*M + (m//64)*64 + (m%16)*4 + (m%64)//16``, which permutes *within* each
    64-row group so a lane's four M tiles land in one dword.

    That permutation is free here. The destination stays inside the 64 bytes
    this program already owns, so the store is the same cache line and only its
    order changes -- measured bit-identical to quantising and converting
    afterwards, and slightly faster than the stock kernel, since transposing
    also turns a strided store into a coalesced one.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    sb = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    descale = tl.exp2(sb - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)
    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    dst = pid_b * M + (offs_m // 64) * 64 + (offs_m % 16) * 4 + (offs_m % 64) // 16
    tl.store(s_ptr + dst, sb.to(tl.uint8), mask=m_mask)


def _quantize_packed(x: torch.Tensor):
    """bf16 ``[M, K]`` -> (fp8 e4m3 values, mori's packed ue8m0 A scale)."""
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import MXFP8_VALUE_DTYPE

    m, k = x.shape
    xq = torch.empty((m, k), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scale = torch.empty((k // 32) * m, dtype=torch.uint8, device=x.device)
    _mxfp8_quant_packed_kernel[(triton.cdiv(m, _QUANT_BLOCK_M), k // 32)](
        x,
        xq,
        scale,
        m,
        k,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        BLOCK_M=_QUANT_BLOCK_M,
    )
    return xq, scale.view(torch.int32)


_logged_config = False


def _shuffled_once():
    global _logged_config
    if _logged_config:
        return True
    _logged_config = True
    return False


def _fused_shape(layer):
    """``wo_b``'s logical (N, K).

    Not ``layer.weight.shape``: ``prepare_mxfp8_native_weight`` rebinds the
    weight to its shuffled form, ``[N/16, K/128, 2048]``, so that first axis is
    N/16 rather than N. Reading it as N silently disqualifies every layer --
    5120 becomes 320, ``supports`` says no because 320 is not a multiple of
    BLOCK_N, and the path falls back without a word. The ue8m0 scale keeps the
    logical shape, ``[N/32, K/32]``, so take it from there.
    """
    sn, sk = layer.weight_scale_mx_e8m0.shape
    return sn * 32, sk * 32


def _fused_weight(layer):
    """The op's B operand and B scale, or None if this layer cannot be fused.

    V4.1-Flash quantises 32-wide ue8m0, and sglang's own native mxfp8 route
    already prepares almost exactly what mori wants -- ``mxfp8_native_ready``
    means ``prepare_mxfp8_native_weight`` ran and left the shuffled fp8 bytes on
    ``layer.weight`` and the ``[N/32, K/32]`` exponent bytes on
    ``layer.weight_scale_mx_e8m0``. Handing over the unprepared weight does not
    fail, it returns an uncorrelated result, so this checks rather than assumes.

    Two conversions, both once per layer and cached on it:

    * **The weight.** ``shuffle_mxfp8_weight`` and mori's ``preshuffle_b`` give
      each lane the same K range and differ only in how the two 64-wide K
      sub-blocks sit: sglang interleaves them inside a lane's 32 bytes, mori
      keeps them as two 16-byte blocks. So the permutation below is exact --
      verified byte-for-byte at [5120, 2048].
    * **The B scale.** mori indexes it K-block major as int32, and the exponent
      bytes are already at the 32x32 granularity it wants (not the per-row
      ``[N, K/32]`` that ``tl.dot_scaled`` takes), so it is a transpose and a
      widen.

    Cached on the layer rather than recomputed: the permutation is a full copy
    of the weight, which at 61 layers would be pointless per-call work.
    """
    global _warned_layout
    cached = getattr(layer, "_mori_b", None)
    if cached is not None:
        return cached

    if not getattr(layer, "mxfp8_native_ready", False) or not hasattr(
        layer, "weight_scale_mx_e8m0"
    ):
        if not _warned_layout:
            _warned_layout = True
            logger.warning(
                "mori fused wo_b: this layer is not mxfp8_native_ready (weight "
                "shape %s), which the fused kernel requires; not fusing.",
                tuple(layer.weight.shape),
            )
        return None

    n, k = _fused_shape(layer)
    w = layer.weight.data.contiguous().view(torch.uint8)
    # [N/16, K/128, 64 lanes, 2 sub-blocks, 16B] -> the two sub-blocks split out
    b = (
        w.reshape(n // 16, k // 128, 64, 2, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(n, k)
        .view(layer.weight.dtype)
    )
    b_scale = (
        layer.weight_scale_mx_e8m0.data.contiguous()
        .view(torch.uint8)
        .t()
        .contiguous()
        .to(torch.int32)
        .reshape(-1)
    )
    layer._mori_b = (b, b_scale)
    return layer._mori_b


class _FusedWoB:
    """The TP communicator and mori's op, one per rank.

    The window, the per-M compile cache and the row padding all live in
    ``GemmAllReduceOp``; this holds the cco ``Communicator`` built from sglang's
    TP group, which is the part mori cannot construct for us.
    """

    def __init__(self, m_max: int, n: int, k: int):
        import torch.distributed as dist
        from mori.cco import Communicator, UniqueId
        from mori.ops.gemm_ar import GemmAllReduceOp
        from sglang.srt.distributed import get_tp_group

        tp = get_tp_group()
        self.rank = tp.rank_in_group
        self.world_size = tp.world_size
        self.n, self.k, self.m_max = n, k, m_max

        payload = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=tp.ranks[0], group=tp.cpu_group)
        uid = UniqueId.from_bytes(payload[0])

        # The window is VMM memory outside torch's allocator, so the run has to
        # leave room for it (lower --mem-fraction-static). Sized from m_max, and
        # it cannot grow afterwards.
        gather_dtype = (
            "fp8" if envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.get() else "bf16"
        )
        window_bytes = GemmAllReduceOp.window_bytes_for(
            self.world_size,
            m_max=m_max,
            n=n,
            block_m=_BLOCK_M,
            gather_dtype=gather_dtype,
        )
        self._comm_ctx = Communicator.init(
            self.world_size,
            self.rank,
            uid,
            per_rank_vmm=4 * window_bytes + (512 << 20),
        )
        self.comm = self._comm_ctx.__enter__()
        try:
            # Which way the fp8 gather moves. The LSA pull widens on the way in
            # instead of in a second kernel, and it wins in both places: 951us
            # against SDMA's 1011 on the layer, 1041.2ms of GPU busy against
            # 1052.1 in the server.
            self.op = GemmAllReduceOp(
                self.comm,
                n=n,
                k=k,
                m_max=m_max,
                quant="mxfp8",
                gather_dtype=gather_dtype,
                gather_transport=os.environ.get(
                    "SGLANG_OPT_FUSED_WO_B_AR_GATHER_TRANSPORT", "lsa"
                ),
            )
            # Prove the collective moves bytes before serving a single token.
            # A mori built without BUILD_CCO_SDMA=ON -- the default, and what the
            # CI image ships -- compiles the puts out: every kernel still
            # launches, nothing moves, the all-reduce returns mostly the local
            # slice, and the model still answers fluently while being wrong.
            # It also measures *faster* that way, which is how a whole
            # end-to-end campaign came out at -6.8% instead of -2.3%. This costs
            # one collective at m_max, on kernels the first real call compiles
            # anyway.
            self.op.self_test()
        except Exception:
            # The communicator holds a VMM reservation the whole process pays
            # for. Half-constructing this object and leaving it open makes the
            # *fallback* path fail too, which is how a simple attribute error
            # here once took the server down.
            self._comm_ctx.__exit__(None, None, None)
            raise
        logger.info(
            "mori fused wo_b: window %.0f MiB, tp=%d, M<=%d, N=%d, K=%d, gather=%s",
            self.op.window_bytes / 2**20,
            self.world_size,
            m_max,
            n,
            k,
            gather_dtype,
        )

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        return self.op.pad_rows(x, m_pad)

    def run(self, q_input, x_scale_raw, weight, weight_scale) -> torch.Tensor:
        return self.op(q_input, weight, x_scale_raw, weight_scale)


def _window_m_max(m_pad: int, world_size: int) -> int:
    """Rows the symmetric window is sized for.

    Taken from the chunked-prefill limit rather than the first request seen, so
    a short prompt arriving first cannot fix a window too small for a full chunk
    later -- the window cannot grow once allocated.
    """
    from sglang.srt.server_args import get_global_server_args

    limit = get_global_server_args().chunked_prefill_size
    if limit is not None and limit > 0:
        return max(m_pad, _padded_m(limit, world_size))
    return m_pad


def _eligible(m: int, n: int, k: int, world_size: int) -> bool:
    """Whether fusing is both expressible and worth it at this shape.

    mori's ``supports`` answers the first; the thresholds here answer the
    second, and stay on this side because they are measured against what the
    model runs today rather than being a property of the kernel.
    """
    from mori.ops.gemm_ar import supports

    if not supports(m, n, k, world_size, quant="mxfp8"):
        return False
    floor = (
        _MIN_FUSED_M_FP8_GATHER
        if envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.get()
        else _MIN_FUSED_M
    )
    m_pad = _padded_m(m, world_size)
    return m_pad >= floor and m >= _MIN_PAD_FILL * m_pad


def fused_wo_b_available() -> bool:
    """Static gate, cheap enough to call per layer."""
    return not _disabled and envs.SGLANG_OPT_FUSED_WO_B_AR.get()


def fused_wo_b(layer, x: torch.Tensor) -> Optional[torch.Tensor]:
    """wo_b's `GEMM + all-reduce`, fused. ``None`` means "use the normal path".

    ``x`` is the bf16 activation this rank holds, ``[M, K]``. The result is the
    all-reduced ``[M, N]``, as a **view of the symmetric window** -- the next
    layer's wo_b overwrites it, which is safe because the caller consumes it in
    the residual add of the same layer. ``SGLANG_DEBUG_FUSED_WO_B_AR`` forces a copy
    and cross-checks against the unfused path.
    """
    global _state, _disabled

    if not fused_wo_b_available():
        return None

    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Fp8GridActivation
    from sglang.srt.distributed import get_tp_group

    # wo_a hands wo_b either a plain bf16 activation or an Fp8GridActivation --
    # a bf16 tensor already rounded onto wo_b's fp8 grid, so quantising it below
    # is lossless rather than a second rounding. Either way what the GEMM needs
    # is the bf16 tensor.
    if isinstance(x, Fp8GridActivation):
        x = x.x

    m, k = x.shape
    if not hasattr(layer, "weight_scale_mx_e8m0"):
        return None
    n, w_k = _fused_shape(layer)
    if w_k != k:
        return None
    world_size = get_tp_group().world_size
    eligible = _eligible(m, n, k, world_size)
    if _SHAPE_LOG:
        _record_shape(m, eligible, world_size)
    if not eligible:
        return None

    m_pad = _padded_m(m, world_size)
    try:
        if _state is None:
            _state = _FusedWoB(
                m_max=_window_m_max(m_pad, world_size),
                n=n,
                k=k,
            )
        if m_pad > _state.m_max or n != _state.n or k != _state.k:
            return None
        prepared = _fused_weight(layer)
        if prepared is None:
            return None
        weight, b_scale = prepared
        x_in = x if m_pad == m else _state.pad_rows(x, m_pad)
        # The quantiser writes mori's packed scale layout itself, so there is no
        # conversion pass after it. Zero-padded rows quantise to zero values
        # (their scale is tiny but finite), and rows are independent in a GEMM,
        # so the padding contributes nothing to any real row.
        q_input, x_scale = _quantize_packed(x_in)
        out = _state.run(q_input, x_scale, weight, b_scale)
        out = out[:m]
    except ValueError as err:
        # A shape or contract rejection is about *this call*, not about the
        # path. Disabling the process on one would be a standing hazard: the op
        # raises ValueError for an M it cannot serve, and a server's M changes
        # with every batch, so one unlucky shape used to switch the whole
        # optimisation off for good.
        global _warned_reject
        if not _warned_reject:
            _warned_reject = True
            logger.warning(
                "mori fused wo_b declined a call and fell back for it; further "
                "declines are silent: %s",
                err,
            )
        return None
    except Exception as err:  # noqa: BLE001 - anything else is not per-call
        _disabled = True
        logger.warning(
            "mori fused wo_b failed and is disabled for this process; "
            "falling back to the split path: %s",
            err,
        )
        return None

    if envs.SGLANG_DEBUG_FUSED_WO_B_AR.get():
        if not _shuffled_once():
            o = _state.op
            logger.info(
                "mori fused wo_b config: rank=%d/%d m_max=%d n=%d k=%d queues=%d "
                "gather=%s/%s m=%d m_pad=%d",
                o.rank,
                o.world_size,
                o.m_max,
                o.n,
                o.k,
                o.sdma_queues,
                o.gather_dtype,
                o.gather_transport,
                m,
                m_pad,
            )
        out = out.clone()
        # Same inputs, immediately again: if the two disagree the server context
        # is racing the collective; if they agree the op is deterministic here
        # and merely disagrees with the reference.
        again = _state.run(q_input, x_scale, weight, b_scale)[:m].clone()
        self_rel = ((out.float() - again.float()).norm() / out.float().norm()).item()
        ref, _ = layer(x)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        logger.info(
            "mori fused wo_b check: M=%d relL2=%.3e self_rel=%.3e", m, rel, self_rel
        )
    return out
