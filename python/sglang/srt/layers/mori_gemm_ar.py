# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DSV4 ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``, so today it runs a block-scale fp8 GEMM and
then an NCCL all-reduce of the ``[M, hidden]`` result. The two can overlap:
mori's fused kernel writes each destination's row band straight into a symmetric
window and, as soon as a band's tiles are done, hands it to the SDMA copy
engines; the reduce and all-gather then run as their own kernels.

At ``[16384, 7168]`` K=2048 on 8x MI355X the pair costs 1419.5us and the fused
kernel 1146.2, i.e. **-19.4%**. Essentially all of it is the overlap: the same
pipeline unfused is 1465.9us, and the GEMM alone is 369.4. In-server, over one
20000-token prefill captured with the same warm-up and flush on both sides, GPU
busy time goes 1101.9ms -> 1071.7, i.e. **-30.2ms / -2.7%**.

Prerequisites, all of which this module checks rather than assumes:

* a mori built with ``BUILD_CCO_SDMA=ON``. Setting ``BUILD_CCO_SDMA=1`` in the
  environment only rebuilds the *device* bitcode; if the host library was built
  without it there are no SDMA queues, every put silently does nothing, and the
  all-reduce quietly produces zeros. Point ``PYTHONPATH`` at a mori built with
  the flag on.
* ``MORI_ENABLE_SDMA=1`` at process start.

The kernel itself is ``mori.ops.gemm_ar.GemmAllReduceOp``, which owns the
symmetric window, the per-M compile cache and the row padding. What stays here
is the part that is sglang's: the TP communicator, how the window is sized, and
whether fusing is worth it at this shape.

The fused epilogue pushes a whole ``BLOCK_M`` row band to one destination, so a
destination's row slice has to be a whole number of bands: M must be a multiple
of ``tp_size * 128``. Ragged M is zero-padded up to that, which costs GEMM rows
and buys the overlap -- see ``_MIN_PAD_FILL``. Decode and small batches still
fall back to the ordinary path, and that is expected rather than a failure.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: Smallest padded M worth fusing. Measured at ``[*, 7168] K=2048`` on 8 ranks
#: against the aiter GEMM + NCCL pair the model runs today: 1024 -2.6%, 2048
#: -7.8%, 4096 -12.3%, 8192 -15.7%, 16384 -19.4%. At 1024 a destination gets a
#: single row band, so there is nothing to overlap and the fused kernel is 1.2%
#: *slower* than the same pipeline unfused.
_MIN_FUSED_M = 4096
#: Same threshold with the fp8 gather on, where it has to be higher. The two
#: conversion kernels are a fixed cost against a transfer that shrinks with M,
#: so fp8 only starts paying once the transfer is big enough to dominate them.
#: Measured on the fused path at ``[*, 7168] K=2048``, fp8 against bf16:
#: M=4096 +2.3% (*slower*), M=8192 -5.5%, M=16384 -9.6%.
_MIN_FUSED_M_FP8_GATHER = 8192
#: Padding buys the overlap and costs GEMM rows, so the fill ratio has to clear
#: the gain at the padded size. 0.88 is break-even at ``_MIN_FUSED_M`` (12%
#: more rows against a 12.3% gain) and increasingly safe above it. Measured:
#: M=3616 padded to 4096 is 342.1us against 371.4 for the unfused pair.
_MIN_PAD_FILL = 0.88
_state: Optional[_FusedWoB] = None
_disabled = False


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
            self.world_size, m_max=m_max, n=n, gather_dtype=gather_dtype
        )
        self._comm_ctx = Communicator.init(
            self.world_size,
            self.rank,
            uid,
            per_rank_vmm=4 * window_bytes + (512 << 20),
        )
        self.comm = self._comm_ctx.__enter__()
        try:
            # gather_transport defaults to the LSA pull, which widens the fp8
            # on the way in instead of in a second kernel: 957us against SDMA's
            # 1019 on the fused layer.
            self.op = GemmAllReduceOp(
                self.comm, n=n, k=k, m_max=m_max, gather_dtype=gather_dtype
            )
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

    from mori.ops.gemm_ar import padded_m

    limit = get_global_server_args().chunked_prefill_size
    if limit is not None and limit > 0:
        return max(m_pad, padded_m(limit, world_size))
    return m_pad


def _eligible(m: int, n: int, k: int, world_size: int) -> bool:
    """Whether fusing is both expressible and worth it at this shape.

    mori's ``supports`` answers the first; the thresholds here answer the
    second, and stay on this side because they are measured against what the
    model runs today rather than being a property of the kernel.
    """
    from mori.ops.gemm_ar import padded_m, supports

    if not supports(m, n, k, world_size):
        return False
    floor = (
        _MIN_FUSED_M_FP8_GATHER
        if envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.get()
        else _MIN_FUSED_M
    )
    m_pad = padded_m(m, world_size)
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

    import aiter

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.quantization.fp8_utils import aiter_per1x128_quant

    from mori.ops.gemm_ar import padded_m

    m, k = x.shape
    n = layer.weight.shape[0]
    world_size = get_tp_group().world_size
    if not _eligible(m, n, k, world_size):
        return None

    m_pad = padded_m(m, world_size)
    try:
        if _state is None:
            _state = _FusedWoB(
                m_max=_window_m_max(m_pad, world_size),
                n=n,
                k=k,
            )
        if m_pad > _state.m_max or n != _state.n or k != _state.k:
            return None
        x_in = x if m_pad == m else _state.pad_rows(x, m_pad)
        # Same quantisation the unfused path does. transpose_scale=True makes
        # the quantiser write the group scale in physical [K/128, M] order,
        # which is exactly what the kernel indexes.
        q_input, x_scale = aiter_per1x128_quant(
            x_in, quant_dtype=aiter.dtypes.fp8, transpose_scale=True
        )
        out = _state.run(q_input, x_scale, layer.weight, layer.weight_scale_inv)
        out = out[:m]
    except Exception as err:  # noqa: BLE001 - one failure disables the path
        _disabled = True
        logger.warning(
            "mori fused wo_b failed and is disabled for this process; "
            "falling back to the split path: %s",
            err,
        )
        return None

    if envs.SGLANG_DEBUG_FUSED_WO_B_AR.get():
        out = out.clone()
        ref, _ = layer(x)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        logger.info("mori fused wo_b check: M=%d relL2=%.3e", m, rel)
    return out
