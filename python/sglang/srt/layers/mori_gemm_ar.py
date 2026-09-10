# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DSV4 ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``, so today it runs a block-scale fp8 GEMM and
then an NCCL all-reduce of the ``[M, hidden]`` result. The two can overlap:
mori's fused kernel writes each destination's row band straight into a symmetric
window and, as soon as a band's tiles are done, hands it to the SDMA copy
engines; the reduce and all-gather then run as their own kernels.

At ``[16384, 7168]`` K=2048 on 8x MI355X the pair costs 1419.5us and the fused
kernel 1144.3, i.e. **-19.4%**. Essentially all of it is the overlap: the same
pipeline unfused is 1462.8us, and our GEMM alone (372.7) matches aiter's
blockscale kernel (374.4). In-server the pair goes 1617us -> 1297 per layer,
which is -19.0ms over a 20000-token prefill.

Prerequisites, all of which this module checks rather than assumes:

* a mori built with ``BUILD_CCO_SDMA=ON``. Setting ``BUILD_CCO_SDMA=1`` in the
  environment only rebuilds the *device* bitcode; if the host library was built
  without it there are no SDMA queues, every put silently does nothing, and the
  all-reduce quietly produces zeros. Point ``PYTHONPATH`` at a mori built with
  the flag on.
* ``MORI_ENABLE_SDMA=1`` at process start.
* ``SGLANG_OPT_FUSED_WO_B_AR_DIR`` pointing at mori's ``benchmark/cco/flydsl``. The
  kernels are not part of the installed package, so the path is explicit.

The fused epilogue pushes a whole ``BLOCK_M`` row band to one destination, so a
destination's row slice has to be a whole number of bands: M must be a multiple
of ``tp_size * 128``. Ragged M is zero-padded up to that, which costs GEMM rows
and buys the overlap -- see ``_MIN_PAD_FILL``. Decode and small batches still
fall back to the ordinary path, and that is expected rather than a failure.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: Row-band granularity of the fused kernel. A destination's slice must be a
#: whole number of these, which is the M constraint below.
_BLOCK_M = 128
_BLOCK_N = 256
#: fp8 block-scale group along K, fixed by the model's quantiser and the kernel.
_SCALE_BK = 128

#: Smallest padded M worth fusing. Measured at ``[*, 7168] K=2048`` on 8 ranks
#: against the aiter GEMM + NCCL pair the model runs today: 1024 -2.6%, 2048
#: -7.8%, 4096 -12.3%, 8192 -15.7%, 16384 -19.4%. At 1024 a destination gets a
#: single row band, so there is nothing to overlap and the fused kernel is 1.2%
#: *slower* than the same pipeline unfused.
_MIN_FUSED_M = 4096
#: Padding buys the overlap and costs GEMM rows, so the fill ratio has to clear
#: the gain at the padded size. 0.88 is break-even at ``_MIN_FUSED_M`` (12%
#: more rows against a 12.3% gain) and increasingly safe above it. Measured:
#: M=3616 padded to 4096 is 342.1us against 371.4 for the unfused pair.
_MIN_PAD_FILL = 0.88
#: Pushes per destination, the overlap mechanism itself: with one chunk a
#: destination's tile counter only fires when its whole slice is done, which
#: under the rotated tile order is the end of the GEMM, so nothing overlaps.
#: 8 is the measured plateau (1291.7us at 1, 1146.2 at 8).
_MAX_CHUNKS = 8

_state: Optional[_FusedWoB] = None
_disabled = False


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FusedWoB:
    """Communicator, symmetric window and compiled-kernel cache, one per rank."""

    def __init__(self, kernel_dir: str, m_max: int, n: int, k: int):
        import torch.distributed as dist
        from mori.cco import (
            GDA_CONNECTION_NONE,
            CCODevCommRequirements,
            Communicator,
            UniqueId,
        )
        from mori.tensor_utils import from_gpu_ptr

        from sglang.srt.distributed import get_tp_group

        ar_dir = os.path.join(kernel_dir, "ar")
        gemm_dir = os.path.join(kernel_dir, "gemm_ar")
        # The kernel modules import each other by bare name, the way mori's own
        # tests load them.
        for d in (ar_dir, gemm_dir):
            if d not in sys.path:
                sys.path.insert(0, d)
        self._layout = _load_module(
            "_mori_ar_layout", os.path.join(ar_dir, "layout.py")
        )
        self._sdma = _load_module(
            "_mori_ar_kernels_sdma", os.path.join(ar_dir, "kernels_sdma.py")
        )
        self._fused = _load_module(
            "_mori_gemm_ar_kernels_fused", os.path.join(gemm_dir, "kernels_fused.py")
        )
        import flydsl.expr as fx

        self._fx = fx
        self._from_gpu_ptr = from_gpu_ptr

        tp = get_tp_group()
        self.rank = tp.rank_in_group
        self.world_size = tp.world_size
        self.n, self.k, self.m_max = n, k, m_max

        payload = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=tp.ranks[0], group=tp.cpu_group)
        uid = UniqueId.from_bytes(payload[0])

        cfg_max = self._make_cfg(m_max)
        window_bytes = cfg_max.window_bytes
        # The window is VMM memory outside torch's allocator; the run has to
        # leave room for it (lower --mem-fraction-static).
        self._comm_ctx = Communicator.init(
            self.world_size,
            self.rank,
            uid,
            per_rank_vmm=4 * window_bytes + (512 << 20),
        )
        self.comm = self._comm_ctx.__enter__()
        self.mem = self.comm.alloc_mem(window_bytes)
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
        self._from_gpu_ptr(self.mem.ptr, (window_bytes,), torch.uint8).zero_()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        # One queue per (source, destination) pair is all the pipeline uses --
        # a destination is one xGMI link. Asking for world_size queues would
        # create world_size *per peer* and touch one of them; inside a server
        # that already holds SDMA engines for its own copies that overruns the
        # per-engine queue slots and hsaKmtCreateQueueExt fails (anvil.cpp:237).
        reqs.sdma_queue_count = 1
        self.dev_comm = self.comm.create_dev_comm(reqs)
        self._cache: dict[int, tuple] = {}
        self._pad_in: Optional[torch.Tensor] = None
        logger.info(
            "mori fused wo_b: window %.0f MiB, tp=%d, M<=%d, N=%d, K=%d",
            window_bytes / 2**20,
            self.world_size,
            m_max,
            n,
            k,
        )

    def _make_cfg(self, m: int):
        cfg = self._layout.ArConfig(
            world_size=self.world_size,
            m=m,
            n=self.n,
            recv_slots=self.world_size,
            counter_chunks=_counter_chunks(m, self.world_size),
        )
        cfg.validate()
        return cfg

    def _compiled(self, m: int):
        """Kernel + SDMA phases for this M, compiled on first sight."""
        hit = self._cache.get(m)
        if hit is not None:
            return hit
        cfg = self._make_cfg(m)
        gemm = self._fused.compile_fused_gemm_scatter(
            cfg,
            self.rank,
            K=self.k,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            b_preshuffled=True,
            fuse=True,
            transport="sdma",
            quant="blockscale",
            sdma_queues=1,
            # The three C-store stages are `compile_fused_gemm_scatter`
            # defaults-off but bench_gemm_ar defaults-on, and blockscale
            # requires swap_ab.
            swap_ab=True,
            permlane=True,
            lane_transpose=True,
        )
        parts = self._sdma.build_sdma_phases(cfg, self.rank, queues=1)
        c = self._from_gpu_ptr(
            self.mem.ptr + cfg.input_off, (m, self.n), torch.bfloat16
        )
        out = self._from_gpu_ptr(
            self.mem.ptr + cfg.output_off, (m, self.n), torch.bfloat16
        )
        hit = (gemm, parts, c, out)
        self._cache[m] = hit
        return hit

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        A GEMM row and a reduce-scatter row both depend only on the same input
        row, so the added rows produce zeros that the caller slices off; padding
        before the quantiser is what keeps the scale's column-major ``[K/128, M]``
        layout intact without touching it.
        """
        m, k = x.shape
        if self._pad_in is None or self._pad_in.shape[0] < m_pad:
            self._pad_in = torch.zeros((m_pad, k), dtype=x.dtype, device=x.device)
        buf = self._pad_in[:m_pad]
        buf[:m].copy_(x)
        buf[m:].zero_()
        return buf

    def run(self, q_input, x_scale_raw, weight, weight_scale) -> torch.Tensor:
        """One fused GEMM + all-reduce; returns a view of the window's output."""
        m = q_input.shape[0]
        gemm, parts, c, out = self._compiled(m)
        stream = self._fx.Stream(torch.cuda.current_stream())
        # Both scale buffers are read linearly, in physical order: x_scale is
        # logically [M, K/128] but column-major (that is why sglang wraps it in
        # view_aiter_fused_rms_transposed_fp8_scale), and weight_scale is
        # [N/128, K/128] row-major.
        gemm(
            q_input.contiguous().view(torch.int8).view(-1),
            weight.contiguous().view(torch.int8).view(-1),
            c.view(-1),
            x_scale_raw.reshape(-1),
            weight_scale.reshape(-1),
            m,
            self.n,
            self.dev_comm.ptr,
            self.win.handle,
            stream=stream,
        )
        parts["drain"](self.dev_comm.ptr, self.win.handle, stream=stream)
        parts["reduce"](self.dev_comm.ptr, self.win.handle, stream=stream)
        parts["gather"](self.dev_comm.ptr, self.win.handle, stream=stream)
        return out


def _padded_m(m: int, world_size: int) -> int:
    """M rounded up to a whole number of BLOCK_M row bands per destination."""
    granule = world_size * _BLOCK_M
    return (m + granule - 1) // granule * granule


def _counter_chunks(m_pad: int, world_size: int) -> int:
    """Chunks must divide the row bands per destination, which at small M is
    fewer than ``_MAX_CHUNKS``; take the largest divisor rather than failing."""
    bands = m_pad // (world_size * _BLOCK_M)
    return max(c for c in range(1, min(_MAX_CHUNKS, bands) + 1) if bands % c == 0)


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
    if not (
        2 <= world_size <= 8
        and n % _BLOCK_N == 0
        and k % _SCALE_BK == 0
        and n % _SCALE_BK == 0
    ):
        return False
    m_pad = _padded_m(m, world_size)
    return m_pad >= _MIN_FUSED_M and m >= _MIN_PAD_FILL * m_pad


def fused_wo_b_available() -> bool:
    """Static gate, cheap enough to call per layer."""
    return (
        not _disabled
        and envs.SGLANG_OPT_FUSED_WO_B_AR.get()
        and bool(envs.SGLANG_OPT_FUSED_WO_B_AR_DIR.get())
    )


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

    m, k = x.shape
    n = layer.weight.shape[0]
    world_size = get_tp_group().world_size
    if not _eligible(m, n, k, world_size):
        return None

    m_pad = _padded_m(m, world_size)
    try:
        if _state is None:
            _state = _FusedWoB(
                envs.SGLANG_OPT_FUSED_WO_B_AR_DIR.get(),
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
