# SPDX-License-Identifier: Apache-2.0
"""Fused GEMM + all-reduce for DSV4 ``wo_b`` on ROCm gfx950, via mori cco.

``wo_b`` is a ``RowParallelLinear``, so today it runs a block-scale fp8 GEMM and
then an NCCL all-reduce of the ``[M, hidden]`` result. At the production prefill
shape -- ``[16384, 7168]`` with K=2048 on 8x MI355X -- the trace says that pair
costs 1491.3us per layer, of which the collective is 1143.3.

The two can overlap. mori's fused kernel writes each destination's row band
straight into a symmetric window and, as soon as a band's tiles are done, hands
it to the SDMA copy engines; the reduce and all-gather then run as their own
kernels. Measured 1146.2us for the same layer, i.e. **345us saved**, and the
decomposition says essentially all of it is the overlap: the GEMM alone is 3.9%
*slower* than aiter's blockscale kernel and the collective alone only 5.9%
faster than NCCL.

Prerequisites, all of which this module checks rather than assumes:

* a mori built with ``BUILD_CCO_SDMA=ON``. Setting ``BUILD_CCO_SDMA=1`` in the
  environment only rebuilds the *device* bitcode; if the host library was built
  without it there are no SDMA queues, every put silently does nothing, and the
  all-reduce quietly produces zeros. Point ``PYTHONPATH`` at a mori built with
  the flag on.
* ``MORI_ENABLE_SDMA=1`` at process start.
* ``SGLANG_OPT_FUSED_WO_B_AR_DIR`` pointing at mori's ``benchmark/cco/flydsl``. The
  kernels are not part of the installed package, so the path is explicit.

Only full prefill chunks engage: M must be a multiple of ``tp_size * 128``,
which keeps this to one compiled kernel. Everything else -- decode, ragged
tails, small batches -- falls back to the ordinary path, and that is expected
rather than a failure.
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
            counter_chunks=8,
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


def _eligible(m: int, n: int, k: int, world_size: int) -> bool:
    return (
        world_size >= 2
        and world_size <= 8
        # A destination's row slice has to be a whole number of BLOCK_M tiles.
        and m % (world_size * _BLOCK_M) == 0
        and n % _BLOCK_N == 0
        and k % _SCALE_BK == 0
        and n % _SCALE_BK == 0
    )


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

    try:
        if _state is None:
            _state = _FusedWoB(
                envs.SGLANG_OPT_FUSED_WO_B_AR_DIR.get(), m_max=m, n=n, k=k
            )
        if m > _state.m_max or n != _state.n or k != _state.k:
            return None
        # Same quantisation the unfused path does. transpose_scale=True makes
        # the quantiser write the group scale in physical [K/128, M] order,
        # which is exactly what the kernel indexes.
        q_input, x_scale = aiter_per1x128_quant(
            x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True
        )
        out = _state.run(q_input, x_scale, layer.weight, layer.weight_scale_inv)
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
