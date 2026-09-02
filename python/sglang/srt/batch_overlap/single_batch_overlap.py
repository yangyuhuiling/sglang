# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.moe.utils import get_deepep_mode, is_sbo_enabled
from sglang.srt.utils import is_blackwell, is_hip


@functools.lru_cache(maxsize=1)
def _mori_aiter_sbo_capable() -> bool:
    try:
        from aiter.ops.triton.moe.sbo import supports_mori_sbo_tile_signal_v1
        from mori.ops.dispatch_combine import EpDispatchCombineOp

        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            return False
        return bool(
            supports_mori_sbo_tile_signal_v1()
            and EpDispatchCombineOp.supports_sbo_tile_signal_v1()
        )
    except (ImportError, AttributeError, RuntimeError):
        return False


class SboFlags:
    # TODO may have: "enable_dispatch_gateup_gemm_two_stream_overlap", ...

    @classmethod
    def enable_combine_down_gemm_two_stream_overlap(cls):
        return (
            is_sbo_enabled()
            # currently only cutedsl backend supports it
            and (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                or (get_moe_runner_backend().is_deep_gemm() and not is_blackwell())
                or cls.enable_mori_aiter_tile_pipeline()
            )
        )

    @classmethod
    def mori_aiter_tile_pipeline_requested(cls):
        return (
            is_sbo_enabled()
            and envs.SGLANG_OPT_MORI_AITER_SBO.get()
            and is_hip()
            and get_moe_a2a_backend().is_mori()
            and get_moe_runner_backend().is_aiter()
            and get_deepep_mode().value == "normal"
        )

    @classmethod
    def enable_mori_aiter_tile_pipeline(cls):
        return cls.mori_aiter_tile_pipeline_requested() and _mori_aiter_sbo_capable()

    @classmethod
    def enable_combine_shared_two_stream_overlap(cls):
        return (
            is_sbo_enabled()
            and not cls.mori_aiter_tile_pipeline_requested()
            and not cls.enable_dispatch_shared_one_stream_overlap()
            and not envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get()
        )

    @classmethod
    def enable_dispatch_shared_one_stream_overlap(cls):
        return (
            is_sbo_enabled()
            and not is_blackwell()
            and not cls.mori_aiter_tile_pipeline_requested()
        )

    @classmethod
    def fuse_shared_experts_inside_sbo(cls):
        return (
            cls.enable_combine_shared_two_stream_overlap()
            or cls.enable_dispatch_shared_one_stream_overlap()
            or cls.enable_mori_aiter_tile_pipeline()
        )


@dataclass
class CombineOverlapArgs:
    # this "overlap" flag means overlapping with down gemm, not the general two-stream overlap
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: Optional[int] = None
    signal: Optional[torch.Tensor] = None
    block_m: Optional[int] = 64
    threshold: Optional[int] = 0
    route_tiles: Optional[torch.Tensor] = None
    tile_state: Optional[torch.Tensor] = None
    expected_n_tiles: int = 0
    abi_version: int = 0


@dataclass
class DownGemmOverlapArgs:
    num_sms: int
    signal: torch.Tensor
    start_event: torch.cuda.Event
    route_tiles: Optional[torch.Tensor] = None
    tile_state: Optional[torch.Tensor] = None
    expected_n_tiles: int = 0
    abi_version: int = 0


@dataclass
class MoriAiterSboWorkspace:
    num_experts: int = 384
    route_tiles: Optional[torch.Tensor] = None
    tile_state: Optional[torch.Tensor] = None
    start_event: Optional[torch.cuda.Event] = None

    def reserve(self, rows: int, topk: int, device: torch.device):
        # DSV4 sorting can pad every global expert to sort_block_m=128 while
        # Stage2 consumes M32 tiles. Reserve that strict v1 worst case.
        tile_capacity = (rows * topk + 31) // 32 + self.num_experts * 4
        if (
            self.route_tiles is None
            or self.route_tiles.device != device
            or self.route_tiles.shape[0] < rows
            or self.route_tiles.shape[1] != topk
        ):
            self.route_tiles = torch.empty(
                (rows, topk), dtype=torch.int32, device=device
            )
        if (
            self.tile_state is None
            or self.tile_state.device != device
            or self.tile_state.numel() < tile_capacity
        ):
            self.tile_state = torch.empty(
                tile_capacity, dtype=torch.int32, device=device
            )
        if self.start_event is None:
            self.start_event = torch.cuda.Event(blocking=False, interprocess=False)
        return (
            self.route_tiles[:rows],
            self.tile_state[:tile_capacity],
            self.start_event,
        )


def compute_overlap_args(dispatch_output, alt_stream, mori_sbo_workspace=None):
    if not (
        SboFlags.enable_combine_down_gemm_two_stream_overlap()
        or SboFlags.enable_combine_shared_two_stream_overlap()
    ):
        return None, None, {}

    hidden_states = dispatch_output.hidden_states

    if SboFlags.enable_mori_aiter_tile_pipeline():
        if hidden_states.ndim != 2 or hidden_states.shape[1] != 7168:
            raise ValueError(
                "MORI AITER SBO v1 requires [recv_capacity, 7168] hidden states"
            )
        topk = dispatch_output.topk_ids.shape[-1]
        if topk != 6:
            raise ValueError(f"MORI AITER SBO v1 requires topk=6, got {topk}")
        tile_m, tile_n = 32, 128
        expected_n_tiles = hidden_states.shape[1] // tile_n
        total_num_sms = torch.cuda.get_device_properties(
            device=hidden_states.device
        ).multi_processor_count
        communicate_num_sms = envs.SGLANG_OPT_MORI_AITER_SBO_COMBINE_BLOCKS.get()
        if communicate_num_sms <= 0 or communicate_num_sms >= total_num_sms:
            raise ValueError(
                "SGLANG_OPT_MORI_AITER_SBO_COMBINE_BLOCKS must be in "
                f"[1, {total_num_sms - 1}], got {communicate_num_sms}"
            )
        # AITER Stage2 keeps its normal launch geometry. MORI's early combine
        # is explicitly capped below the CU count; no unsupported AITER CU
        # partition is advertised through this ABI.
        compute_num_sms = total_num_sms
        if mori_sbo_workspace is None:
            mori_sbo_workspace = MoriAiterSboWorkspace()
        route_tiles, tile_state, start_event = mori_sbo_workspace.reserve(
            hidden_states.shape[0], topk, hidden_states.device
        )
        combine_overlap_args = CombineOverlapArgs(
            overlap=True,
            stream=alt_stream,
            wait_event=start_event,
            num_sms=communicate_num_sms,
            route_tiles=route_tiles,
            tile_state=tile_state,
            expected_n_tiles=expected_n_tiles,
            block_m=tile_m,
            threshold=expected_n_tiles,
            abi_version=1,
        )
        down_gemm_overlap_args = DownGemmOverlapArgs(
            num_sms=compute_num_sms,
            signal=tile_state,
            start_event=start_event,
            route_tiles=route_tiles,
            tile_state=tile_state,
            expected_n_tiles=expected_n_tiles,
            abi_version=1,
        )
        return (
            combine_overlap_args,
            down_gemm_overlap_args,
            {
                "mori_sbo_abi_version": 1,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "expected_n_tiles": expected_n_tiles,
                "compute_num_sms": compute_num_sms,
            },
        )

    num_local_experts, num_tokens_static, hidden_dim = hidden_states.shape

    total_num_sms = torch.cuda.get_device_properties(
        device="cuda"
    ).multi_processor_count

    if envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.is_set():
        communicate_num_sms = envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.get()
    else:
        communicate_num_sms = 32 if is_blackwell() else 3
    compute_num_sms = total_num_sms - communicate_num_sms

    assert alt_stream is not None
    combine_wait_event = torch.cuda.Event()
    combine_overlap_args = CombineOverlapArgs(
        overlap=False,
        num_sms=communicate_num_sms,
        stream=alt_stream,
        wait_event=combine_wait_event,
    )
    meta_overlap_args = dict(
        compute_num_sms=compute_num_sms,
    )
    down_gemm_overlap_args = None

    if SboFlags.enable_combine_down_gemm_two_stream_overlap():
        # TODO use zero_allocator to remove this `torch.zeros` call
        # NOTE ours v2 use uint32 not int32 currently
        if is_blackwell():
            combine_signal = torch.zeros(
                num_local_experts, dtype=torch.uint32, device=hidden_states.device
            )
        else:
            MIN_BLOCK_M = 64
            combine_signal_size = num_local_experts * (
                (num_tokens_static + MIN_BLOCK_M - 1) // MIN_BLOCK_M
            )
            combine_signal = torch.zeros(
                combine_signal_size, dtype=torch.int32, device=hidden_states.device
            )

        down_gemm_overlap_args = DownGemmOverlapArgs(
            signal=combine_signal,
            start_event=combine_wait_event,
            num_sms=compute_num_sms,
        )
        combine_overlap_args.overlap = True
        combine_overlap_args.signal = combine_signal
        combine_overlap_args.threshold = compute_num_sms
    else:
        meta_overlap_args |= dict(
            record_event_after_down=combine_wait_event,
        )

    return combine_overlap_args, down_gemm_overlap_args, meta_overlap_args
