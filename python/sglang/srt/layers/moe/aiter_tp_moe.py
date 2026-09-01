"""Experimental AITER TP-MoE path for DeepSeek-V4 DP-attention decode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import dp_reduce_scatter_tensor
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui_fp8"
_GEMM2_BM = 32
_GEMM2_BN = 128
_GEMM2_BK = 128


def tp_moe_stage1_prequant_disable_reason(
    moe: DeepseekV2MoE,
    forward_batch: ForwardBatch,
    *,
    shared_expert_is_local: bool,
) -> str | None:
    """Return why the phase-1 TP-MoE fast path cannot run."""

    if not envs.SGLANG_OPT_USE_AITER_TP_MOE_STAGE1.get():
        return "feature disabled"
    if not is_hip() or not is_gfx95_supported():
        return "requires ROCm gfx95"
    if not get_moe_runner_backend().is_aiter():
        return "requires the AITER MoE runner"
    if not get_moe_a2a_backend().is_none():
        return "requires the non-EP standard dispatcher"

    parallel = get_parallel()
    if parallel.tp_size not in (4, 8):
        return "requires TP4 or TP8"
    if parallel.tp_size != parallel.attn_dp_size or parallel.attn_tp_size != 1:
        return "requires pure DP-attention over the TP group"

    # DP-attention may run DECODE on one rank and IDLE on the others. The
    # branch must therefore be identical across ranks; IDLE contributes the
    # same MAX_LEN-padded rows with zero routing weights.
    if not forward_batch.forward_mode.is_decode_or_idle():
        return "phase 1 supports decode/idle steps only"
    if (
        forward_batch.dp_padding_mode is None
        or not forward_batch.dp_padding_mode.is_max_len()
    ):
        return "phase 1 requires equal MAX_LEN padding"
    if get_is_capture_mode():
        return "phase 1 does not support CUDA Graph capture"
    if moe.is_nextn:
        return "phase 1 does not support speculative draft layers"
    if moe.num_fused_shared_experts:
        return "phase 1 does not support fused shared experts"
    if moe.n_shared_experts and not shared_expert_is_local:
        return "shared expert must use the DP-local TP1 path"
    if get_exec().moe.enable_eplb:
        return "phase 1 does not support EPLB"

    quant_method = getattr(moe.experts, "quant_method", None)
    if not getattr(quant_method, "is_fp4_expert", False):
        return "requires DSV4 MXFP4 routed-expert weights"
    for name in (
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ):
        if not hasattr(moe.experts, name):
            return f"missing routed-expert tensor {name}"
    return None


class AiterTpMoeStage1Bridge:
    """Run AITER TPMoEStage1, GEMM2, and the existing TP reduce-scatter."""

    def __init__(self, moe: DeepseekV2MoE):
        from aiter.ops.flydsl.kernels.mega_moe import TPMoEStage1

        experts = moe.experts
        parallel = get_parallel()
        self.model_dim = int(moe.config.hidden_size)
        self.inter_dim = int(experts.intermediate_size_per_partition)
        self.num_experts = int(moe.config.n_routed_experts)
        self.topk = int(moe.config.num_experts_per_tok)

        w13 = experts.w13_weight
        w13_scale = experts.w13_weight_scale_inv
        w2 = experts.w2_weight
        w2_scale = experts.w2_weight_scale_inv

        expected_w13 = (
            self.num_experts,
            2 * self.inter_dim,
            self.model_dim // 2,
        )
        expected_w2 = (
            self.num_experts,
            self.model_dim,
            self.inter_dim // 2,
        )
        if tuple(w13.shape) != expected_w13:
            raise ValueError(
                f"AITER TP-MoE expected w13 shape {expected_w13}, got {tuple(w13.shape)}"
            )
        if tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"AITER TP-MoE expected w2 shape {expected_w2}, got {tuple(w2.shape)}"
            )
        if not getattr(w13, "is_shuffled", False) or not getattr(
            w2, "is_shuffled", False
        ):
            raise ValueError("AITER TP-MoE requires preshuffled MXFP4 weights")

        self.stage1 = TPMoEStage1(
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            experts=self.num_experts,
            topk=self.topk,
            w1=w13,
            w1_scale=w13_scale,
            group=get_tp_group().device_group,
            tp_size=parallel.tp_size,
            tp_rank=parallel.tp_rank,
            device=w13.device,
            swiglu_limit=float(getattr(moe.config, "swiglu_limit", 0.0) or 0.0),
            stage1_kernel_name=_STAGE1_KERNEL,
        )
        self.w2_u8 = w2.view(torch.uint8)
        self.w2_scale_u8 = w2_scale.view(torch.uint8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

        x_fp8, x_scale = self.stage1.quantize(hidden_states)
        stage1_out = self.stage1.forward_prequant(
            x_fp8,
            x_scale,
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )

        partial = torch.zeros(
            (stage1_out.m_logical, self.model_dim),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        mxfp4_moe_gemm2(
            inter_sorted_quant=stage1_out.inter_sorted_quant,
            inter_sorted_shuffled_scale=stage1_out.inter_sorted_shuffled_scale,
            w2_u8=self.w2_u8,
            w2_scale_u8=self.w2_scale_u8,
            sorted_expert_ids=stage1_out.sorted_expert_ids,
            cumsum_tensor=stage1_out.num_valid_ids,
            sorted_token_ids=stage1_out.sorted_token_ids,
            sorted_weights=stage1_out.sorted_weights,
            out=partial,
            M_logical=stage1_out.m_logical,
            max_sorted=stage1_out.max_sorted,
            NE=self.num_experts,
            D_HIDDEN=self.model_dim,
            D_INTER=self.inter_dim,
            topk=self.topk,
            BM=_GEMM2_BM,
            BN=_GEMM2_BN,
            BK=_GEMM2_BK,
            a_dtype="fp8",
            b_dtype="fp4",
            epilog="atomic",
            SBM=stage1_out.sort_block_m,
            out_dtype="bf16",
        )

        output = torch.empty_like(hidden_states)
        dp_reduce_scatter_tensor(output, partial)
        return output


def run_aiter_tp_moe_stage1(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Route local tokens, run the prequantized TP-MoE path, and return local rows."""

    router_logits = moe.gate(hidden_states)
    topk_kwargs = {"input_ids": input_ids} if moe.is_hash else {}
    topk_output = moe.topk(
        hidden_states,
        router_logits,
        num_token_non_padded=forward_batch.num_token_non_padded,
        **topk_kwargs,
    )
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard(layer_id=moe.layer_id)
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise ValueError(
            "AITER TP-MoE requires standard top-k ids and weights, "
            f"got format={topk_output.format}"
        )

    bridge = getattr(moe, "_aiter_tp_moe_stage1_bridge", None)
    if bridge is None:
        bridge = AiterTpMoeStage1Bridge(moe)
        moe._aiter_tp_moe_stage1_bridge = bridge
    return bridge.forward(
        hidden_states,
        topk_output.topk_weights,
        topk_output.topk_ids,
    )
