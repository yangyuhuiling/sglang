import os
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.moe.aiter_tp_moe as tp_moe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, name: str):
        self.name = name

    def is_decode_or_idle(self) -> bool:
        return self.name in ("decode", "idle")


class _PaddingMode:
    def is_max_len(self) -> bool:
        return True


def _moe():
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_fp4_expert=True),
        w13_weight=object(),
        w13_weight_scale_inv=object(),
        w2_weight=object(),
        w2_weight_scale_inv=object(),
    )
    return SimpleNamespace(
        experts=experts,
        is_nextn=False,
        num_fused_shared_experts=0,
        n_shared_experts=1,
    )


def _forward_batch(mode: str):
    return SimpleNamespace(
        forward_mode=_ForwardMode(mode),
        dp_padding_mode=_PaddingMode(),
    )


@contextmanager
def _eligible_runtime():
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(os.environ, {"SGLANG_OPT_USE_AITER_TP_MOE_STAGE1": "1"})
        )
        stack.enter_context(patch.object(tp_moe, "is_hip", return_value=True))
        stack.enter_context(
            patch.object(tp_moe, "is_gfx95_supported", return_value=True)
        )
        stack.enter_context(
            patch.object(
                tp_moe,
                "get_moe_runner_backend",
                return_value=SimpleNamespace(is_aiter=lambda: True),
            )
        )
        stack.enter_context(
            patch.object(
                tp_moe,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            )
        )
        stack.enter_context(
            patch.object(
                tp_moe,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=8, attn_dp_size=8, attn_tp_size=1),
            )
        )
        stack.enter_context(
            patch.object(tp_moe, "get_is_capture_mode", return_value=False)
        )
        stack.enter_context(
            patch.object(
                tp_moe,
                "get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
            )
        )
        yield


class TestAiterTpMoeEligibility(CustomTestCase):
    def test_feature_is_disabled_by_default(self):
        with patch.dict(os.environ, {"SGLANG_OPT_USE_AITER_TP_MOE_STAGE1": "0"}):
            reason = tp_moe.tp_moe_stage1_prequant_disable_reason(
                _moe(), _forward_batch("decode"), shared_expert_is_local=True
            )

        self.assertEqual(reason, "feature disabled")

    def test_decode_and_idle_ranks_take_the_same_fast_path(self):
        with _eligible_runtime():
            for mode in ("decode", "idle"):
                with self.subTest(mode=mode):
                    reason = tp_moe.tp_moe_stage1_prequant_disable_reason(
                        _moe(),
                        _forward_batch(mode),
                        shared_expert_is_local=True,
                    )
                    self.assertIsNone(reason)

    def test_non_decode_mode_falls_back_before_collectives(self):
        with _eligible_runtime():
            reason = tp_moe.tp_moe_stage1_prequant_disable_reason(
                _moe(),
                _forward_batch("target_verify"),
                shared_expert_is_local=True,
            )

        self.assertEqual(reason, "phase 1 supports decode/idle steps only")

    def test_shared_expert_must_stay_local(self):
        with _eligible_runtime():
            reason = tp_moe.tp_moe_stage1_prequant_disable_reason(
                _moe(),
                _forward_batch("decode"),
                shared_expert_is_local=False,
            )

        self.assertEqual(reason, "shared expert must use the DP-local TP1 path")


if __name__ == "__main__":
    unittest.main()
