import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.batch_overlap.single_batch_overlap import (
    MoriAiterSboWorkspace,
    SboFlags,
    compute_overlap_args,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
)
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    CombineDtype,
    _MoriEPDispatcherImplNormal,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestMoriAiterSbo(CustomTestCase):
    def _runtime_scope(self):
        return get_flags().moe.override(
            sbo_enabled=True,
            a2a_backend=MoeA2ABackend.MORI,
            runner_backend=MoeRunnerBackend.AITER,
            deepep_mode=DeepEPMode.NORMAL,
        )

    def test_flag_requires_explicit_opt_in(self):
        with self._runtime_scope(), envs.SGLANG_OPT_MORI_AITER_SBO.override(False):
            self.assertFalse(SboFlags.enable_mori_aiter_tile_pipeline())

    def test_missing_capability_falls_back_without_other_sbo_variants(self):
        with (
            self._runtime_scope(),
            envs.SGLANG_OPT_MORI_AITER_SBO.override(True),
            patch(
                "sglang.srt.batch_overlap.single_batch_overlap._mori_aiter_sbo_capable",
                return_value=False,
            ),
        ):
            self.assertTrue(SboFlags.mori_aiter_tile_pipeline_requested())
            self.assertFalse(SboFlags.enable_mori_aiter_tile_pipeline())
            self.assertFalse(SboFlags.enable_dispatch_shared_one_stream_overlap())
            self.assertFalse(SboFlags.enable_combine_shared_two_stream_overlap())
            self.assertFalse(SboFlags.fuse_shared_experts_inside_sbo())

    def test_workspace_is_grow_only_and_shared_by_producer_consumer(self):
        if not torch.cuda.is_available():
            self.skipTest("requires an AMD GPU")

        dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((128, 7168), dtype=torch.bfloat16, device="cuda"),
            topk_ids=torch.zeros((128, 6), dtype=torch.int32, device="cuda"),
        )
        workspace = MoriAiterSboWorkspace()
        alt_stream = torch.cuda.Stream()

        with (
            self._runtime_scope(),
            envs.SGLANG_OPT_MORI_AITER_SBO.override(True),
            patch(
                "sglang.srt.batch_overlap.single_batch_overlap._mori_aiter_sbo_capable",
                return_value=True,
            ),
        ):
            combine, down, meta = compute_overlap_args(
                dispatch_output, alt_stream, workspace
            )
            route_ptr = combine.route_tiles.data_ptr()
            state_ptr = combine.tile_state.data_ptr()
            smaller = SimpleNamespace(
                hidden_states=dispatch_output.hidden_states[:64],
                topk_ids=dispatch_output.topk_ids[:64],
            )
            combine2, down2, _ = compute_overlap_args(smaller, alt_stream, workspace)

        self.assertEqual(meta["mori_sbo_abi_version"], 1)
        self.assertEqual(combine.expected_n_tiles, 56)
        self.assertEqual(combine.route_tiles.shape, (128, 6))
        self.assertEqual(combine.tile_state.numel(), 1560)
        self.assertIs(combine.route_tiles, down.route_tiles)
        self.assertIs(combine.tile_state, down.tile_state)
        self.assertEqual(combine2.route_tiles.data_ptr(), route_ptr)
        self.assertEqual(combine2.tile_state.data_ptr(), state_ptr)
        self.assertEqual(combine2.route_tiles.shape, (64, 6))
        self.assertEqual(combine2.tile_state.numel(), 1548)
        self.assertIs(combine2.route_tiles, down2.route_tiles)

    def test_empty_aiter_producer_records_start_event(self):
        start_event = MagicMock()
        overlap = SimpleNamespace(abi_version=1, start_event=start_event)
        runner = AiterRunnerCore(SimpleNamespace(no_combine=False))
        runner_input = AiterRunnerInput(
            hidden_states=torch.empty((0, 7168), dtype=torch.bfloat16),
            topk_ids=torch.empty((0, 6), dtype=torch.int32),
            topk_weights=torch.empty((0, 6), dtype=torch.float32),
            quant_type=AiterQuantType.PER_1X32,
        )
        empty = torch.empty(0)
        quant_info = AiterMoeQuantInfo(w13_weight=empty, w2_weight=empty)

        output = runner.run(
            runner_input,
            quant_info,
            {"down_gemm_overlap_args": overlap},
        )

        start_event.record.assert_called_once_with()
        self.assertEqual(output.hidden_states.shape, (0, 7168))

    def test_mori_effective_dtype_gate_is_evaluated_lazily(self):
        dispatcher = _MoriEPDispatcherImplNormal.__new__(
            _MoriEPDispatcherImplNormal
        )
        dispatcher.router_topk = 6
        dispatcher.num_local_experts = 48
        dispatcher.hidden_size = 7168
        dispatcher.enable_sdma = False
        dispatcher.use_external_inp_buf = True
        dispatcher._apply_dispatch_dtype_override = MagicMock()
        dispatcher.combine_dtype = CombineDtype.bf16
        self.assertFalse(dispatcher.supports_mori_aiter_sbo_v1())

        dispatcher.combine_dtype = CombineDtype.fp8
        self.assertTrue(dispatcher.supports_mori_aiter_sbo_v1())


if __name__ == "__main__":
    unittest.main()
