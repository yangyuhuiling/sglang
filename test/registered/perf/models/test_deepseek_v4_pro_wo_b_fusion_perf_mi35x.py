"""MI35x nightly benchmark for DeepSeek-V4-Pro with the fused wo_b GEMM+AllReduce.

`wo_b` is a RowParallelLinear: a block-scale fp8 GEMM followed by an all-reduce
of the [M, hidden] result. `mori.ops.gemm_ar` fuses the two, handing each
destination's row band to the SDMA copy engines as soon as its tiles are done.
This measures the three wire configurations against each other on one shape:

    base    unfused, GEMM + NCCL
    fused   fused, bf16 wire
    fp8     fused, fp8 all-gather leg

Each variant gets its own server (the wire is chosen by environment variable at
launch) and each server is measured the same way: GSM8K first -- which also warms
it -- then bench_one_batch_server.

Both halves are needed. A fused all-reduce that does not actually move bytes is
*faster* than one that does, so a perf table on its own cannot tell an
optimisation from a broken transport; the accuracy gate is what makes the perf
number mean something. mori's own `GemmAllReduceOp.self_test` covers the same
ground at construction, and this file skips the fused variants when it trips,
rather than publishing numbers from a stack that is not doing the work.

Registry: nightly-perf-8-gpu-mi35x-deepseek-v4-pro-wo-b-fusion suite
"""

import json
import os
import subprocess
import unittest
from types import SimpleNamespace
from typing import Dict, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Three 1.6T loads, three GSM8K runs and three benchmark sweeps.
register_amd_ci(
    est_time=21600,
    suite="nightly-perf-8-gpu-mi35x-deepseek-v4-pro-wo-b-fusion",
    nightly=True,
)

MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP8", "sgl-project/DeepSeek-V4-Pro-FP8"
)
# Pro is 1.6T; weight load plus warmup is far longer than Flash 285B.
SERVER_LAUNCH_TIMEOUT = 5400
FLASHMLA_BACKEND = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")

#: Same gate as test/registered/amd/test_deepseek_v4_pro_fp8.py. The fp8 wire
#: costs relL2 2.5e-2 on the gather leg, which is below what this resolves --
#: scoring 10941 tokens of real text put it inside the bf16 run-to-run band.
GSM8K_MIN_ACCURACY = 0.91

#: input 4096 / output 512 is the shape the other MI35x perf suites use, and it
#: is the one that exercises both halves: M=4096 clears the fused path's
#: threshold in prefill, while decode runs at M=batch_size and never fuses.
INPUT_LEN = 4096
OUTPUT_LEN = 512
#: The leading duplicate is a warmup and is dropped from the report. It matters
#: more here than usual: mori compiles a kernel per distinct M on first sight,
#: several seconds each, and that lands on whichever request gets there first.
BATCH_SIZES = ["1", "1", "8", "16", "64"]

#: Mirrors the launch this checkpoint is known to work under -- the one every
#: measurement in the wo_b campaign used. Taking the env from
#: test_deepseek_v4_pro_fp8.py instead looks right and is not: that file targets
#: sgl-project/DeepSeek-V4-Pro-FP8, and without SGLANG_USE_AITER the MoE loader
#: takes a path where w13 arrives half width ("size of tensor a (7168) must
#: match b (3584)") and the server never finishes loading.
COMMON_ENV_VARS = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_USE_ROCM700A": "0",
    "TORCH_BLAS_PREFER_HIPBLASLT": "1",
    "SGLANG_HACK_FLASHMLA_BACKEND": FLASHMLA_BACKEND,
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_OPT_USE_AITER_BATCHED_GEMM": "true",
    # Without this the comm has no SDMA queues and every put silently does
    # nothing -- see the module docstring.
    "MORI_ENABLE_SDMA": "1",
    "MORI_SOCKET_IFNAME": "lo",
}

VARIANTS = [
    {"name": "base", "env": {}},
    {"name": "fused", "env": {"SGLANG_OPT_FUSED_WO_B_AR": "1"}},
    {
        "name": "fp8",
        "env": {
            "SGLANG_OPT_FUSED_WO_B_AR": "1",
            "SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER": "1",
        },
    },
]

SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--attention-backend",
    "dsv4",
    # GSM8K runs 1319 requests at once. Without a cap the scheduler admits far
    # more than the SWA allocator has room for and decode faults with an illegal
    # access in prepare_for_decode -- which is why the repo's own DSV4-Pro test
    # pairs parallel=1319 with this exact limit.
    "--max-running-requests",
    "256",
    "--page-size",
    "256",
    # 0.88 rather than 0.90: the symmetric window is ~700 MiB of VMM memory and
    # sits outside the torch allocator.
    "--mem-fraction-static",
    "0.88",
    "--swa-full-tokens-ratio",
    "0.15",
    "--enforce-shared-experts-fusion",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--chunked-prefill-size",
    "16384",
    # No radix cache: a prefix that survives between points turns a measured
    # prefill into a lookup. Same choice the other MI35x perf suites make.
    "--disable-radix-cache",
]


#: What the layer logs when it gives up. `fused_wo_b` disables itself on any
#: failure -- a mori without SDMA, a weight that is not aiter-preshuffled, a
#: self-test that did not come back with the right sum -- and falls back to the
#: unfused path. That is the right behaviour, and it is also *silent*: the model
#: is correct, GSM8K passes, and the perf table reports a fused variant that
#: never fused. Only the server's own log says so.
_FALLBACK_MARKER = "mori fused wo_b failed and is disabled"


def _fallback_reason(server_log: str) -> Optional[str]:
    """The line where the server gave up on the fused path, if it did."""
    try:
        with open(server_log, errors="ignore") as f:
            for line in f:
                if _FALLBACK_MARKER in line:
                    return line.strip()[-300:]
    except OSError:
        return None
    return None


class TestDeepseekV4ProWoBFusionPerfMI35x(CustomTestCase):
    """One server per wire, GSM8K then bench_one_batch_server on each."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.report: List[str] = []
        cls.accuracy: Dict[str, float] = {}

    @classmethod
    def tearDownClass(cls):
        if cls.report and is_in_ci():
            write_github_step_summary("\n".join(cls.report) + "\n")

    def _launch(self, variant):
        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)
        env.update(variant["env"])
        log_path = f"/tmp/dsv4_pro_wo_b_fusion_{variant['name']}.serverlog"
        log = open(log_path, "w")
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
            env=env,
            return_stdout_stderr=(log, log),
        )
        return process, log_path

    def _gsm8k(self, variant_name):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        accuracy = metrics["accuracy"]
        self.accuracy[variant_name] = accuracy
        print(f"[{variant_name}] gsm8k accuracy={accuracy:.3f}")
        self.assertGreater(
            accuracy,
            GSM8K_MIN_ACCURACY,
            f"{variant_name}: GSM8K {accuracy:.3f} below {GSM8K_MIN_ACCURACY}. "
            f"A fused all-reduce that does not move bytes still answers "
            f"fluently and still measures fast; this is the check that sees it.",
        )

    def _bench(self, variant_name):
        json_output = f"/tmp/dsv4_pro_wo_b_fusion_{variant_name}.json"
        if os.path.exists(json_output):
            os.remove(json_output)
        cmd = [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            self.base_url,
            "--batch-size",
            *BATCH_SIZES,
            "--input-len",
            str(INPUT_LEN),
            "--output-len",
            str(OUTPUT_LEN),
            "--show-report",
            f"--pydantic-result-filename={json_output}",
            "--no-append-to-github-summary",
            "--trust-remote-code",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            self.fail(f"bench_one_batch_server failed (rc={result.returncode})")
        self.assertTrue(os.path.exists(json_output), f"{json_output} not found")
        with open(json_output) as f:
            rows = json.load(f)
        self.assertTrue(rows, "No benchmark results returned")
        if len(rows) > 1 and rows[0]["batch_size"] == rows[1]["batch_size"]:
            rows = rows[1:]  # drop the warmup

        self.report.append(
            f"### {variant_name} (input_len={INPUT_LEN} output_len={OUTPUT_LEN}, "
            f"gsm8k={self.accuracy.get(variant_name, float('nan')):.3f})"
        )
        self.report.append(
            "| batch size | latency (s) | input throughput (tok/s) | "
            "output throughput (tok/s) | ITL (ms) |"
        )
        self.report.append(
            "| ---------- | ----------- | ------------------------ | "
            "------------------------- | -------- |"
        )
        for r in rows:
            bs = r["batch_size"]
            latency = r.get("latency", 0.0)
            in_tp = r.get("input_throughput", 0.0)
            out_tp = r.get("output_throughput", 0.0)
            itl = 1 / (out_tp / bs) * 1000 if out_tp > 0 else float("inf")
            self.report.append(
                f"| {bs} | {latency:.2f} | {in_tp:.2f} | {out_tp:.2f} | {itl:.2f} |"
            )
            print(
                f"[{variant_name}] bs={bs} latency={latency:.2f}s "
                f"in_tp={in_tp:.2f} out_tp={out_tp:.2f} ITL={itl:.2f}ms"
            )

    def _run_variant(self, variant):
        process, log_path = self._launch(variant)
        try:
            self._gsm8k(variant["name"])
            if variant["env"]:
                # After GSM8K, not before: the layer is constructed lazily on
                # the first eligible call, so nothing has decided yet at launch.
                reason = _fallback_reason(log_path)
                if reason is not None:
                    self.skipTest(
                        f"{variant['name']}: the fused wo_b disabled itself and "
                        f"ran unfused, so a perf number here would describe the "
                        f"base path under a fused label. Server said: {reason}"
                    )
            self._bench(variant["name"])
        finally:
            kill_process_tree(process.pid)

    def test_a_base(self):
        self._run_variant(VARIANTS[0])

    def test_b_fused(self):
        self._run_variant(VARIANTS[1])

    def test_c_fp8(self):
        self._run_variant(VARIANTS[2])


if __name__ == "__main__":
    unittest.main()
