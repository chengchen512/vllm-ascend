# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import os

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["HCCL_BUFFSIZE"] = "512"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "wemaster/deepseek_mtp_main_random_bf16"
PROMPTS = [
    " ".join(["The capital of France is"] * 96),
    " ".join(["Hello, my name is Tom, I am"] * 96),
]


@wait_until_npu_memory_free()
def test_deepseek_mtp_tp2_chunked_prefill_full_decode_only():
    sampling_params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=False)

    with VllmRunner(
        MODEL,
        load_format="dummy",
        max_model_len=2048,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.5,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enable_expert_parallel=True,
        block_size=128,
        speculative_config={
            "num_speculative_tokens": 3,
            "method": "deepseek_mtp",
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4],
        },
        async_scheduling=False,
    ) as runner:
        runner.model.generate(PROMPTS, sampling_params)
