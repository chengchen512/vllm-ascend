import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode import init_speculator


def test_init_speculator_routes_dflash_before_eagle(monkeypatch):
    class FakeDflashSpeculator:
        def __init__(self, vllm_config, device):
            self.vllm_config = vllm_config
            self.device = device

    module = types.ModuleType("vllm_ascend.worker.v2.spec_decode.dflash.speculator")
    module.AscendDFlashSpeculator = FakeDflashSpeculator
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator",
        module,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            use_dflash=lambda: True,
            use_eagle=lambda: True,
        ),
    )

    speculator = init_speculator(vllm_config, torch.device("cpu"))

    assert isinstance(speculator, FakeDflashSpeculator)
    assert speculator.vllm_config is vllm_config


def test_scheduler_utils_preserve_dflash_lookahead_and_dynamic_spec_tokens():
    from vllm_ascend.core.scheduler_utils import (
        get_effective_lookahead_tokens,
        scheduler_output_compat_kwargs,
    )

    @dataclass
    class OutputWithoutSpecTokens:
        total_num_scheduled_tokens: int = 0

    @dataclass
    class OutputWithSpecTokens:
        total_num_scheduled_tokens: int = 0
        num_spec_tokens_to_schedule: int = 0

    scheduler = SimpleNamespace(
        num_lookahead_tokens=4,
        num_spec_tokens=8,
        dynamic_sd_lookup=[0, 3, 5],
        use_eagle=False,
    )

    assert get_effective_lookahead_tokens(scheduler, load_kv_async=True) == 4
    assert scheduler_output_compat_kwargs(
        OutputWithSpecTokens,
        scheduler,
        {"req0": 1, "req1": 1},
    ) == {"num_spec_tokens_to_schedule": 5}
    assert scheduler_output_compat_kwargs(
        OutputWithoutSpecTokens,
        scheduler,
        {"req0": 1},
    ) == {}

    scheduler.use_eagle = True
    assert get_effective_lookahead_tokens(scheduler, load_kv_async=True) == 0


def test_build_attn_metadata_passes_dflash_causal_flag():
    captured = {}

    class FakeBuilder:
        def build(self, common_prefix_len, common_attn_metadata, **kwargs):
            captured["causal"] = common_attn_metadata.causal
            return SimpleNamespace()

    class FakeGroup:
        layer_names = ["draft_layer"]

        def get_metadata_builder(self, *args):
            return FakeBuilder()

    metadata = build_attn_metadata(
        attn_groups=[[FakeGroup()]],
        num_reqs=1,
        num_tokens=2,
        query_start_loc_gpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_seq_len=8,
        block_tables=[torch.zeros((1, 1), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 2), dtype=torch.int32),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace()]),
        causal=False,
    )

    assert "draft_layer" in metadata
    assert captured["causal"] is False


def test_dflash_v2_runner_aux_hidden_state_config():
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
            ),
        ),
    )

    assert runner._dflash_uses_aux_hidden_state() is True
    assert runner._get_dflash_aux_layers_from_config() == (2, 10, 18, 26, 34)

    hf_config = SimpleNamespace(
        dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
    )
    NPUModelRunner._configure_dflash_aux_hidden_state_layers(
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="dflash",
                draft_model_config=SimpleNamespace(hf_config=hf_config),
            ),
        ),
    )
    assert hf_config.eagle_aux_hidden_state_layer_ids == [2, 10, 18, 26, 34]

    runner.speculative_config.draft_model_config.hf_config.dflash_config[
        "use_aux_hidden_state"
    ] = False
    assert runner._dflash_uses_aux_hidden_state() is False
