from types import SimpleNamespace

import torch

from vllm_ascend.patch.worker.patch_qwen3_dflash import (
    precompute_and_store_context_kv,
)


def test_precompute_context_kv_uses_rotary_return_value():
    captured = {}

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class FakeRotary(torch.nn.Module):
        def forward(self, positions, query, key):
            assert key is not None
            return query + 100, key

    class FakeImpl:
        def do_kv_cache_update(
            self,
            attn,
            key,
            value,
            kv_cache,
            slot_mapping,
        ):
            captured["key"] = key.clone()
            captured["value"] = value.clone()
            captured["slot_mapping"] = slot_mapping.clone()

    fake_self = SimpleNamespace(
        hidden_norm=Identity(),
        _num_attn_layers=1,
        _kv_size=2,
        _head_dim=2,
        _num_kv_heads=1,
        _fused_kv_weight=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
        _fused_kv_bias=None,
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    k_norm=Identity(),
                    rotary_emb=FakeRotary(),
                ),
            ),
        ],
        _attn_layers=[
            SimpleNamespace(
                impl=FakeImpl(),
                kv_cache=object(),
            ),
        ],
    )

    context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    context_positions = torch.tensor([7, 8], dtype=torch.int64)
    context_slot_mapping = torch.tensor([11, 12], dtype=torch.int32)

    precompute_and_store_context_kv(
        fake_self,
        context_states,
        context_positions,
        context_slot_mapping,
    )

    torch.testing.assert_close(
        captured["key"],
        torch.tensor([[[101.0, 102.0]], [[103.0, 104.0]]]),
    )
    torch.testing.assert_close(
        captured["value"],
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
    )
    torch.testing.assert_close(captured["slot_mapping"], context_slot_mapping)
