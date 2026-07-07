from types import SimpleNamespace

import torch

from vllm_ascend.attention.utils import split_decodes_and_prefills


def test_split_decodes_and_prefills_accepts_common_metadata_without_pcp():
    common_metadata = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=5,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
    )

    assert split_decodes_and_prefills(common_metadata) == (2, 1, 2, 3)
