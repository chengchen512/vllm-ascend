from dataclasses import fields
from functools import cache
from typing import Any

_NUM_SPEC_TOKENS_TO_SCHEDULE = "num_spec_tokens_to_schedule"


@cache
def _field_names(output_cls: type[Any]) -> frozenset[str]:
    return frozenset(field.name for field in fields(output_cls))


def get_num_spec_tokens_to_schedule(
    scheduler: Any,
    num_scheduled_tokens: dict[str, int],
) -> int:
    num_spec_tokens = getattr(scheduler, "num_spec_tokens", 0)
    dynamic_sd_lookup = getattr(scheduler, "dynamic_sd_lookup", None)
    if dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
        return dynamic_sd_lookup[len(num_scheduled_tokens)]
    return num_spec_tokens


def get_effective_lookahead_tokens(scheduler: Any, load_kv_async: bool) -> int:
    if load_kv_async and getattr(scheduler, "use_eagle", False):
        return 0
    return getattr(scheduler, "num_lookahead_tokens", 0)


def scheduler_output_compat_kwargs(
    output_cls: type[Any],
    scheduler: Any,
    num_scheduled_tokens: dict[str, int],
) -> dict[str, int]:
    if _NUM_SPEC_TOKENS_TO_SCHEDULE not in _field_names(output_cls):
        return {}

    return {
        _NUM_SPEC_TOKENS_TO_SCHEDULE: get_num_spec_tokens_to_schedule(
            scheduler,
            num_scheduled_tokens,
        ),
    }
