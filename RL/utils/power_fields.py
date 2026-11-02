from __future__ import annotations

import math
from typing import Mapping, MutableMapping

from RL.utils.constants import ELECTRICAL_POWER_VARIABLE, REAL_POWER_VARIABLE


def finite_float_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def synchronized_power_values(values: Mapping[str, object]) -> tuple[float | None, float | None]:
    real_power = finite_float_or_none(values.get(REAL_POWER_VARIABLE))
    electrical_power = finite_float_or_none(values.get(ELECTRICAL_POWER_VARIABLE))
    if real_power is None:
        real_power = electrical_power
    if electrical_power is None:
        electrical_power = real_power
    return real_power, electrical_power


def synchronize_power_fields(payload: MutableMapping[str, float | None], default: float = 0.0) -> None:
    real_power, electrical_power = synchronized_power_values(payload)
    payload[REAL_POWER_VARIABLE] = default if real_power is None else real_power
    payload[ELECTRICAL_POWER_VARIABLE] = default if electrical_power is None else electrical_power