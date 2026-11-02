from __future__ import annotations

import math
from typing import Iterable


def _finite_values(values: Iterable[float]) -> list[float]:
    finite: list[float] = []
    for value in values:
        value = float(value)
        if math.isfinite(value):
            finite.append(value)
    return finite


def mae(values: list[float]) -> float:
    finite = _finite_values(values)
    if not finite:
        return 0.0
    return sum(abs(v) for v in finite) / len(finite)


def mse(values: list[float]) -> float:
    finite = _finite_values(values)
    if not finite:
        return 0.0
    return sum(v * v for v in finite) / len(finite)


def rmse(values: list[float]) -> float:
    return math.sqrt(mse(values))


def summarize_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    summary: dict[str, float] = {}
    for key in keys:
        values = _finite_values(float(item[key]) for item in items if key in item)
        if values:
            summary[f'{key}_mean'] = sum(values) / len(values)
    return summary
