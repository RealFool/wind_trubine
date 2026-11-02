from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


DUAL_CONSTRAINT_SCHEMA_VERSION = 'dual_constraint_controller_v1'
CONSTRAINT_NAMES = ('load_blade_1', 'load_blade_2', 'load_blade_3', 'energy')


@dataclass(frozen=True)
class DualConstraintConfig:
    eta: dict[str, float]
    lambda_max: dict[str, float]
    epsilon_load_per_blade: np.ndarray
    epsilon_energy_percent: float
    initial_lambda: dict[str, float]
    aggregation: str

    @classmethod
    def from_reward_config(cls, reward_config: Mapping[str, Any]) -> DualConstraintConfig | None:
        document = reward_config.get('dual_constraint')
        if document is None:
            return None
        if not isinstance(document, Mapping):
            raise ValueError('reward.dual_constraint must be an object')
        if document.get('schema_version') != DUAL_CONSTRAINT_SCHEMA_VERSION:
            raise ValueError('reward.dual_constraint schema version is incompatible')
        required = ('eta', 'lambda_max', 'epsilon_load_per_blade', 'epsilon_energy_percent', 'initial_lambda', 'aggregation')
        missing = [name for name in required if name not in document]
        if missing:
            raise ValueError(f'reward.dual_constraint is missing required fields: {missing}')
        if document['aggregation'] != 'mean_complete_conditions_at_round_boundary':
            raise ValueError('reward.dual_constraint aggregation is incompatible')
        try:
            eta = {name: float(document['eta'][name]) for name in CONSTRAINT_NAMES}
            lambda_max = {name: float(document['lambda_max'][name]) for name in CONSTRAINT_NAMES}
            initial_lambda = {name: float(document['initial_lambda'][name]) for name in CONSTRAINT_NAMES}
            epsilon_load = np.asarray(document['epsilon_load_per_blade'], dtype=np.float64)
            epsilon_energy_percent = float(document['epsilon_energy_percent'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('reward.dual_constraint numeric fields are invalid') from exc
        if epsilon_load.shape != (3,) or not np.isfinite(epsilon_load).all() or np.any(epsilon_load < 0.0):
            raise ValueError('reward.dual_constraint epsilon_load_per_blade must be three finite non-negative values')
        if not math.isfinite(epsilon_energy_percent) or epsilon_energy_percent < 0.0:
            raise ValueError('reward.dual_constraint epsilon_energy_percent must be finite and non-negative')
        for name in CONSTRAINT_NAMES:
            if not all(math.isfinite(value) and value >= 0.0 for value in (
                eta[name], lambda_max[name], initial_lambda[name],
            )):
                raise ValueError(f'reward.dual_constraint {name} values must be finite and non-negative')
            if initial_lambda[name] > lambda_max[name]:
                raise ValueError(f'reward.dual_constraint initial_lambda exceeds lambda_max for {name}')
        return cls(
            eta=eta,
            lambda_max=lambda_max,
            epsilon_load_per_blade=epsilon_load,
            epsilon_energy_percent=epsilon_energy_percent,
            initial_lambda=initial_lambda,
            aggregation=str(document['aggregation']),
        )


@dataclass
class ConditionConstraintAccumulator:
    pid_energy_j: float = 0.0
    rl_energy_j: float = 0.0
    pid_peak_abs_nm: np.ndarray | None = None
    rl_peak_abs_nm: np.ndarray | None = None

    def observe(
        self,
        *,
        pid_power_w: float,
        rl_power_w: float,
        blade_root_my_nm: np.ndarray,
        delta_s: float,
        pid_peak_abs_nm: np.ndarray,
    ) -> None:
        values = np.asarray(blade_root_my_nm, dtype=np.float64)
        reference = np.asarray(pid_peak_abs_nm, dtype=np.float64)
        if values.shape != (3,) or reference.shape != (3,):
            raise ValueError('Condition constraint load vectors must have length three')
        if not all(math.isfinite(value) for value in (pid_power_w, rl_power_w, delta_s)):
            raise ValueError('Condition constraint power and duration values must be finite')
        if delta_s <= 0.0 or not np.isfinite(values).all() or not np.isfinite(reference).all() or np.any(reference <= 0.0):
            raise ValueError('Condition constraint observations are invalid')
        self.pid_energy_j += float(pid_power_w) * float(delta_s)
        self.rl_energy_j += float(rl_power_w) * float(delta_s)
        self.pid_peak_abs_nm = reference.copy()
        current_abs = np.abs(values)
        self.rl_peak_abs_nm = (
            current_abs
            if self.rl_peak_abs_nm is None
            else np.maximum(self.rl_peak_abs_nm, current_abs)
        )

    def finalize(self, config: DualConstraintConfig) -> dict[str, float]:
        if self.pid_peak_abs_nm is None or self.rl_peak_abs_nm is None:
            raise ValueError('Condition constraint has no complete load observations')
        delta_energy_j = self.rl_energy_j - self.pid_energy_j
        load_delta = self.rl_peak_abs_nm / self.pid_peak_abs_nm - 1.0
        g_load = np.maximum(load_delta - config.epsilon_load_per_blade, 0.0)
        pid_energy_scale = max(abs(self.pid_energy_j), 1.0)
        energy_deficit_percent = max(-delta_energy_j / pid_energy_scale * 100.0, 0.0)
        g_energy = max(energy_deficit_percent - config.epsilon_energy_percent, 0.0)
        return {
            'condition_pid_energy_j': self.pid_energy_j,
            'condition_rl_energy_j': self.rl_energy_j,
            'condition_delta_energy_j': delta_energy_j,
            'condition_energy_deficit_percent': energy_deficit_percent,
            'condition_g_energy': g_energy,
            **{
                f'condition_pid_peak_blade_{index + 1}_nm': float(self.pid_peak_abs_nm[index])
                for index in range(3)
            },
            **{
                f'condition_rl_peak_blade_{index + 1}_nm': float(self.rl_peak_abs_nm[index])
                for index in range(3)
            },
            **{
                f'condition_delta_load_blade_{index + 1}': float(load_delta[index])
                for index in range(3)
            },
            **{
                f'condition_g_load_blade_{index + 1}': float(g_load[index])
                for index in range(3)
            },
        }


class DualConstraintController:
    def __init__(self, config: DualConstraintConfig) -> None:
        self.config = config
        self.lambdas = dict(config.initial_lambda)
        self._complete_condition_violations: list[dict[str, float]] = []

    def step_penalty(self, *, optimization_state_code: int, g_load_increment: float, power_deficit_fraction: float) -> float:
        if optimization_state_code < 0:
            return 0.0
        load_penalty = sum(self.lambdas[f'load_blade_{blade}'] for blade in range(1, 4)) * max(
            float(g_load_increment), 0.0
        )
        energy_penalty = self.lambdas['energy'] * max(float(power_deficit_fraction), 0.0)
        return load_penalty + energy_penalty

    def complete_condition(self, metrics: Mapping[str, float]) -> dict[str, float]:
        violation = {
            'load_blade_1': float(metrics['condition_g_load_blade_1']),
            'load_blade_2': float(metrics['condition_g_load_blade_2']),
            'load_blade_3': float(metrics['condition_g_load_blade_3']),
            'energy': float(metrics['condition_g_energy']),
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in violation.values()):
            raise ValueError('Condition dual constraint violations must be finite and non-negative')
        self._complete_condition_violations.append(violation)
        return {f'dual_lambda_{name}': self.lambdas[name] for name in CONSTRAINT_NAMES}

    def update_at_round_boundary(self) -> dict[str, float]:
        if not self._complete_condition_violations:
            return {
                'dual_constraint_updated': 0.0,
                **{f'dual_lambda_{name}': self.lambdas[name] for name in CONSTRAINT_NAMES},
            }
        means = {
            name: float(np.mean([row[name] for row in self._complete_condition_violations]))
            for name in CONSTRAINT_NAMES
        }
        for name in CONSTRAINT_NAMES:
            self.lambdas[name] = float(np.clip(
                self.lambdas[name] + self.config.eta[name] * means[name],
                0.0,
                self.config.lambda_max[name],
            ))
        count = len(self._complete_condition_violations)
        self._complete_condition_violations.clear()
        return {
            'dual_constraint_updated': 1.0,
            'dual_constraint_complete_conditions': float(count),
            **{f'dual_mean_g_{name}': means[name] for name in CONSTRAINT_NAMES},
            **{f'dual_lambda_{name}': self.lambdas[name] for name in CONSTRAINT_NAMES},
        }