from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class CriticQHealthConfig:
    enabled: bool = False
    window_updates: int = 200
    max_guard_rate: float = 0.05
    min_success_rate: float = 0.90
    fail_after_consecutive_windows: int = 3
    failure_mode: str = 'abort'

    @classmethod
    def from_train_config(cls, train_config: dict[str, Any]) -> CriticQHealthConfig:
        document = train_config.get('critic_q_health', {})
        if not isinstance(document, dict):
            raise ValueError('train.critic_q_health must be an object')
        config = cls(
            enabled=bool(document.get('enabled', False)),
            window_updates=int(document.get('window_updates', 200)),
            max_guard_rate=float(document.get('max_guard_rate', 0.05)),
            min_success_rate=float(document.get('min_success_rate', 0.90)),
            fail_after_consecutive_windows=int(document.get('fail_after_consecutive_windows', 3)),
            failure_mode=str(document.get('failure_mode', 'abort')).strip().lower(),
        )
        if config.window_updates <= 0:
            raise ValueError('train.critic_q_health.window_updates must be positive')
        if not 0.0 <= config.max_guard_rate <= 1.0:
            raise ValueError('train.critic_q_health.max_guard_rate must be in [0, 1]')
        if not 0.0 <= config.min_success_rate <= 1.0:
            raise ValueError('train.critic_q_health.min_success_rate must be in [0, 1]')
        if config.fail_after_consecutive_windows <= 0:
            raise ValueError('train.critic_q_health.fail_after_consecutive_windows must be positive')
        if config.failure_mode not in {'abort', 'continue_freeze_actor'}:
            raise ValueError(
                'train.critic_q_health.failure_mode must be abort or continue_freeze_actor'
            )
        return config


class CriticQHealthMonitor:
    def __init__(self, config: CriticQHealthConfig) -> None:
        self.config = config
        self._window: deque[tuple[bool, bool]] = deque(maxlen=config.window_updates)
        self._consecutive_violations = 0
        self._failure_triggered = False
        self._degraded = False

    def observe(self, update_metrics: dict[str, float]) -> dict[str, float]:
        successful = 'critic1_loss' in update_metrics
        guarded = update_metrics.get('update_skipped_q_guard', 0.0) > 0.0
        self._window.append((successful, guarded))
        scheduled = len(self._window)
        success_rate = sum(success for success, _guarded in self._window) / scheduled
        guard_rate = sum(guard for _success, guard in self._window) / scheduled
        ready = scheduled == self.config.window_updates
        violated = ready and (
            guard_rate >= self.config.max_guard_rate
            or success_rate <= self.config.min_success_rate
        )
        if ready:
            if violated:
                self._consecutive_violations += 1
            else:
                self._consecutive_violations = 0
        failure_reached = (
            self.config.enabled
            and self._consecutive_violations >= self.config.fail_after_consecutive_windows
        )
        failure_triggered = failure_reached and not self._failure_triggered
        if failure_triggered:
            self._failure_triggered = True
            self._degraded = True
        metrics = {
            'critic_q_health_window_scheduled': float(scheduled),
            'critic_q_health_window_successful': float(sum(success for success, _guarded in self._window)),
            'critic_q_health_window_guarded': float(sum(guard for _success, guard in self._window)),
            'critic_q_health_window_success_rate': float(success_rate),
            'critic_q_health_window_guard_rate': float(guard_rate),
            'critic_q_health_window_ready': float(ready),
            'critic_q_health_window_evaluated': float(ready),
            'critic_q_health_window_violated': float(violated),
            'critic_q_health_consecutive_violations': float(self._consecutive_violations),
            'critic_q_health_failure_triggered': float(failure_triggered),
            'critic_q_health_degraded': float(self._degraded),
            'critic_q_health_continue_freeze_actor': float(
                self.config.failure_mode == 'continue_freeze_actor'
            ),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError('Critic Q-health monitor produced non-finite metrics')
        if failure_reached and self.config.failure_mode == 'abort':
            raise RuntimeError(
                'Critic Q-health fail-fast: '
                f'guard_rate={guard_rate:.6f} limit={self.config.max_guard_rate:.6f}, '
                f'success_rate={success_rate:.6f} minimum={self.config.min_success_rate:.6f}, '
                f'window_updates={self.config.window_updates}, '
                f'consecutive_violations={self._consecutive_violations}'
            )
        if ready:
            self._window.clear()
        return metrics

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def failure_triggered(self) -> bool:
        return self._failure_triggered


@dataclass(frozen=True)
class ActorQHealthGateConfig:
    enabled: bool = False
    physical_gate_passed: bool = False
    min_successful_critic_updates: int = 1000
    window_updates: int = 200
    max_guard_rate: float = 0.05
    min_success_rate: float = 0.90
    max_td_error_relative_increase: float = 0.10
    max_q_gap_relative_increase: float = 0.10
    td_error_relative_denominator_floor: float = 0.05
    q_gap_relative_denominator_floor: float = 0.02
    max_td_error_absolute_increase: float = 0.01
    max_q_gap_absolute_increase: float = 0.003
    reopen_td_error_relative_increase: float = 0.15
    reopen_q_gap_relative_increase: float = 0.15
    reopen_max_td_error_absolute_increase: float = 0.015
    reopen_max_q_gap_absolute_increase: float = 0.0045
    max_target_q_abs_mean: float = 0.0

    @classmethod
    def from_train_config(cls, train_config: dict[str, Any]) -> ActorQHealthGateConfig:
        document = train_config.get('actor_q_health_gate', {})
        if not isinstance(document, dict):
            raise ValueError('train.actor_q_health_gate must be an object')
        config = cls(
            enabled=bool(document.get('enabled', False)),
            physical_gate_passed=bool(document.get('physical_gate_passed', False)),
            min_successful_critic_updates=int(document.get('min_successful_critic_updates', 1000)),
            window_updates=int(document.get('window_updates', 200)),
            max_guard_rate=float(document.get('max_guard_rate', 0.05)),
            min_success_rate=float(document.get('min_success_rate', 0.90)),
            max_td_error_relative_increase=float(document.get('max_td_error_relative_increase', 0.10)),
            max_q_gap_relative_increase=float(document.get('max_q_gap_relative_increase', 0.10)),
            td_error_relative_denominator_floor=float(
                document.get('td_error_relative_denominator_floor', 0.05)
            ),
            q_gap_relative_denominator_floor=float(
                document.get('q_gap_relative_denominator_floor', 0.02)
            ),
            max_td_error_absolute_increase=float(
                document.get('max_td_error_absolute_increase', 0.01)
            ),
            max_q_gap_absolute_increase=float(
                document.get('max_q_gap_absolute_increase', 0.003)
            ),
            reopen_td_error_relative_increase=float(
                document.get('reopen_td_error_relative_increase', 0.15)
            ),
            reopen_q_gap_relative_increase=float(
                document.get('reopen_q_gap_relative_increase', 0.15)
            ),
            reopen_max_td_error_absolute_increase=float(
                document.get('reopen_max_td_error_absolute_increase', 0.015)
            ),
            reopen_max_q_gap_absolute_increase=float(
                document.get('reopen_max_q_gap_absolute_increase', 0.0045)
            ),
            max_target_q_abs_mean=float(document.get('max_target_q_abs_mean', 0.0)),
        )
        if config.min_successful_critic_updates <= 0:
            raise ValueError('train.actor_q_health_gate.min_successful_critic_updates must be positive')
        if config.window_updates < 2:
            raise ValueError('train.actor_q_health_gate.window_updates must be at least 2')
        if not 0.0 <= config.max_guard_rate <= 1.0:
            raise ValueError('train.actor_q_health_gate.max_guard_rate must be in [0, 1]')
        if not 0.0 <= config.min_success_rate <= 1.0:
            raise ValueError('train.actor_q_health_gate.min_success_rate must be in [0, 1]')
        if config.max_td_error_relative_increase < 0.0 or not math.isfinite(config.max_td_error_relative_increase):
            raise ValueError('train.actor_q_health_gate.max_td_error_relative_increase must be finite and non-negative')
        if config.max_q_gap_relative_increase < 0.0 or not math.isfinite(config.max_q_gap_relative_increase):
            raise ValueError('train.actor_q_health_gate.max_q_gap_relative_increase must be finite and non-negative')
        non_negative_fields = (
            'td_error_relative_denominator_floor',
            'q_gap_relative_denominator_floor',
            'max_td_error_absolute_increase',
            'max_q_gap_absolute_increase',
            'reopen_td_error_relative_increase',
            'reopen_q_gap_relative_increase',
            'reopen_max_td_error_absolute_increase',
            'reopen_max_q_gap_absolute_increase',
        )
        for field_name in non_negative_fields:
            value = getattr(config, field_name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(
                    f'train.actor_q_health_gate.{field_name} must be finite and non-negative'
                )
        if config.max_target_q_abs_mean < 0.0 or not math.isfinite(config.max_target_q_abs_mean):
            raise ValueError('train.actor_q_health_gate.max_target_q_abs_mean must be finite and non-negative')
        return config


class ActorQHealthGate:
    def __init__(self, config: ActorQHealthGateConfig) -> None:
        self.config = config
        self._window: deque[tuple[bool, bool, float, float, float]] = deque(maxlen=config.window_updates)
        self._successful_critic_updates = 0
        self._gate_open = False
        self._has_opened = False
        self._updates_since_window_evaluation = 0
        self._critic_health_degraded = False

    @staticmethod
    def _relative_half_increase(values: list[float], denominator_floor: float = 0.0) -> float:
        midpoint = len(values) // 2
        first_mean = sum(values[:midpoint]) / midpoint
        second_mean = sum(values[midpoint:]) / (len(values) - midpoint)
        return (second_mean - first_mean) / max(abs(first_mean), denominator_floor, 1.0e-8)

    @staticmethod
    def _absolute_half_increase(values: list[float]) -> float:
        midpoint = len(values) // 2
        first_mean = sum(values[:midpoint]) / midpoint
        second_mean = sum(values[midpoint:]) / (len(values) - midpoint)
        return second_mean - first_mean

    def observe(self, update_metrics: dict[str, float]) -> dict[str, float]:
        self._critic_health_degraded = self._critic_health_degraded or (
            update_metrics.get('critic_q_health_degraded', 0.0) > 0.0
        )
        successful = 'critic1_loss' in update_metrics
        guarded = update_metrics.get('update_skipped_q_guard', 0.0) > 0.0
        if successful:
            self._successful_critic_updates += 1
        td_error = max(
            float(update_metrics.get('critic1_td_error_abs_mean', 0.0)),
            float(update_metrics.get('critic2_td_error_abs_mean', 0.0)),
        )
        q_gap = float(update_metrics.get('critic_q_abs_gap_mean', 0.0))
        target_q_abs_mean = float(update_metrics.get('target_q_abs_mean', 0.0))
        if not all(math.isfinite(value) for value in (td_error, q_gap, target_q_abs_mean)):
            raise RuntimeError('Actor Q-health gate received non-finite diagnostics')
        self._window.append((successful, guarded, td_error, q_gap, target_q_abs_mean))
        self._updates_since_window_evaluation += 1
        if len(self._window) == self.config.window_updates and (
            self._updates_since_window_evaluation >= self.config.window_updates
        ):
            self._update_gate_latch()
            self._updates_since_window_evaluation = 0
        return self.metrics()

    def _update_gate_latch(self) -> None:
        metrics = self._trend_metrics()
        if not metrics['ready']:
            return
        common_healthy = (
            self.config.physical_gate_passed
            and metrics['enough_updates']
            and metrics['guard_rate'] < self.config.max_guard_rate
            and metrics['success_rate'] > self.config.min_success_rate
            and metrics['target_q_healthy']
            and not self._critic_health_degraded
        )
        close_healthy = common_healthy and (
            metrics['td_relative_increase'] <= self.config.max_td_error_relative_increase
            and metrics['q_gap_relative_increase'] <= self.config.max_q_gap_relative_increase
            and metrics['td_absolute_increase'] <= self.config.max_td_error_absolute_increase
            and metrics['q_gap_absolute_increase'] <= self.config.max_q_gap_absolute_increase
        )
        reopen_healthy = common_healthy and (
            metrics['td_relative_increase'] <= self.config.reopen_td_error_relative_increase
            and metrics['q_gap_relative_increase'] <= self.config.reopen_q_gap_relative_increase
            and metrics['td_absolute_increase'] <= self.config.reopen_max_td_error_absolute_increase
            and metrics['q_gap_absolute_increase'] <= self.config.reopen_max_q_gap_absolute_increase
        )
        if self._gate_open:
            self._gate_open = close_healthy
        elif self._has_opened:
            self._gate_open = reopen_healthy
        else:
            self._gate_open = close_healthy
        self._has_opened = self._has_opened or self._gate_open

    def _trend_metrics(self) -> dict[str, float | bool]:
        scheduled = len(self._window)
        ready = scheduled == self.config.window_updates
        values = list(self._window)
        td_values = [row[2] for row in values]
        q_gap_values = [row[3] for row in values]
        td_relative_increase = (
            self._relative_half_increase(
                td_values,
                self.config.td_error_relative_denominator_floor,
            )
            if ready else 0.0
        )
        q_gap_relative_increase = (
            self._relative_half_increase(
                q_gap_values,
                self.config.q_gap_relative_denominator_floor,
            )
            if ready else 0.0
        )
        td_absolute_increase = self._absolute_half_increase(td_values) if ready else 0.0
        q_gap_absolute_increase = self._absolute_half_increase(q_gap_values) if ready else 0.0
        success_rate = sum(row[0] for row in values) / scheduled if scheduled else 0.0
        guard_rate = sum(row[1] for row in values) / scheduled if scheduled else 0.0
        target_q_abs_mean = max((row[4] for row in values), default=0.0)
        enough_updates = self._successful_critic_updates >= self.config.min_successful_critic_updates
        target_q_healthy = (
            self.config.max_target_q_abs_mean <= 0.0
            or target_q_abs_mean < self.config.max_target_q_abs_mean
        )
        return {
            'ready': ready,
            'success_rate': success_rate,
            'guard_rate': guard_rate,
            'target_q_abs_mean': target_q_abs_mean,
            'enough_updates': enough_updates,
            'target_q_healthy': target_q_healthy,
            'td_relative_increase': td_relative_increase,
            'q_gap_relative_increase': q_gap_relative_increase,
            'td_absolute_increase': td_absolute_increase,
            'q_gap_absolute_increase': q_gap_absolute_increase,
        }

    def metrics(self) -> dict[str, float]:
        trend = self._trend_metrics()
        gate_open = self._gate_open and not self._critic_health_degraded
        return {
            'actor_q_health_gate_enabled': float(self.config.enabled),
            'actor_q_health_gate_open': float(not self.config.enabled or gate_open),
            'actor_q_health_gate_latched_open': float(gate_open),
            'actor_q_health_physical_gate_passed': float(self.config.physical_gate_passed),
            'actor_q_health_successful_critic_updates': float(self._successful_critic_updates),
            'actor_q_health_window_ready': float(trend['ready']),
            'actor_q_health_window_success_rate': float(trend['success_rate']),
            'actor_q_health_window_guard_rate': float(trend['guard_rate']),
            'actor_q_health_td_error_relative_increase': float(trend['td_relative_increase']),
            'actor_q_health_q_gap_relative_increase': float(trend['q_gap_relative_increase']),
            'actor_q_health_td_error_absolute_increase': float(trend['td_absolute_increase']),
            'actor_q_health_q_gap_absolute_increase': float(trend['q_gap_absolute_increase']),
            'actor_q_health_target_q_abs_max': float(trend['target_q_abs_mean']),
            'actor_q_health_target_q_scale_healthy': float(trend['target_q_healthy']),
            'actor_q_health_critic_degraded': float(self._critic_health_degraded),
        }

    @property
    def allows_actor_update(self) -> bool:
        return not self._critic_health_degraded and self.metrics()['actor_q_health_gate_open'] > 0.0


ACTOR_OPTIMIZER_HEALTH_GATE_SCHEMA_VERSION = 'actor_optimizer_health_gate_v1'


@dataclass(frozen=True)
class ActorOptimizerHealthGateConfig:
    enabled: bool
    recovery_consecutive_healthy_critic_updates: int
    require_fixed_offline_batch_health_check: bool
    fixed_offline_batch_size: int

    @classmethod
    def from_train_config(cls, train_config: dict[str, Any]) -> ActorOptimizerHealthGateConfig | None:
        document = train_config.get('actor_optimizer_health_gate')
        if document is None:
            return None
        if not isinstance(document, dict):
            raise ValueError('train.actor_optimizer_health_gate must be an object')
        if document.get('schema_version') != ACTOR_OPTIMIZER_HEALTH_GATE_SCHEMA_VERSION:
            raise ValueError('train.actor_optimizer_health_gate schema version is incompatible')
        required = (
            'enabled',
            'recovery_consecutive_healthy_critic_updates',
            'require_fixed_offline_batch_health_check',
            'fixed_offline_batch_size',
        )
        missing = [name for name in required if name not in document]
        if missing:
            raise ValueError(
                'train.actor_optimizer_health_gate is missing required fields: '
                f'{missing}'
            )
        if not isinstance(document['enabled'], bool):
            raise ValueError('train.actor_optimizer_health_gate.enabled must be a boolean')
        if not isinstance(document['require_fixed_offline_batch_health_check'], bool):
            raise ValueError(
                'train.actor_optimizer_health_gate.require_fixed_offline_batch_health_check must be a boolean'
            )
        config = cls(
            enabled=document['enabled'],
            recovery_consecutive_healthy_critic_updates=int(
                document['recovery_consecutive_healthy_critic_updates']
            ),
            require_fixed_offline_batch_health_check=document[
                'require_fixed_offline_batch_health_check'
            ],
            fixed_offline_batch_size=int(document['fixed_offline_batch_size']),
        )
        if config.recovery_consecutive_healthy_critic_updates <= 0:
            raise ValueError(
                'train.actor_optimizer_health_gate.recovery_consecutive_healthy_critic_updates must be positive'
            )
        if config.fixed_offline_batch_size <= 0:
            raise ValueError(
                'train.actor_optimizer_health_gate.fixed_offline_batch_size must be positive'
            )
        if not config.require_fixed_offline_batch_health_check:
            raise ValueError(
                'train.actor_optimizer_health_gate requires fixed offline batch health checking'
            )
        return config


class ActorOptimizerHealthGate:
    """Freeze only Actor optimizer updates when critic numerical health is not trustworthy."""

    def __init__(self, config: ActorOptimizerHealthGateConfig) -> None:
        self.config = config
        self._consecutive_healthy_critic_updates = 0
        self._closed_reason_code = 1
        self._gate_open = not config.enabled
        self._recovery_count = 0

    @staticmethod
    def _critic_update_numerically_healthy(update_metrics: dict[str, float]) -> bool:
        if any(
            update_metrics.get(key, 0.0) > 0.0
            for key in (
                'update_skipped_nonfinite_batch',
                'update_skipped_nonfinite_loss',
                'update_skipped_nonfinite_gradient',
                'update_skipped_nonfinite_parameter',
                'update_skipped_q_guard',
                'critic_q_health_degraded',
                'critic_q_health_failure_triggered',
            )
        ):
            return False
        required_metrics = ('critic1_loss', 'critic2_loss')
        if any(name not in update_metrics for name in required_metrics):
            return False
        return all(math.isfinite(float(update_metrics[name])) for name in required_metrics)

    def observe(
        self,
        update_metrics: dict[str, float],
        *,
        fixed_offline_batch_healthy: bool,
    ) -> dict[str, float]:
        critic_healthy = self._critic_update_numerically_healthy(update_metrics)
        offline_healthy = bool(fixed_offline_batch_healthy)
        healthy = critic_healthy and offline_healthy
        was_open = self._gate_open
        if not self.config.enabled:
            self._gate_open = True
            self._closed_reason_code = 0
        elif healthy:
            self._consecutive_healthy_critic_updates += 1
            if self._consecutive_healthy_critic_updates >= self.config.recovery_consecutive_healthy_critic_updates:
                self._gate_open = True
                self._closed_reason_code = 0
                if not was_open:
                    self._recovery_count += 1
        else:
            self._gate_open = False
            self._consecutive_healthy_critic_updates = 0
            self._closed_reason_code = 2 if not critic_healthy else 3
        return self.metrics(
            critic_update_numerically_healthy=critic_healthy,
            fixed_offline_batch_healthy=offline_healthy,
        )

    def metrics(
        self,
        *,
        critic_update_numerically_healthy: bool = False,
        fixed_offline_batch_healthy: bool = False,
    ) -> dict[str, float]:
        return {
            'actor_optimizer_health_gate_enabled': float(self.config.enabled),
            'actor_optimizer_health_gate_open': float(self._gate_open),
            'actor_optimizer_health_gate_closed_reason_code': float(self._closed_reason_code),
            'actor_optimizer_health_consecutive_healthy_critic_updates': float(
                self._consecutive_healthy_critic_updates
            ),
            'actor_optimizer_health_required_consecutive_healthy_critic_updates': float(
                self.config.recovery_consecutive_healthy_critic_updates
            ),
            'actor_optimizer_health_fixed_offline_batch_healthy': float(fixed_offline_batch_healthy),
            'actor_optimizer_health_critic_update_numerically_healthy': float(
                critic_update_numerically_healthy
            ),
            'actor_optimizer_health_recovery_count': float(self._recovery_count),
        }

    @property
    def allows_actor_update(self) -> bool:
        return self._gate_open

def resolve_actor_gate_mode(train_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve which actor health gate is actually instantiated.

    The legacy queue path historically preferred the optimizer gate whenever it
    was configured and silently skipped the trend gate.  This helper makes that
    precedence explicit and validates explicit overrides.
    """
    trend = train_config.get('actor_q_health_gate', {})
    optimizer = train_config.get('actor_optimizer_health_gate', {})
    if trend is None:
        trend = {}
    if optimizer is None:
        optimizer = {}
    if not isinstance(trend, dict):
        raise ValueError('train.actor_q_health_gate must be an object')
    if not isinstance(optimizer, dict):
        raise ValueError('train.actor_optimizer_health_gate must be an object')
    trend_enabled = bool(trend.get('enabled', False))
    optimizer_enabled = bool(optimizer.get('enabled', False))
    requested = str(train_config.get('actor_gate_mode', 'auto')).strip().lower()
    if requested not in {'auto', 'trend', 'optimizer', 'none'}:
        raise ValueError('train.actor_gate_mode must be auto, trend, optimizer or none')
    note = ''
    if requested == 'auto':
        mode = 'optimizer' if optimizer_enabled else 'trend' if trend_enabled else 'none'
        if optimizer_enabled and trend_enabled:
            note = 'optimizer gate takes precedence; trend gate configured but not instantiated'
    elif requested == 'optimizer':
        if not optimizer_enabled:
            raise ValueError('train.actor_gate_mode=optimizer requires actor_optimizer_health_gate.enabled=true')
        mode = 'optimizer'
    elif requested == 'trend':
        if not trend_enabled:
            raise ValueError('train.actor_gate_mode=trend requires actor_q_health_gate.enabled=true')
        mode = 'trend'
    else:
        mode = 'none'
        if trend_enabled or optimizer_enabled:
            note = 'configured actor health gates disabled by actor_gate_mode=none'
    return {
        'mode': mode,
        'requested': requested,
        'trend_configured': trend_enabled,
        'optimizer_configured': optimizer_enabled,
        'note': note,
    }
