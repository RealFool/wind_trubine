from __future__ import annotations

import unittest

from RL.utils.critic_q_health import (
    ActorQHealthGate,
    ActorQHealthGateConfig,
    ActorOptimizerHealthGate,
    ActorOptimizerHealthGateConfig,
    CriticQHealthConfig,
    CriticQHealthMonitor,
)


class CriticQHealthMonitorTests(unittest.TestCase):
    def test_reports_scheduled_success_and_guard_rates(self) -> None:
        monitor = CriticQHealthMonitor(
            CriticQHealthConfig(window_updates=4, max_guard_rate=0.5, min_success_rate=0.5)
        )
        monitor.observe({'critic1_loss': 1.0})
        monitor.observe({'update_skipped_q_guard': 1.0})
        monitor.observe({'critic1_loss': 0.5})
        metrics = monitor.observe({'update_skipped_nonfinite_loss': 1.0})

        self.assertEqual(metrics['critic_q_health_window_scheduled'], 4.0)
        self.assertEqual(metrics['critic_q_health_window_successful'], 2.0)
        self.assertEqual(metrics['critic_q_health_window_guarded'], 1.0)
        self.assertEqual(metrics['critic_q_health_window_success_rate'], 0.5)
        self.assertEqual(metrics['critic_q_health_window_guard_rate'], 0.25)
        self.assertEqual(metrics['critic_q_health_window_ready'], 1.0)

    def test_healthy_window_resets_violation_streak(self) -> None:
        monitor = CriticQHealthMonitor(
            CriticQHealthConfig(window_updates=2, max_guard_rate=0.5, min_success_rate=0.5)
        )
        monitor.observe({'update_skipped_q_guard': 1.0})
        violated = monitor.observe({'update_skipped_q_guard': 1.0})
        partial = monitor.observe({'critic1_loss': 1.0})
        recovered = monitor.observe({'critic1_loss': 1.0})

        self.assertEqual(violated['critic_q_health_consecutive_violations'], 1.0)
        self.assertEqual(partial['critic_q_health_consecutive_violations'], 1.0)
        self.assertEqual(partial['critic_q_health_window_evaluated'], 0.0)
        self.assertEqual(recovered['critic_q_health_consecutive_violations'], 0.0)

    def test_fails_after_sustained_threshold_violation(self) -> None:
        monitor = CriticQHealthMonitor(
            CriticQHealthConfig(
                enabled=True,
                window_updates=2,
                max_guard_rate=0.5,
                min_success_rate=0.5,
                fail_after_consecutive_windows=2,
            )
        )
        monitor.observe({'update_skipped_q_guard': 1.0})
        monitor.observe({'update_skipped_q_guard': 1.0})
        monitor.observe({'update_skipped_q_guard': 1.0})
        with self.assertRaisesRegex(RuntimeError, 'Critic Q-health fail-fast'):
            monitor.observe({'update_skipped_q_guard': 1.0})

    def test_continue_mode_marks_degraded_without_aborting(self) -> None:
        monitor = CriticQHealthMonitor(
            CriticQHealthConfig(
                enabled=True,
                window_updates=2,
                max_guard_rate=0.5,
                min_success_rate=0.5,
                fail_after_consecutive_windows=2,
                failure_mode='continue_freeze_actor',
            )
        )
        for _ in range(4):
            metrics = monitor.observe({'update_skipped_q_guard': 1.0})

        self.assertTrue(monitor.degraded)
        self.assertTrue(monitor.failure_triggered)
        self.assertEqual(metrics['critic_q_health_failure_triggered'], 1.0)
        self.assertEqual(metrics['critic_q_health_degraded'], 1.0)
        self.assertEqual(metrics['critic_q_health_continue_freeze_actor'], 1.0)

    def test_actor_gate_stays_closed_after_critic_health_degradation(self) -> None:
        gate = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=True,
            min_successful_critic_updates=2,
            window_updates=2,
        ))
        healthy = {
            'critic1_loss': 1.0,
            'critic1_td_error_abs_mean': 1.0,
            'critic2_td_error_abs_mean': 1.0,
            'critic_q_abs_gap_mean': 1.0,
        }
        gate.observe(healthy)
        gate.observe(healthy)
        self.assertTrue(gate.allows_actor_update)

        metrics = gate.observe({**healthy, 'critic_q_health_degraded': 1.0})

        self.assertFalse(gate.allows_actor_update)
        self.assertEqual(metrics['actor_q_health_critic_degraded'], 1.0)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, 'window_updates'):
            CriticQHealthConfig.from_train_config(
                {'critic_q_health': {'enabled': True, 'window_updates': 0}}
            )


class ActorQHealthGateTests(unittest.TestCase):
    @staticmethod
    def _healthy_metrics(value: float = 1.0) -> dict[str, float]:
        return {
            'critic1_loss': value,
            'critic1_td_error_abs_mean': value,
            'critic2_td_error_abs_mean': value,
            'critic_q_abs_gap_mean': value,
        }

    def test_disabled_gate_preserves_existing_actor_schedule(self) -> None:
        gate = ActorQHealthGate(ActorQHealthGateConfig(enabled=False))
        self.assertTrue(gate.allows_actor_update)

    def test_gate_requires_physical_evidence_and_full_healthy_window(self) -> None:
        blocked = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=False,
            min_successful_critic_updates=2,
            window_updates=2,
        ))
        blocked.observe(self._healthy_metrics())
        blocked.observe(self._healthy_metrics())
        self.assertFalse(blocked.allows_actor_update)

        gate = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=True,
            min_successful_critic_updates=2,
            window_updates=2,
        ))
        gate.observe(self._healthy_metrics())
        self.assertFalse(gate.allows_actor_update)
        metrics = gate.observe(self._healthy_metrics())
        self.assertTrue(gate.allows_actor_update)
        self.assertEqual(metrics['actor_q_health_gate_open'], 1.0)

    def test_increasing_td_error_or_q_gap_keeps_gate_closed(self) -> None:
        gate = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=True,
            min_successful_critic_updates=4,
            window_updates=4,
            max_td_error_relative_increase=0.1,
            max_q_gap_relative_increase=0.1,
        ))
        for value in (1.0, 1.0, 2.0, 2.0):
            metrics = gate.observe(self._healthy_metrics(value))
        self.assertFalse(gate.allows_actor_update)
        self.assertGreater(metrics['actor_q_health_td_error_relative_increase'], 0.1)

    def test_target_q_warning_closes_actor_gate_without_marking_critic_update_failed(self) -> None:
        gate = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=True,
            min_successful_critic_updates=2,
            window_updates=2,
            max_target_q_abs_mean=10.0,
        ))
        gate.observe({**self._healthy_metrics(), 'target_q_abs_mean': 11.0})
        metrics = gate.observe({**self._healthy_metrics(), 'target_q_abs_mean': 11.0})

        self.assertFalse(gate.allows_actor_update)
        self.assertEqual(metrics['actor_q_health_window_success_rate'], 1.0)
        self.assertEqual(metrics['actor_q_health_target_q_abs_max'], 11.0)
        self.assertEqual(metrics['actor_q_health_target_q_scale_healthy'], 0.0)

    def test_actor_gate_uses_non_overlapping_latched_hysteresis_windows(self) -> None:
        gate = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=True,
            min_successful_critic_updates=4,
            window_updates=4,
            max_td_error_relative_increase=0.1,
            max_q_gap_relative_increase=0.1,
            td_error_relative_denominator_floor=0.05,
            q_gap_relative_denominator_floor=0.02,
            max_td_error_absolute_increase=0.1,
            max_q_gap_absolute_increase=0.1,
            reopen_td_error_relative_increase=0.2,
            reopen_q_gap_relative_increase=0.2,
            reopen_max_td_error_absolute_increase=0.3,
            reopen_max_q_gap_absolute_increase=0.3,
        ))
        for _ in range(4):
            metrics = gate.observe(self._healthy_metrics(1.0))
        self.assertTrue(metrics['actor_q_health_gate_open'])

        for value in (1.0, 1.0, 1.2, 1.2):
            metrics = gate.observe(self._healthy_metrics(value))
        self.assertFalse(metrics['actor_q_health_gate_open'])

        metrics = gate.observe(self._healthy_metrics(1.0))
        self.assertFalse(metrics['actor_q_health_gate_open'])
        metrics = gate.observe(self._healthy_metrics(1.0))
        self.assertFalse(metrics['actor_q_health_gate_open'])
        metrics = gate.observe(self._healthy_metrics(1.2))
        self.assertFalse(metrics['actor_q_health_gate_open'])
        metrics = gate.observe(self._healthy_metrics(1.2))
        self.assertTrue(metrics['actor_q_health_gate_open'])

    def test_relative_trend_floor_does_not_mask_large_absolute_growth(self) -> None:
        gate = ActorQHealthGate(ActorQHealthGateConfig(
            enabled=True,
            physical_gate_passed=True,
            min_successful_critic_updates=4,
            window_updates=4,
            td_error_relative_denominator_floor=0.05,
            q_gap_relative_denominator_floor=0.02,
            max_td_error_absolute_increase=0.01,
            max_q_gap_absolute_increase=0.003,
        ))
        for _ in range(4):
            gate.observe(self._healthy_metrics(1.0))
        for _ in range(3):
            metrics = gate.observe(self._healthy_metrics(1.0))
        metrics = gate.observe(self._healthy_metrics(2.0))
        self.assertFalse(metrics['actor_q_health_gate_open'])

    def test_rejects_invalid_actor_gate_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, 'window_updates'):
            ActorQHealthGateConfig.from_train_config({
                'actor_q_health_gate': {'enabled': True, 'window_updates': 1},
            })


class ActorOptimizerHealthGateTests(unittest.TestCase):
    @staticmethod
    def _config() -> ActorOptimizerHealthGateConfig:
        return ActorOptimizerHealthGateConfig(
            enabled=True,
            recovery_consecutive_healthy_critic_updates=2,
            require_fixed_offline_batch_health_check=True,
            fixed_offline_batch_size=8,
        )

    @staticmethod
    def _healthy() -> dict[str, float]:
        return {'critic1_loss': 1.0, 'critic2_loss': 1.0, 'target_q_abs_mean': 99.0}

    def test_requires_consecutive_healthy_critic_and_fixed_batch_before_opening(self) -> None:
        gate = ActorOptimizerHealthGate(self._config())
        first = gate.observe(self._healthy(), fixed_offline_batch_healthy=True)
        second = gate.observe(self._healthy(), fixed_offline_batch_healthy=True)

        self.assertEqual(first['actor_optimizer_health_gate_open'], 0.0)
        self.assertEqual(second['actor_optimizer_health_gate_open'], 1.0)
        self.assertTrue(gate.allows_actor_update)
        self.assertEqual(second['actor_optimizer_health_recovery_count'], 1.0)

    def test_numerical_guard_closes_without_replacing_actor_execution_policy(self) -> None:
        gate = ActorOptimizerHealthGate(self._config())
        gate.observe(self._healthy(), fixed_offline_batch_healthy=True)
        gate.observe(self._healthy(), fixed_offline_batch_healthy=True)
        closed = gate.observe({'update_skipped_q_guard': 1.0}, fixed_offline_batch_healthy=True)

        self.assertFalse(gate.allows_actor_update)
        self.assertEqual(closed['actor_optimizer_health_gate_closed_reason_code'], 2.0)
        self.assertEqual(closed['actor_optimizer_health_critic_update_numerically_healthy'], 0.0)

    def test_config_requires_fixed_offline_batch_health_check(self) -> None:
        with self.assertRaisesRegex(ValueError, 'requires fixed offline batch'):
            ActorOptimizerHealthGateConfig.from_train_config({
                'actor_optimizer_health_gate': {
                    'schema_version': 'actor_optimizer_health_gate_v1',
                    'enabled': True,
                    'recovery_consecutive_healthy_critic_updates': 2,
                    'require_fixed_offline_batch_health_check': False,
                    'fixed_offline_batch_size': 8,
                }
            })


if __name__ == '__main__':
    unittest.main()