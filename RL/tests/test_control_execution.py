from __future__ import annotations

import unittest

import numpy as np

from RL.train_rl import _should_store_transition
from RL.utils.constants import OBS_DIM, OBSERVATION_VARIABLES
from RL.utils.control_execution import (
    EXECUTION_REPLAY_CONTRACT_SCHEMA_VERSION,
    PreSendExecutionMode,
    post_step_outcome_metrics,
    pre_send_execution_mode_code,
)
from RL.utils.safe_residual_control import SafeResidualController


class TestControlExecution(unittest.TestCase):
    @staticmethod
    def _controller() -> SafeResidualController:
        controller = SafeResidualController(
            {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.1],
                'action_representation': 'residual',
                'residual_limit': [1_000_000.0, 0.1],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
                'ood_threshold': 1.0e12,
                'low_speed_protection_mode': 'disabled',
                'pitch_protection_mode': 'disabled',
                'low_power_threshold': None,
            },
        )
        controller.set_observation_distribution(np.zeros(OBS_DIM), np.ones(OBS_DIM))
        return controller

    @staticmethod
    def _observation(*, rotor_speed: float = 0.8, power_w: float = 1_000_000.0) -> np.ndarray:
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        observation[OBSERVATION_VARIABLES.index('I_RotorSpeed')] = rotor_speed
        observation[OBSERVATION_VARIABLES.index('I_RealPower')] = power_w
        return observation

    def test_optimizer_gate_closure_does_not_replace_valid_actor_execution(self) -> None:
        controller = self._controller()
        observation = self._observation()
        controller.reset_condition(observation, previous_action_real=np.asarray([20_000_000.0, 0.1]))

        action = controller.action_from_policy(
            np.asarray([1.0, 0.0]),
            observation,
            allow_policy=True,
            actor_execution_valid=True,
        )

        self.assertEqual(action.pre_send_execution_mode, PreSendExecutionMode.ACTOR)
        self.assertTrue(action.residual_admitted)
        self.assertFalse(action.fallback_active)
        self.assertGreater(float(action.action_real[0]), 20_000_000.0)

    def test_physical_protection_priority_selects_one_mode(self) -> None:
        controller = self._controller()
        observation = self._observation()
        controller.reset_condition(observation)

        rollback = controller.action_from_policy(
            np.asarray([1.0, 0.0]),
            observation,
            allow_policy=True,
            actor_execution_valid=False,
        )
        explicit = controller.action_from_policy(
            np.asarray([1.0, 0.0]),
            observation,
            allow_policy=False,
            baseline_execution_mode=PreSendExecutionMode.EXPLICIT_BASELINE_STAGE,
        )

        self.assertEqual(rollback.pre_send_execution_mode, PreSendExecutionMode.ACTOR_EXECUTION_ROLLBACK_BASELINE)
        self.assertEqual(explicit.pre_send_execution_mode, PreSendExecutionMode.EXPLICIT_BASELINE_STAGE)
        self.assertFalse(rollback.residual_admitted)
        self.assertFalse(explicit.residual_admitted)

    def test_execution_mode_replay_rejects_outcomes_and_non_replay_modes(self) -> None:
        train_config = {
            'execution_replay_contract': {
                'schema_version': EXECUTION_REPLAY_CONTRACT_SCHEMA_VERSION,
                'use_execution_mode_replay': True,
                'td3_critic_replay_modes': [
                    'actor',
                    'explicit_baseline_stage',
                    'actor_execution_rollback_baseline',
                    'low_speed_recovery_hold',
                ],
                'reject_post_step_outcomes': True,
            }
        }
        base = {
            'transition_store_eligible': 1.0,
            'control_pre_send_execution_mode': float(
                pre_send_execution_mode_code(PreSendExecutionMode.ACTOR)
            ),
            **post_step_outcome_metrics({}, unsafe_stop=False),
        }
        self.assertTrue(_should_store_transition(base, train_config))

        recovery = {
            **base,
            'control_pre_send_execution_mode': float(
                pre_send_execution_mode_code(PreSendExecutionMode.LOW_SPEED_RECOVERY_HOLD)
            ),
        }
        self.assertTrue(_should_store_transition(recovery, train_config))

        ood = {
            **base,
            'control_pre_send_execution_mode': float(
                pre_send_execution_mode_code(PreSendExecutionMode.OOD_BASELINE)
            ),
        }
        self.assertFalse(_should_store_transition(ood, train_config))

        timeout = {**base, 'control_post_step_timeout': 1.0}
        self.assertFalse(_should_store_transition(timeout, train_config))
        unsafe = {**base, 'control_post_step_unsafe_stop': 1.0}
        self.assertFalse(_should_store_transition(unsafe, train_config))


if __name__ == '__main__':
    unittest.main()