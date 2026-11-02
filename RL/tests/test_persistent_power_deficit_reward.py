from __future__ import annotations

import unittest

import numpy as np

from RL.utils.constants import STATE_DIM
from RL.utils.reward import RewardFunction


class TestPersistentPowerDeficitReward(unittest.TestCase):
    @staticmethod
    def _reward(weight: float = 1.0) -> RewardFunction:
        return RewardFunction({
            'rated_power': 10_000_000.0,
            'reward': {
                'power_weight': 0.0,
                'blade_root_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
                'persistent_power_deficit_weight': weight,
                'persistent_power_deficit_ewma_tau_s': 1.0,
                'persistent_power_deficit_deadband_fraction': 0.01,
                'persistent_power_deficit_clip': 0.2,
            },
        })

    @staticmethod
    def _compute(reward: RewardFunction, power: float, reference: float, time_t: float, time_next: float):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        return reward.compute(
            state_t=state,
            state_next=state,
            reward_metrics={'Electrical power': power},
            action_t=np.zeros(2, dtype=np.float32),
            prev_action_t=np.zeros(2, dtype=np.float32),
            wind_t=10.0,
            power_t=power,
            sim_time_t=time_t,
            sim_time_next=time_next,
            power_reference_raw=reference,
        )

    def test_sustained_deficit_accumulates_and_recovery_decays(self) -> None:
        reward = self._reward()
        first, first_breakdown = self._compute(reward, 8_000_000.0, 10_000_000.0, 0.0, 0.1)
        second, second_breakdown = self._compute(reward, 8_000_000.0, 10_000_000.0, 0.1, 0.2)
        recovered, recovered_breakdown = self._compute(reward, 10_000_000.0, 10_000_000.0, 0.2, 0.3)

        self.assertLess(second, first)
        self.assertLess(recovered_breakdown['reward_persistent_power_deficit_term'], second_breakdown['reward_persistent_power_deficit_term'])
        self.assertLess(recovered, 0.0)
        self.assertEqual(first_breakdown['reward_persistent_power_deficit_raw'], 1_900_000.0)

    def test_disabled_term_does_not_require_reference_or_time(self) -> None:
        reward = self._reward(weight=0.0)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        value, breakdown = reward.compute(
            state_t=state,
            state_next=state,
            reward_metrics={'Electrical power': 8_000_000.0},
            action_t=np.zeros(2, dtype=np.float32),
            prev_action_t=np.zeros(2, dtype=np.float32),
            wind_t=10.0,
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(breakdown['reward_persistent_power_deficit_term'], 0.0)

    def test_enabled_term_requires_reference_and_strict_time(self) -> None:
        reward = self._reward()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'paired power reference'):
            reward.compute(
                state_t=state,
                state_next=state,
                reward_metrics={'Electrical power': 8_000_000.0},
                action_t=np.zeros(2, dtype=np.float32),
                prev_action_t=np.zeros(2, dtype=np.float32),
                wind_t=10.0,
            )
        with self.assertRaisesRegex(ValueError, 'strictly increasing simulation time'):
            self._compute(reward, 8_000_000.0, 10_000_000.0, 0.1, 0.1)

    def test_reset_clears_ewma_state(self) -> None:
        reward = self._reward()
        self._compute(reward, 8_000_000.0, 10_000_000.0, 0.0, 0.1)
        reward.reset_condition()
        _, breakdown = self._compute(reward, 10_000_000.0, 10_000_000.0, 0.1, 0.2)
        self.assertEqual(breakdown['reward_persistent_power_deficit_ewma'], 0.0)


if __name__ == '__main__':
    unittest.main()