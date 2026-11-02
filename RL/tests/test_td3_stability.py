from __future__ import annotations

import json
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from RL.agents.td3 import TD3Agent
from RL.models.critic_mlp import CriticMLP
from RL.utils.constants import STATE_DIM, STATE_VARIABLES
from RL.utils.operating_state import OptimizationState
from RL.utils.reward import RewardFunction


class TestTd3Stability(unittest.TestCase):
    def test_terminal_mask_suppresses_bootstrap_in_critic_target(self) -> None:
        class ConstantActor(torch.nn.Module):
            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

        class ConstantCritic(torch.nn.Module):
            def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                return torch.full((obs.shape[0], 1), 4.0, dtype=obs.dtype, device=obs.device)

        agent = TD3Agent(
            obs_dim=3,
            action_dim=2,
            config={'hidden_dims': [8], 'gamma': 0.5, 'policy_noise': 0.0},
            device=torch.device('cpu'),
        )
        agent.actor_target = ConstantActor()
        agent.critic1_target = ConstantCritic()
        agent.critic2_target = ConstantCritic()
        batch = SimpleNamespace(
            obs=torch.zeros(2, 3),
            actions=torch.zeros(2, 2),
            rewards=torch.ones(2, 1),
            next_obs=torch.zeros(2, 3),
            dones=torch.tensor([[0.0], [1.0]]),
        )

        metrics = agent.update(batch, update_actor=False)

        self.assertAlmostEqual(metrics['target_q_mean'], 2.0)

    def test_critic_uses_configured_observation_normalizer(self) -> None:
        critic = CriticMLP(obs_dim=2, action_dim=1, hidden_dims=[4])
        critic.set_obs_normalizer(torch.tensor([10.0, 1000.0]), torch.tensor([2.0, 100.0]))
        scaled = critic.normalize_obs(torch.tensor([[12.0, 1200.0]]))
        self.assertTrue(torch.allclose(scaled, torch.tensor([[1.0, 2.0]])))

    def test_critic_only_update_defers_actor(self) -> None:
        config = {
            'hidden_dims': [8],
            'policy_delay': 1,
            'td3_bc_enabled': True,
            'td3_bc_target_ratio': 0.2,
        }
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config, device=torch.device('cpu'))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        batch = SimpleNamespace(
            obs=torch.randn(8, 3),
            actions=torch.randn(8, 2).clamp(-1.0, 1.0),
            rewards=torch.randn(8, 1),
            next_obs=torch.randn(8, 3),
            dones=torch.zeros(8, 1),
        )
        metrics = agent.update(batch, update_actor=False)
        self.assertEqual(metrics['actor_update_deferred'], 1.0)
        self.assertIn('critic_obs_scaled_abs_mean', metrics)
        self.assertNotIn('actor_loss', metrics)

    def test_actor_preflight_closure_applies_immediately_across_same_step_updates(self) -> None:
        agent = TD3Agent(
            obs_dim=3,
            action_dim=2,
            config={'hidden_dims': [8], 'policy_delay': 1, 'policy_noise': 0.0},
            device=torch.device('cpu'),
        )
        batch = SimpleNamespace(
            obs=torch.randn(8, 3),
            actions=torch.randn(8, 2).clamp(-1.0, 1.0),
            rewards=torch.randn(8, 1),
            next_obs=torch.randn(8, 3),
            dones=torch.zeros(8, 1),
        )
        actor_before = {name: value.detach().clone() for name, value in agent.actor.state_dict().items()}
        critic_before = {name: value.detach().clone() for name, value in agent.critic1.state_dict().items()}
        gate = {'open': True, 'preflight_calls': 0}

        def preflight(metrics: dict[str, float]) -> bool:
            gate['preflight_calls'] += 1
            self.assertIn('critic1_loss', metrics)
            # Simulate this Critic update closing the health gate; the same
            # update and the next update in this control step must see it.
            gate['open'] = False
            return gate['open']

        update_metrics = [
            agent.update(batch, update_actor=True, actor_update_preflight=preflight)
            for _ in range(2)
        ]

        self.assertEqual(gate['preflight_calls'], 2)
        self.assertTrue(all(metrics['actor_update_deferred'] == 1.0 for metrics in update_metrics))
        self.assertTrue(all(torch.equal(actor_before[name], value) for name, value in agent.actor.state_dict().items()))
        self.assertTrue(any(
            not torch.equal(critic_before[name], value)
            for name, value in agent.critic1.state_dict().items()
        ))

    def test_smooth_l1_critic_update_reports_td_error(self) -> None:
        agent = TD3Agent(
            obs_dim=3,
            action_dim=2,
            config={'hidden_dims': [8], 'critic_loss_type': 'smooth_l1', 'critic_grad_norm': 5.0},
            device=torch.device('cpu'),
        )
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        batch = SimpleNamespace(
            obs=torch.randn(8, 3),
            actions=torch.randn(8, 2).clamp(-1.0, 1.0),
            rewards=torch.randn(8, 1),
            next_obs=torch.randn(8, 3),
            dones=torch.zeros(8, 1),
        )
        metrics = agent.update(batch, update_actor=False)
        self.assertIn('critic1_td_error_abs_mean', metrics)
        self.assertGreaterEqual(metrics['critic1_td_error_abs_mean'], 0.0)
        self.assertIn('critic_q_abs_gap_mean', metrics)
        self.assertGreaterEqual(metrics['critic_q_abs_gap_mean'], 0.0)

    def test_available_fraction_reward_is_zero_without_available_power(self) -> None:
        reward = RewardFunction({
            'rated_power': 5_000_000.0,
            'reward': {
                'power_reward_mode': 'available_fraction',
                'power_available_mode': 'wind_cubic',
                'rated_wind_speed': 12.0,
                'cut_in_wind_speed': 3.0,
                'power_weight': 1.0,
                'tower_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state_next = state.copy()
        state_next[5] = 100_000.0
        metrics = {
            'Electrical power': 100_000.0,
        }
        _, breakdown = reward.compute(state, state_next, metrics, np.zeros(2), np.zeros(2), wind_t=2.0)
        self.assertEqual(breakdown['reward_power_available_raw'], 0.0)
        self.assertEqual(breakdown['reward_power_term'], 0.0)

    def test_reward_scalar_sanitization_preserves_nonfinite_contract(self) -> None:
        self.assertEqual(RewardFunction._sanitize_scalar(float('nan')), 0.0)
        self.assertEqual(RewardFunction._sanitize_scalar(float('inf')), 1.0e6)
        self.assertEqual(RewardFunction._sanitize_scalar(float('-inf')), -1.0e6)
        self.assertEqual(RewardFunction._sanitize_scalar(12.5), 12.5)

    def test_negative_power_has_no_independent_reward_penalty(self) -> None:
        reward = RewardFunction({
            'rated_power': 5_000_000.0,
            'reward': {
                'power_reward_mode': 'power_ref_fraction',
                'power_ref': 5_000_000.0,
                'power_available_mode': 'wind_cubic',
                'rated_wind_speed': 12.0,
                'cut_in_wind_speed': 3.0,
                'power_weight': 1.0,
                'tower_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)
        positive_reward, positive_breakdown = reward.compute(
            state,
            state.copy(),
            {'Electrical power': 1_000_000.0},
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
        )
        reverse_state = state.copy()
        reverse_state[5] = -1_000_000.0
        reverse_reward, reverse_breakdown = reward.compute(
            state,
            reverse_state,
            {'Electrical power': -1_000_000.0},
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
        )

        self.assertNotIn('reward_negative_power_raw', positive_breakdown)
        self.assertNotIn('reward_negative_power_raw', reverse_breakdown)
        self.assertAlmostEqual(positive_reward, 0.2)
        # Negative power saturates the main power reward at its lower bound;
        # there is no separate C_neg term.
        self.assertAlmostEqual(reverse_reward, 0.0)
        self.assertEqual(reverse_breakdown['reward_power_term'], 0.0)

    def test_rotor_speed_change_has_no_direct_reward_component(self) -> None:
        reward = RewardFunction({
            'rated_power': 5_000_000.0,
            'reward': {
                'power_weight': 0.0,
                'blade_root_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)
        slow_state = state.copy()
        fast_state = state.copy()
        slow_state[reward.rotor_speed_index] = 0.2
        fast_state[reward.rotor_speed_index] = 1.2
        metrics = {
            'Electrical power': 0.0,
            'Blade root 1 My (Nm)': 0.0,
            'Blade root 2 My (Nm)': 0.0,
            'Blade root 3 My (Nm)': 0.0,
        }

        slow_reward, slow_breakdown = reward.compute(
            state, slow_state, metrics, np.zeros(2), np.zeros(2), wind_t=8.0
        )
        fast_reward, fast_breakdown = reward.compute(
            state, fast_state, metrics, np.zeros(2), np.zeros(2), wind_t=8.0
        )

        self.assertEqual(slow_reward, fast_reward)
        self.assertNotIn('reward_rotor_speed_raw', slow_breakdown)
        self.assertNotIn('reward_region_rotor_speed_term', slow_breakdown)
        self.assertNotIn('reward_rotor_speed_variation_term', fast_breakdown)
        self.assertNotIn('reward_sustained_low_rotor_speed_term', fast_breakdown)

    def test_state_conditioned_reward_uses_only_current_reward_components(self) -> None:
        weights = {
            state.value: {
                'power': 0.0,
                'blade_root': 0.0,
                'continuous_load': 0.0,
                'persistent_power_deficit': 0.0,
                'action_continuity': 0.0,
            }
            for state in OptimizationState
        }
        weights['below_rated'].update(power=1.0, blade_root=1.0)
        reward = RewardFunction({
            'rated_power': 5_000_000.0,
            'reward': {
                'power_ref': 2_000_000.0,
                'power_reward_mode': 'power_ref_fraction',
                'power_weight': 1.0,
                'blade_root_weight': 1.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
                'persistent_power_deficit_weight': 0.0,
                'blade_root_continuous_weight': 0.0,
                'state_conditioning': {
                    'schema_version': 'state_conditioned_reward_v1',
                    'weights': weights,
                },
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state_next = state.copy()
        metrics = {
            'Electrical power': 1_000_000.0,
            'Blade root 1 My (Nm)': 0.0,
            'Blade root 2 My (Nm)': 0.0,
            'Blade root 3 My (Nm)': 0.0,
        }
        state_weights = {name: 0.0 for name in weights}
        state_weights['below_rated'] = 1.0

        value, breakdown = reward.compute(
            state,
            state_next,
            metrics,
            np.zeros(2),
            np.zeros(2),
            wind_t=8.0,
            reward_state_weights=state_weights,
        )

        self.assertAlmostEqual(value, 0.5)
        self.assertEqual(breakdown['reward_state_power_multiplier'], 1.0)
        self.assertEqual(breakdown['reward_state_blade_root_multiplier'], 1.0)

    def test_blade_root_reward_penalizes_only_load_above_soft_limit(self) -> None:
        reward = RewardFunction({
            'rated_power': 5_000_000.0,
            'reward': {
                'power_weight': 0.0,
                'tower_weight': 0.0,
                'blade_root_weight': 0.1,
                'blade_root_ref': 7_500_000.0,
                'blade_root_soft_limit': 1.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)
        metrics = {
            'Electrical power': 0.0,
            'Blade root 1 My (Nm)': 9_000_000.0,
            'Blade root 2 My (Nm)': -9_000_000.0,
            'Blade root 3 My (Nm)': 9_000_000.0,
        }
        value, breakdown = reward.compute(state, state, metrics, np.zeros(2), np.zeros(2), wind_t=12.0)

        self.assertAlmostEqual(breakdown['reward_blade_root_raw'], 9_000_000.0)
        self.assertAlmostEqual(breakdown['reward_blade_root_term'], 0.2)
        self.assertAlmostEqual(value, -0.02)

    def test_energy_increment_reward_uses_strict_trapezoidal_simulation_time(self) -> None:
        reward = RewardFunction({
            'rated_power': 15_000_000.0,
            'bladed_step_duration_ms': 20.0,
            'reward': {
                'power_reward_mode': 'energy_increment',
                'energy_reference_j': 300_000.0,
                'power_weight': 1.0,
                'blade_root_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state_next = state.copy()
        value, breakdown = reward.compute(
            state,
            state_next,
            {'Electrical power': 20_000_000.0},
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
            power_t=10_000_000.0,
            sim_time_t=3.00,
            sim_time_next=3.02,
        )

        self.assertAlmostEqual(breakdown['reward_energy_delta_t_s'], 0.02)
        self.assertAlmostEqual(breakdown['reward_energy_increment_j'], 300_000.0)
        self.assertEqual(breakdown['reward_energy_increment_valid'], 1.0)
        self.assertAlmostEqual(breakdown['reward_power_term'], 1.0)
        self.assertAlmostEqual(value, 1.0)

        invalid_value, invalid_breakdown = reward.compute(
            state,
            state_next,
            {'Electrical power': 20_000_000.0},
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
            power_t=10_000_000.0,
            sim_time_t=3.02,
            sim_time_next=3.02,
        )
        self.assertEqual(invalid_breakdown['reward_energy_increment_valid'], 0.0)
        self.assertEqual(invalid_breakdown['reward_energy_increment_j'], 0.0)
        self.assertEqual(invalid_value, 0.0)

    def test_blade_root_peak_increment_penalizes_only_new_pid_reference_excess(self) -> None:
        reward = RewardFunction({
            'rated_power': 15_000_000.0,
            'reward': {
                'power_weight': 0.0,
                'blade_root_weight': 1.0,
                'blade_root_load_mode': 'my_peak_increment',
                'blade_root_peak_reference_nm': [10_000_000.0, 20_000_000.0, 40_000_000.0],
                'blade_root_soft_limit': 1.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
            },
        })
        state = np.zeros(STATE_DIM, dtype=np.float32)

        first_value, first = reward.compute(
            state,
            state,
            {
                'Electrical power': 0.0,
                'Blade root 1 My (Nm)': 12_000_000.0,
                'Blade root 2 My (Nm)': 15_000_000.0,
                'Blade root 3 My (Nm)': 40_000_000.0,
            },
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
        )
        repeated_value, repeated = reward.compute(
            state,
            state,
            {
                'Electrical power': 0.0,
                'Blade root 1 My (Nm)': 12_000_000.0,
                'Blade root 2 My (Nm)': 15_000_000.0,
                'Blade root 3 My (Nm)': 40_000_000.0,
            },
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
        )
        second_value, second = reward.compute(
            state,
            state,
            {
                'Electrical power': 0.0,
                'Blade root 1 My (Nm)': 13_000_000.0,
                'Blade root 2 My (Nm)': 25_000_000.0,
                'Blade root 3 My (Nm)': 40_000_000.0,
            },
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
        )

        self.assertAlmostEqual(first['reward_blade_root_peak_excess_increment'], 0.2)
        self.assertAlmostEqual(first_value, -0.2)
        self.assertEqual(repeated['reward_blade_root_peak_excess_increment'], 0.0)
        self.assertEqual(repeated_value, 0.0)
        self.assertAlmostEqual(second['reward_blade_root_peak_excess_increment'], 0.25)
        self.assertAlmostEqual(second_value, -0.25)
        self.assertAlmostEqual(second['reward_blade_root_2_peak_nm'], 25_000_000.0)
        self.assertAlmostEqual(second['reward_blade_root_2_peak_reference_nm'], 20_000_000.0)

        reward.reset_episode()
        reset_value, reset_breakdown = reward.compute(
            state,
            state,
            {
                'Electrical power': 0.0,
                'Blade root 1 My (Nm)': 13_000_000.0,
                'Blade root 2 My (Nm)': 25_000_000.0,
                'Blade root 3 My (Nm)': 40_000_000.0,
            },
            np.zeros(2),
            np.zeros(2),
            wind_t=12.0,
        )
        self.assertAlmostEqual(reset_breakdown['reward_blade_root_peak_excess_increment'], 0.3)
        self.assertAlmostEqual(reset_value, -0.3)

    def test_pid_excess_continuous_load_term_is_zero_below_pid_reference(self) -> None:
        with tempfile.TemporaryDirectory(prefix='pid_excess_reference_') as tmpdir:
            reference_path = f'{tmpdir}/pid_peaks.json'
            with open(reference_path, 'w', encoding='utf-8') as file:
                json.dump(
                    {
                        'reward': {
                            'blade_root_peak_reference': [
                                {'wind_speed': 3.0, 'peak_abs_nm': [10_000_000.0] * 3},
                                {'wind_speed': 25.0, 'peak_abs_nm': [10_000_000.0] * 3},
                            ]
                        },
                        'metadata': {'data_schema_version': 'bladed_udp_19_si_my_v1'},
                    },
                    file,
                )
            reward = RewardFunction(
                {
                    'rated_power': 15_000_000.0,
                    'reward': {
                        'power_weight': 0.0,
                        'blade_root_weight': 0.0,
                        'blade_root_continuous_weight': 1.0,
                        'blade_root_continuous_mode': 'pid_excess',
                        'pid_blade_root_peak_reference_path': reference_path,
                        'acc_weight': 0.0,
                        'shaft_weight': 0.0,
                        'pitch_rate_weight': 0.0,
                        'torque_rate_weight': 0.0,
                        'command_excess_weight': 0.0,
                        'power_tracking_weight': 0.0,
                    },
                }
            )
            state = np.zeros(STATE_DIM, dtype=np.float32)
            low_value, low = reward.compute(
                state,
                state,
                {
                    'Electrical power': 0.0,
                    'Blade root 1 My (Nm)': 8_000_000.0,
                    'Blade root 2 My (Nm)': 8_000_000.0,
                    'Blade root 3 My (Nm)': 8_000_000.0,
                },
                np.zeros(2),
                np.zeros(2),
                wind_t=12.0,
            )
            high_value, high = reward.compute(
                state,
                state,
                {
                    'Electrical power': 0.0,
                    'Blade root 1 My (Nm)': 12_000_000.0,
                    'Blade root 2 My (Nm)': 12_000_000.0,
                    'Blade root 3 My (Nm)': 12_000_000.0,
                },
                np.zeros(2),
                np.zeros(2),
                wind_t=12.0,
            )

        self.assertEqual(low['reward_blade_root_continuous_term'], 0.0)
        self.assertEqual(low_value, 0.0)
        self.assertAlmostEqual(high['reward_blade_root_continuous_term'], 0.2)
        self.assertAlmostEqual(high_value, -0.2)

    def test_command_power_uses_generator_speed_in_rad_per_second(self) -> None:
        reward = RewardFunction({'rated_power': 15_000_000.0})
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[STATE_VARIABLES.index('I_GeneratorSpeed')] = 2.5

        command_power = reward._estimate_command_power(
            np.asarray([4_000_000.0, 0.0], dtype=np.float32),
            state,
        )

        self.assertAlmostEqual(command_power, 10_000_000.0)


if __name__ == '__main__':
    unittest.main()