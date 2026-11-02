from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from RL.utils.constants import DATA_SCHEMA_VERSION, STATE_DIM
from RL.utils.reward import RewardFunction
from RL.utils.safe_residual_control import SafeResidualController


class TestRelativePidEnergyReward(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _reference_fixture(self, root: Path) -> tuple[Path, Path]:
        reference_path = root / 'data_windspeed8.csv'
        with reference_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['Time', 'Electrical power'])
            writer.writeheader()
            writer.writerows([
                {'Time': '0.00', 'Electrical power': '10000000'},
                {'Time': '0.02', 'Electrical power': '12000000'},
                {'Time': '0.04', 'Electrical power': '14000000'},
                {'Time': '0.06', 'Electrical power': '16000000'},
            ])
        manifest_path = root / 'paired_pid_power_reference.json'
        manifest_path.write_text(json.dumps({
            'schema_version': 'paired_pid_power_reference_v1',
            'metadata': {
                'data_schema_version': DATA_SCHEMA_VERSION,
                'source_role': 'paired_pid_outcome_reference',
                'reference_signal': 'Electrical power',
                'pairing_contract': {
                    'case_identity': 'same_bladed_case',
                    'turbulence_seed_identity': 'same_seed',
                    'time_window_identity': 'same_simulation_time',
                    'time_alignment': 'exact_timestamp_no_interpolation',
                },
                'input_manifest_sha256': 'b' * 64,
            },
            'references': [{
                'condition_id': 8,
                'path': str(reference_path),
                'sha256': self._sha256(reference_path),
                'rows': 4,
                'start_time_s': 0.0,
                'end_time_s': 0.06,
                'nominal_step_s': 0.02,
            }],
        }, indent=2), encoding='utf-8')
        return manifest_path, reference_path

    @staticmethod
    def _config(manifest_path: Path) -> dict[str, object]:
        return {
            'rated_power': 15_000_000.0,
            'bladed_step_duration_ms': 20.0,
            'reward': {
                'power_reward_mode': 'relative_pid_energy',
                'energy_reference_j': 20_000.0,
                'relative_energy_clip': 2.0,
                'pid_power_reference_manifest_path': str(manifest_path),
                'power_weight': 1.0,
                'blade_root_weight': 0.0,
                'acc_weight': 0.0,
                'shaft_weight': 0.0,
                'pitch_rate_weight': 0.0,
                'torque_rate_weight': 0.0,
                'command_excess_weight': 0.0,
                'power_tracking_weight': 0.0,
                'power_floor_weight': 0.0,
                'blade_root_continuous_weight': 0.0,
            },
        }

    @staticmethod
    def _compute(
        reward_function: RewardFunction,
        *,
        power_w: float,
        reference_w: float | None,
        time_t: float = 0.0,
        time_next: float = 0.02,
    ) -> tuple[float, dict[str, float]]:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state_next = state.copy()
        state_next[reward_function.real_power_index] = power_w
        return reward_function.compute(
            state_t=state,
            state_next=state_next,
            reward_metrics={'Electrical power': power_w},
            action_t=np.zeros(2, dtype=np.float32),
            prev_action_t=np.zeros(2, dtype=np.float32),
            wind_t=8.0,
            power_t=power_w,
            sim_time_t=time_t,
            sim_time_next=time_next,
            power_reference_raw=reference_w,
        )

    def test_exact_pid_reference_is_zero_and_power_difference_has_both_signs(self) -> None:
        with tempfile.TemporaryDirectory(prefix='relative_pid_reward_') as tmpdir:
            manifest_path, _ = self._reference_fixture(Path(tmpdir))
            reward_function = RewardFunction(self._config(manifest_path))
            reference = reward_function.reference_power(8, 0.02)

            zero, zero_breakdown = self._compute(
                reward_function,
                power_w=12_000_000.0,
                reference_w=reference,
            )
            positive, positive_breakdown = self._compute(
                reward_function,
                power_w=13_000_000.0,
                reference_w=reference,
            )
            negative, negative_breakdown = self._compute(
                reward_function,
                power_w=11_000_000.0,
                reference_w=reference,
            )

            self.assertEqual(zero, 0.0)
            self.assertEqual(zero_breakdown['reward_relative_energy_increment_j'], 0.0)
            self.assertEqual(positive, 1.0)
            self.assertEqual(positive_breakdown['reward_relative_power_raw'], 1_000_000.0)
            self.assertEqual(negative, -1.0)
            self.assertEqual(negative_breakdown['reward_relative_power_raw'], -1_000_000.0)

    def test_reference_requires_exact_condition_time_and_matching_file_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix='relative_pid_reward_') as tmpdir:
            manifest_path, reference_path = self._reference_fixture(Path(tmpdir))
            reward_function = RewardFunction(self._config(manifest_path))

            self.assertEqual(reward_function.reference_power(8, 0.04), 14_000_000.0)
            with self.assertRaisesRegex(ValueError, 'condition 9'):
                reward_function.reference_power(9, 0.02)
            with self.assertRaisesRegex(ValueError, 'exact timestamp'):
                reward_function.reference_power(8, 0.03)

            reference_path.write_text(
                reference_path.read_text(encoding='utf-8').replace('12000000', '12000001'),
                encoding='utf-8',
            )
            second_reward_function = RewardFunction(self._config(manifest_path))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                second_reward_function.reference_power(8, 0.02)

    def test_missing_reference_or_nonpositive_time_delta_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory(prefix='relative_pid_reward_') as tmpdir:
            manifest_path, _ = self._reference_fixture(Path(tmpdir))
            reward_function = RewardFunction(self._config(manifest_path))

            with self.assertRaisesRegex(ValueError, 'paired PID power reference'):
                self._compute(
                    reward_function,
                    power_w=12_000_000.0,
                    reference_w=None,
                )
            with self.assertRaisesRegex(ValueError, 'strictly increasing simulation time'):
                self._compute(
                    reward_function,
                    power_w=12_000_000.0,
                    reference_w=12_000_000.0,
                    time_t=0.02,
                    time_next=0.02,
                )

    def test_clipping_and_cumulative_relative_energy_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory(prefix='relative_pid_reward_') as tmpdir:
            manifest_path, _ = self._reference_fixture(Path(tmpdir))
            reward_function = RewardFunction(self._config(manifest_path))

            clipped, clipped_breakdown = self._compute(
                reward_function,
                power_w=20_000_000.0,
                reference_w=12_000_000.0,
            )
            self.assertEqual(clipped, 2.0)
            self.assertEqual(clipped_breakdown['reward_power_term'], 2.0)

            differences_w = [500_000.0, -250_000.0, 125_000.0]
            terms = []
            relative_energy_j = []
            for difference_w in differences_w:
                value, breakdown = self._compute(
                    reward_function,
                    power_w=12_000_000.0 + difference_w,
                    reference_w=12_000_000.0,
                )
                terms.append(value)
                relative_energy_j.append(breakdown['reward_relative_energy_increment_j'])

            self.assertAlmostEqual(
                sum(terms) * reward_function.energy_reference_j,
                sum(relative_energy_j),
            )
            self.assertAlmostEqual(sum(terms), sum(differences_w) * 0.02 / 20_000.0)

    def test_manifest_content_hash_is_part_of_online_control_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix='relative_pid_reward_') as tmpdir:
            manifest_path, _ = self._reference_fixture(Path(tmpdir))
            env_config = {
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                **self._config(manifest_path),
            }
            control_config = {
                'baseline_action': [18_000_000.0, 0.05],
                'residual_limit': [2_500_000.0, 0.07],
                'safe_action_low': [0.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.43],
            }
            original = SafeResidualController(env_config, control_config)
            document = json.loads(manifest_path.read_text(encoding='utf-8'))
            document['metadata']['input_manifest_sha256'] = 'c' * 64
            manifest_path.write_text(json.dumps(document), encoding='utf-8')
            modified = SafeResidualController(env_config, control_config)

            self.assertNotEqual(original.control_fingerprint, modified.control_fingerprint)
            self.assertNotEqual(
                original.online_training_fingerprint,
                modified.online_training_fingerprint,
            )
            self.assertEqual(original.actor_control_fingerprint, modified.actor_control_fingerprint)


if __name__ == '__main__':
    unittest.main()