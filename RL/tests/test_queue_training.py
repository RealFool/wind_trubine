from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from RL.agents import TD3Agent
from RL.envs.wind_turbine_env import StepResult
from RL.replay import ConditionStratifiedReplayBuffer
from RL.train_rl import (
    _batch_source_metrics_for_condition,
    _finalize_queue_condition_step_stats,
    _queue_context_from_info,
    _queue_condition_stats,
    _queue_rejection_metric_key,
    _record_queue_condition_step_stats,
    _should_store_transition,
    run_queue_training,
)
from RL.utils.constants import OBSERVATION_VARIABLES
from RL.utils.logger import ExperimentLogger
from RL.utils.safe_residual_control import SafeResidualController


class _QueueEnvironment:
    queue_lifecycle_enabled = True
    rollout_segment_steps = 2
    obs_dim = 3
    action_dim = 2

    def __init__(self) -> None:
        self._step_index = 0
        self._obs = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        self._last_sent_action = np.asarray([20_400_000.0, 0.049461], dtype=np.float32)

    @property
    def last_sent_action(self) -> np.ndarray:
        return self._last_sent_action.copy()

    @staticmethod
    def _info(
        condition_id: int,
        condition_instance_id: int,
        *,
        boundary: bool,
        next_condition_id: int | None = None,
        next_instance_id: int | None = None,
        eligible: bool = True,
    ) -> dict[str, float]:
        next_condition_id = condition_id if next_condition_id is None else next_condition_id
        next_instance_id = condition_instance_id if next_instance_id is None else next_instance_id
        return {
            'power': 10_000_000.0,
            'condition_boundary': float(boundary),
            'condition_train_active': 1.0,
            'transition_store_eligible': float(eligible),
            'udp_condition_id': float(condition_id),
            'udp_condition_instance_id': float(condition_instance_id),
            'udp_phase_elapsed_s': 0.02,
            'udp_next_condition_id': float(next_condition_id),
            'udp_next_condition_instance_id': float(next_instance_id),
            'env_invalid_prediction': 0.0,
            'env_invalid_reward': 0.0,
            'control_fallback_active': 0.0,
            'control_unsafe_stop': 0.0,
            'udp_inter_frame_timeout': 0.0,
        }

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        return self._obs.copy(), self._info(3, 101, boundary=False)

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        events = [
            (3, 101, False, 3, 101),
            (3, 101, True, 4, 102),
            (4, 102, False, 4, 102),
            (4, 102, True, 4, 102),
        ]
        condition_id, instance_id, boundary, next_condition_id, next_instance_id = events[self._step_index]
        self._step_index += 1
        return StepResult(
            obs=self._obs.copy(),
            reward=1.0,
            done=boundary,
            info=self._info(
                condition_id,
                instance_id,
                boundary=boundary,
                next_condition_id=next_condition_id,
                next_instance_id=next_instance_id,
                eligible=not boundary,
            ),
        )

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        return self._obs.copy(), self._info(4, 102, boundary=False)


class _PredictorShadowQueueEnvironment:
    queue_lifecycle_enabled = True
    rollout_segment_steps = 128
    obs_dim = 3
    action_dim = 2
    include_operating_state_observation = True

    class _ActionScaler:
        low = np.asarray([13_000_000.0, 0.0], dtype=np.float64)
        high = np.asarray([21_000_000.0, 0.33], dtype=np.float64)

    action_scaler = _ActionScaler()

    class _RewardReference:
        @staticmethod
        def pid_blade_root_peak_reference(condition_id: int) -> np.ndarray:
            del condition_id
            return np.asarray([50_000_000.0] * 3, dtype=np.float64)

    reward_fn = _RewardReference()

    def __init__(self, condition_steps: int = 80) -> None:
        self.condition_steps = condition_steps
        self._step_index = 0
        self._obs = np.zeros(3, dtype=np.float32)
        self._last_sent_action = np.asarray([20_000_000.0, 0.05], dtype=np.float32)

    @property
    def last_sent_action(self) -> np.ndarray:
        return self._last_sent_action.copy()

    @property
    def current_payload(self) -> dict[str, float]:
        timestamp = self._step_index * 0.02
        return {
            'Time': timestamp,
            'I_RotorSpeed': 0.8,
            'I_GeneratorSpeed': 0.8,
            'I_RealPower': 15_000_000.0,
            'I_ElectricalTorque': 20_000_000.0,
            'I_PitchPosition1': 0.05,
            'I_NacelleForeAftAcceleration': 0.0,
            'I_NacelleSideSideAcceleration': 0.0,
            'I_AverageWindSpeed': 12.0,
            'I_RotorAzimuth': timestamp,
            'Blade root 1 My (Nm)': 45_000_000.0,
            'Blade root 2 My (Nm)': 44_000_000.0,
            'Blade root 3 My (Nm)': 43_000_000.0,
        }

    def _info(self, *, boundary: bool, phase_elapsed_s: float) -> dict[str, float]:
        return {
            'power': 15_000_000.0,
            'condition_boundary': float(boundary),
            'condition_train_active': 1.0,
            'transition_store_eligible': float(not boundary),
            'udp_condition_id': 3.0,
            'udp_condition_instance_id': 101.0,
            'udp_phase_elapsed_s': phase_elapsed_s,
            'udp_next_condition_id': 3.0,
            'udp_next_condition_instance_id': 102.0 if boundary else 101.0,
            'env_invalid_prediction': 0.0,
            'env_invalid_reward': 0.0,
            'control_fallback_active': 0.0,
            'control_unsafe_stop': 0.0,
            'udp_inter_frame_timeout': 0.0,
            'action_torque': float(self._last_sent_action[0]),
            'action_pitch': float(self._last_sent_action[1]),
            'udp_sim_time': max(0.0, phase_elapsed_s - 0.02),
            'udp_next_sim_time': phase_elapsed_s,
            'power_t_metric_raw': 15_000_000.0,
            'power_metric_raw': 15_000_000.0,
            'reward_power_t_raw': 15_000_000.0,
            'reward_power_raw': 15_000_000.0,
        }

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        return self._obs.copy(), self._info(boundary=False, phase_elapsed_s=0.0)

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        self._step_index += 1
        boundary = self._step_index >= self.condition_steps
        self._last_sent_action = np.asarray([20_000_000.0, 0.05], dtype=np.float32)
        elapsed = self._step_index * 0.02
        return StepResult(
            obs=self._obs.copy(),
            reward=1.0,
            done=boundary,
            info=self._info(boundary=boundary, phase_elapsed_s=elapsed),
        )

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        return self._obs.copy(), self._info(boundary=False, phase_elapsed_s=0.0)


class _NegativePowerQueueEnvironment(_QueueEnvironment):
    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        result = super().step(action, on_action_sent=on_action_sent)
        result.info['power'] = -1.0
        return result


class _DelayedTimeoutQueueEnvironment(_QueueEnvironment):
    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        result = super().step(action, on_action_sent=on_action_sent)
        if self._step_index == 3:
            result.info['udp_inter_frame_timeout'] = 1.0
        return result


class _SkippedConditionQueueEnvironment(_QueueEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self._pending_condition = (3, 101)

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        self._pending_condition = (3, 101)
        info = self._info(3, 101, boundary=False)
        info['udp_phase_elapsed_s'] = 0.0
        return self._obs.copy(), info

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        events = [
            (3, 101, False, 3, 101, False),
            (3, 101, False, 3, 101, True),
            (4, 102, False, 4, 102, False),
            (4, 102, True, 4, 102, False),
        ]
        condition_id, instance_id, boundary, next_condition_id, next_instance_id, timeout = events[self._step_index]
        self._step_index += 1
        self._pending_condition = (next_condition_id, next_instance_id)
        info = self._info(
            condition_id,
            instance_id,
            boundary=boundary,
            next_condition_id=next_condition_id,
            next_instance_id=next_instance_id,
            eligible=not boundary and not timeout,
        )
        if timeout:
            info.pop('udp_phase_elapsed_s')
            info['udp_inter_frame_timeout'] = 1.0
        return StepResult(obs=self._obs.copy(), reward=1.0, done=boundary or timeout, info=info)

    def recover_after_train_timeout(self) -> tuple[np.ndarray, dict[str, float]]:
        self._pending_condition = (4, 102)
        info = self._info(4, 102, boundary=False)
        info['udp_phase_elapsed_s'] = 0.0
        info.update(
            {
                'udp_timeout_recovery_active': 1.0,
                'udp_timeout_recovery_wait_ms': 42.0,
                'udp_timeout_recovery_condition_changed': 1.0,
                'udp_timeout_recovery_from_queue_epoch': 1.0,
                'udp_timeout_recovery_from_condition_instance_id': 101.0,
            }
        )
        return self._obs.copy(), info

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        condition_id, instance_id = self._pending_condition
        return self._obs.copy(), self._info(condition_id, instance_id, boundary=False)


class _SkippedCriticUpdateQueueEnvironment(_SkippedConditionQueueEnvironment):
    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        events = [
            (3, 101, False, 3, 101, False),
            (3, 101, False, 3, 101, True),
            (4, 102, False, 4, 102, False),
            (4, 102, True, 3, 103, False),
            (3, 103, False, 3, 103, False),
            (3, 103, True, 4, 104, False),
            (4, 104, False, 4, 104, False),
            (4, 104, True, 4, 104, False),
        ]
        condition_id, instance_id, boundary, next_condition_id, next_instance_id, timeout = events[self._step_index]
        self._step_index += 1
        self._pending_condition = (next_condition_id, next_instance_id)
        info = self._info(
            condition_id,
            instance_id,
            boundary=boundary,
            next_condition_id=next_condition_id,
            next_instance_id=next_instance_id,
            eligible=not boundary and not timeout,
        )
        if timeout:
            info.pop('udp_phase_elapsed_s')
            info['udp_inter_frame_timeout'] = 1.0
        return StepResult(obs=self._obs.copy(), reward=1.0, done=boundary or timeout, info=info)


class _RetriedConditionQueueEnvironment(_SkippedConditionQueueEnvironment):
    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        events = [
            (3, 101, False, 3, 101, False),
            (3, 101, False, 3, 101, True),
            (3, 102, False, 3, 102, False),
            (3, 102, True, 4, 103, False),
            (4, 103, False, 4, 103, False),
            (4, 103, True, 4, 103, False),
        ]
        condition_id, instance_id, boundary, next_condition_id, next_instance_id, timeout = events[self._step_index]
        self._step_index += 1
        self._pending_condition = (next_condition_id, next_instance_id)
        info = self._info(
            condition_id,
            instance_id,
            boundary=boundary,
            next_condition_id=next_condition_id,
            next_instance_id=next_instance_id,
            eligible=not boundary and not timeout,
        )
        if timeout:
            info.pop('udp_phase_elapsed_s')
            info['udp_inter_frame_timeout'] = 1.0
        return StepResult(obs=self._obs.copy(), reward=1.0, done=boundary or timeout, info=info)

    def recover_after_train_timeout(self) -> tuple[np.ndarray, dict[str, float]]:
        self._pending_condition = (3, 102)
        info = self._info(3, 102, boundary=False)
        info['udp_phase_elapsed_s'] = 0.0
        info.update(
            {
                'udp_timeout_recovery_active': 1.0,
                'udp_timeout_recovery_wait_ms': 42.0,
                'udp_timeout_recovery_condition_changed': 1.0,
                'udp_timeout_recovery_from_queue_epoch': 1.0,
                'udp_timeout_recovery_from_condition_instance_id': 101.0,
            }
        )
        return self._obs.copy(), info


class _VerifiedPhysicalSkipQueueEnvironment:
    queue_lifecycle_enabled = True
    rollout_segment_steps = 2
    obs_dim = len(OBSERVATION_VARIABLES)
    action_dim = 2
    udp_protocol_version = 'bladed_udp_19_si_my_queue_v2'

    def __init__(self, physical_wind_prefix: tuple[float, float, float] = (4.0, 4.1, 4.2)) -> None:
        self._step_index = 0
        self._active_condition_id = 3
        self._active_condition_instance_id = 101
        self._last_sent_action = np.asarray([20_400_000.0, 0.049461], dtype=np.float32)
        self._physical_wind_prefix = physical_wind_prefix
        self.probe_calls = 0
        self.advance_calls = 0
        self.step_metadata: list[tuple[int, int]] = []

    @property
    def last_sent_action(self) -> np.ndarray:
        return self._last_sent_action.copy()

    @staticmethod
    def _obs(wind_speed: float) -> np.ndarray:
        observation = np.zeros(len(OBSERVATION_VARIABLES), dtype=np.float32)
        observation[OBSERVATION_VARIABLES.index('I_AverageWindSpeed')] = wind_speed
        return observation

    @staticmethod
    def _info(
        condition_id: int,
        condition_instance_id: int,
        *,
        phase_elapsed_s: float,
        boundary: bool,
        timeout: bool = False,
    ) -> dict[str, float]:
        return {
            'power': 10_000_000.0,
            'condition_boundary': float(boundary),
            'condition_train_active': 1.0,
            'transition_store_eligible': float(not boundary and not timeout),
            'udp_condition_id': float(condition_id),
            'udp_condition_instance_id': float(condition_instance_id),
            'udp_phase_elapsed_s': phase_elapsed_s,
            'udp_next_condition_id': float(condition_id),
            'udp_next_condition_instance_id': float(condition_instance_id),
            'env_invalid_prediction': 0.0,
            'env_invalid_reward': 0.0,
            'control_fallback_active': 0.0,
            'control_unsafe_stop': 0.0,
            'udp_inter_frame_timeout': float(timeout),
        }

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        self._active_condition_id = 3
        self._active_condition_instance_id = 101
        return self._obs(3.0), self._info(3, 101, phase_elapsed_s=0.0, boundary=False)

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        self.step_metadata.append((self._active_condition_id, self._active_condition_instance_id))
        if self._step_index == 0:
            self._step_index += 1
            return StepResult(
                obs=self._obs(3.0),
                reward=-10.0,
                done=True,
                info=self._info(3, 101, phase_elapsed_s=0.0, boundary=False, timeout=True),
            )
        if self._step_index == 1:
            self._step_index += 1
            return StepResult(
                obs=self._obs(4.3),
                reward=1.0,
                done=True,
                info=self._info(4, 103, phase_elapsed_s=0.02, boundary=True),
            )
        raise AssertionError('verified physical skip test sent an unexpected extra action')

    def recover_after_train_timeout(self) -> tuple[np.ndarray, dict[str, float]]:
        self._active_condition_id = 3
        self._active_condition_instance_id = 102
        info = self._info(3, 102, phase_elapsed_s=0.0, boundary=False)
        info.update(
            {
                'udp_timeout_recovery_active': 1.0,
                'udp_timeout_recovery_wait_ms': 42.0,
                'udp_timeout_recovery_condition_changed': 1.0,
                'udp_timeout_recovery_from_queue_epoch': 101.0,
                'udp_timeout_recovery_from_condition_instance_id': 101.0,
            }
        )
        return self._obs(self._physical_wind_prefix[0]), info

    def probe_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        self.probe_calls += 1
        if self.probe_calls > 2:
            raise AssertionError('physical identity probe requested too many frames')
        return self._obs(self._physical_wind_prefix[self.probe_calls]), self._info(
            3,
            102,
            phase_elapsed_s=0.02 * self.probe_calls,
            boundary=False,
        )

    def advance_pending_incomplete_slot(self) -> tuple[np.ndarray, dict[str, float]]:
        if self.probe_calls != 2:
            raise AssertionError('physical skip remap must wait for the complete wind prefix')
        self.advance_calls += 1
        self._active_condition_id = 4
        self._active_condition_instance_id = 103
        return self._obs(4.3), self._info(4, 103, phase_elapsed_s=0.0, boundary=False)


class _DirectSkippedConditionQueueEnvironment(_QueueEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self._pending_condition = (3, 101)

    @classmethod
    def _info_at_elapsed(
        cls,
        condition_id: int,
        condition_instance_id: int,
        *,
        phase_elapsed_s: float,
        boundary: bool,
        next_condition_id: int | None = None,
        next_instance_id: int | None = None,
        eligible: bool = True,
    ) -> dict[str, float]:
        info = cls._info(
            condition_id,
            condition_instance_id,
            boundary=boundary,
            next_condition_id=next_condition_id,
            next_instance_id=next_instance_id,
            eligible=eligible,
        )
        info['udp_phase_elapsed_s'] = phase_elapsed_s
        return info

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        self._pending_condition = (3, 101)
        return self._obs.copy(), self._info_at_elapsed(3, 101, phase_elapsed_s=0.0, boundary=False)

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        events = [
            (3, 101, 0.02, False, 3, 101),
            (3, 101, 0.02, True, 4, 102),
            (4, 102, 0.02, False, 4, 102),
            (4, 102, 0.04, True, 4, 102),
        ]
        condition_id, instance_id, elapsed_s, boundary, next_condition_id, next_instance_id = events[self._step_index]
        self._step_index += 1
        self._pending_condition = (next_condition_id, next_instance_id)
        return StepResult(
            obs=self._obs.copy(),
            reward=1.0,
            done=boundary,
            info=self._info_at_elapsed(
                condition_id,
                instance_id,
                phase_elapsed_s=elapsed_s,
                boundary=boundary,
                next_condition_id=next_condition_id,
                next_instance_id=next_instance_id,
                eligible=not boundary,
            ),
        )

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        condition_id, instance_id = self._pending_condition
        return self._obs.copy(), self._info_at_elapsed(
            condition_id,
            instance_id,
            phase_elapsed_s=0.0,
            boundary=False,
        )


class _CriticUpdateQueueEnvironment(_QueueEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self._pending_condition = (3, 101)

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        self._pending_condition = (3, 101)
        return self._obs.copy(), self._info(3, 101, boundary=False)

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(123.0)
        events = [
            (3, 101, False, 3, 101),
            (3, 101, True, 4, 102),
            (4, 102, False, 4, 102),
            (4, 102, True, 3, 103),
            (3, 103, False, 3, 103),
            (3, 103, True, 4, 104),
            (4, 104, False, 4, 104),
            (4, 104, True, 4, 104),
        ]
        condition_id, instance_id, boundary, next_condition_id, next_instance_id = events[self._step_index]
        self._step_index += 1
        self._pending_condition = (next_condition_id, next_instance_id)
        return StepResult(
            obs=self._obs.copy(),
            reward=1.0,
            done=boundary,
            info=self._info(
                condition_id,
                instance_id,
                boundary=boundary,
                next_condition_id=next_condition_id,
                next_instance_id=next_instance_id,
                eligible=not boundary,
            ),
        )

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        condition_id, instance_id = self._pending_condition
        return self._obs.copy(), self._info(condition_id, instance_id, boundary=False)


class _ConditionCheckpointQueueEnvironment(_QueueEnvironment):
    def __init__(self, terminal_phase_elapsed_s: float = 4.0) -> None:
        super().__init__()
        self._pending_condition = (3, 101)
        self._terminal_phase_elapsed_s = terminal_phase_elapsed_s

    @classmethod
    def _info_at_elapsed(
        cls,
        condition_id: int,
        condition_instance_id: int,
        *,
        phase_elapsed_s: float,
        boundary: bool,
        next_condition_id: int | None = None,
        next_instance_id: int | None = None,
        eligible: bool = True,
    ) -> dict[str, float]:
        info = cls._info(
            condition_id,
            condition_instance_id,
            boundary=boundary,
            next_condition_id=next_condition_id,
            next_instance_id=next_instance_id,
            eligible=eligible,
        )
        info['udp_phase_elapsed_s'] = phase_elapsed_s
        return info

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        self._step_index = 0
        self._pending_condition = (3, 101)
        return self._obs.copy(), self._info_at_elapsed(3, 101, phase_elapsed_s=0.0, boundary=False)

    def step(self, action: np.ndarray, on_action_sent=None) -> StepResult:
        del action
        if on_action_sent is not None:
            on_action_sent(0.0)
        events = [
            (3, 101, 1.0, False, 3, 101),
            (3, 101, 2.0, False, 3, 101),
            (3, 101, 3.0, False, 3, 101),
            (3, 101, self._terminal_phase_elapsed_s, True, 4, 102),
            (4, 102, 1.0, False, 4, 102),
            (4, 102, 2.0, False, 4, 102),
            (4, 102, 3.0, False, 4, 102),
            (4, 102, self._terminal_phase_elapsed_s, True, 4, 102),
        ]
        condition_id, instance_id, phase_elapsed_s, boundary, next_condition_id, next_instance_id = events[self._step_index]
        self._step_index += 1
        self._pending_condition = (next_condition_id, next_instance_id)
        return StepResult(
            obs=self._obs.copy(),
            reward=1.0,
            done=boundary,
            info=self._info_at_elapsed(
                condition_id,
                instance_id,
                phase_elapsed_s=phase_elapsed_s,
                boundary=boundary,
                next_condition_id=next_condition_id,
                next_instance_id=next_instance_id,
                eligible=not boundary,
            ),
        )

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        condition_id, instance_id = self._pending_condition
        return self._obs.copy(), self._info_at_elapsed(
            condition_id,
            instance_id,
            phase_elapsed_s=0.0,
            boundary=False,
        )


class _FixedResidualAgent:
    def __init__(self, action: list[float]) -> None:
        self._action = torch.tensor(action, dtype=torch.float32)
        self.noise_stds: list[float] = []
        self.deterministic_calls: list[bool] = []

    def select_action(self, obs: torch.Tensor, *, noise_std: float, deterministic: bool) -> torch.Tensor:
        self.noise_stds.append(noise_std)
        self.deterministic_calls.append(deterministic)
        return self._action.to(obs.device).unsqueeze(0)


class _FixedBaselineActor(torch.nn.Module):
    def __init__(self, action: list[float]) -> None:
        super().__init__()
        self.register_buffer('_action', torch.tensor(action, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._action.to(obs.device).expand(obs.shape[0], -1)


class _FixedBaselineAgent:
    def __init__(self, action: list[float]) -> None:
        self.actor = _FixedBaselineActor(action)
        self.select_action_calls = 0

    def select_action(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        self.select_action_calls += 1
        raise AssertionError('force_bc_baseline must use the frozen BC actor directly')


class _NoActionAgent:
    def __init__(self) -> None:
        self.select_action_calls = 0

    def select_action(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        self.select_action_calls += 1
        raise AssertionError('force_pid_history must not select an actor action')


class _PidHistoryQueueEnvironment(_QueueEnvironment):
    obs_dim = len(OBSERVATION_VARIABLES)

    def __init__(self) -> None:
        super().__init__()
        self._obs = np.zeros(self.obs_dim, dtype=np.float32)


def _build_verified_physical_skip_fixture(
    root: Path,
) -> tuple[dict, SafeResidualController, TD3Agent, ConditionStratifiedReplayBuffer]:
    manifest_path = root / 'manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'protocol_version': 'bladed_udp_19_si_my_queue_v2',
                'execution_mode': 'independent_case_reload',
                'total_condition_rounds': 1,
                'base_condition_order': [3, 4],
                'rotation_stride': 0,
                'queue_epoch_start': 1,
                'condition_instance_start': 1,
                'post_load_discard_sim_s': 0.0,
                'train_sim_s': 0.02,
            }
        ),
        encoding='utf-8',
    )
    for condition_id, winds in ((3, [3.0, 3.1, 3.2]), (4, [4.0, 4.1, 4.2])):
        trace_path = root / f'data_windspeed{condition_id}.csv'
        with trace_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['I_AverageWindSpeed'])
            writer.writeheader()
            writer.writerows({'I_AverageWindSpeed': wind_speed} for wind_speed in winds)

    config = {
        'data_schema_version': 'bladed_udp_19_si_my_v1',
        'algorithm': 'td3',
        'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
        'env': {},
        'train': {
            'online_rl_enabled': False,
            'online_eval_enabled': False,
            'eval_every': 0,
            'batch_size': 2,
            'updates_per_step': 1,
            'update_every_steps': 1,
            'step_log_every': 10,
            'warmup_exploration_noise': 0.0,
            'exploration_noise': 0.0,
            'condition_replay': {
                'sampling': 'condition_stratified',
                'capacity_per_condition': 4,
                'min_valid_transitions_per_condition': 1,
                'min_per_condition_in_batch': 1,
            },
        },
        'queue': {
            'manifest_path': str(manifest_path),
            'execution_mode': 'independent_case_reload',
            'condition_ids': [3, 4],
            'total_condition_rounds': 1,
            'train_sim_s': 0.02,
            'train_sim_tolerance_s': 1.0e-6,
            'critic_start_after_rounds': 1,
            'actor_start_after_rounds': 2,
            'collect_exploration_noise': 0.0,
            'skip_failed_condition_on_train_timeout': True,
            'require_zero_skipped_conditions_for_updates': True,
            'require_zero_retried_condition_attempts_for_updates': True,
            'abort_on_retried_condition_attempt': True,
            'reconcile_verified_physical_skip': True,
            'case_identity_validation': {
                'enabled': True,
                'source_glob': str(root / 'data_windspeed*.csv'),
                'wind_sample_count': 3,
                'wind_absolute_tolerance': 0.01,
            },
        },
        'offline_eval': {'enabled': False},
    }
    controller = SafeResidualController(
        {
            'action_low': [13_000_000.0, 0.0],
            'action_high': [21_000_000.0, 0.33],
            'action_limiter_owner': 'controller',
            'action_delta_low': [-15_000.0, -0.0003],
            'action_delta_high': [15_000.0, 0.0003],
        },
        {
            'baseline_action': [20_000_000.0, 0.05],
            'residual_limit': [1_000_000.0, 0.05],
            'safe_action_low': [13_000_000.0, 0.0],
            'safe_action_high': [21_000_000.0, 0.33],
            'ood_threshold': 100.0,
        },
    )
    controller.set_observation_distribution(
        np.zeros(len(OBSERVATION_VARIABLES), dtype=np.float32),
        np.ones(len(OBSERVATION_VARIABLES), dtype=np.float32),
    )
    agent = TD3Agent(
        obs_dim=len(OBSERVATION_VARIABLES),
        action_dim=2,
        config=config['agent'],
        device=torch.device('cpu'),
    )
    agent.set_actor_obs_normalizer(torch.zeros(len(OBSERVATION_VARIABLES)), torch.ones(len(OBSERVATION_VARIABLES)))
    agent.set_critic_obs_normalizer(torch.zeros(len(OBSERVATION_VARIABLES)), torch.ones(len(OBSERVATION_VARIABLES)))
    replay = ConditionStratifiedReplayBuffer(
        len(OBSERVATION_VARIABLES),
        2,
        [3, 4],
        capacity_per_condition=4,
        min_per_condition_in_batch=1,
    )
    return config, controller, agent, replay


class TestQueueTraining(unittest.TestCase):
    def test_condition_summary_streams_trapezoid_energy_and_per_blade_peak_timing(self) -> None:
        class Reference:
            def reference_power(self, condition_id: int, sim_time_s: float) -> float:
                self.assert_condition = condition_id
                return 100.0 + 10.0 * sim_time_s

        environment = type('Environment', (), {'reward_fn': Reference()})()
        stats = _queue_condition_stats()
        stats['queue_condition_eligible_transition_count'] = 2.0

        first = {
            'control_pre_send_execution_mode': 1.0,
            'control_low_speed_stage': 0.0,
            'control_actor_update_gate_open': 1.0,
            'control_action_projection_applied': 0.0,
            'control_action_rate_limited': 1.0,
            'control_ood_score': 0.5,
            'udp_sim_time': 0.0,
            'udp_next_sim_time': 1.0,
            'reward_power_t_raw': 100.0,
            'reward_power_raw': 200.0,
            'reward_pid_power_reference_raw': 110.0,
            'reward_blade_root_1_raw': 90.0,
            'reward_blade_root_2_raw': -120.0,
            'reward_blade_root_3_raw': 80.0,
            'reward_blade_root_1_peak_reference_nm': 100.0,
            'reward_blade_root_2_peak_reference_nm': 100.0,
            'reward_blade_root_3_peak_reference_nm': 100.0,
        }
        second = {
            **first,
            'control_pre_send_execution_mode': 8.0,
            'control_low_speed_stage': 4.0,
            'control_actor_update_gate_open': 0.0,
            'control_action_projection_applied': 1.0,
            'control_action_rate_limited': 0.0,
            'control_ood_score': 1.5,
            'udp_sim_time': 1.0,
            'udp_next_sim_time': 2.0,
            'reward_power_t_raw': 200.0,
            'reward_power_raw': 300.0,
            'reward_pid_power_reference_raw': 120.0,
            'reward_blade_root_1_raw': 110.0,
            'reward_blade_root_2_raw': 130.0,
            'reward_blade_root_3_raw': 85.0,
        }
        _record_queue_condition_step_stats(stats, first, environment, condition_id=3, transition_eligible=True)
        _record_queue_condition_step_stats(stats, second, environment, condition_id=3, transition_eligible=True)
        _finalize_queue_condition_step_stats(stats, condition_id=3)

        self.assertEqual(stats['queue_condition_rl_trapezoid_energy_j'], 400.0)
        self.assertEqual(stats['queue_condition_pid_trapezoid_energy_j'], 220.0)
        self.assertEqual(stats['queue_condition_rl_pdt_diagnostic_energy_j'], 500.0)
        self.assertEqual(stats['queue_condition_pid_pdt_diagnostic_energy_j'], 230.0)
        self.assertEqual(stats['queue_condition_trapezoid_inputs_complete'], 1.0)
        self.assertEqual(stats['queue_condition_blade_2_peak_abs_nm'], 130.0)
        self.assertEqual(stats['queue_condition_blade_2_peak_time_s'], 2.0)
        self.assertEqual(stats['queue_condition_blade_1_first_pid_exceed_time_s'], 2.0)
        self.assertEqual(stats['queue_condition_execution_mode_transition_count'], 1.0)
        self.assertEqual(stats['queue_condition_low_speed_stage_transition_count'], 1.0)
        self.assertEqual(stats['queue_condition_actor_optimizer_gate_transition_count'], 1.0)
        self.assertEqual(stats['queue_condition_action_projection_count'], 1.0)
        self.assertEqual(stats['queue_condition_action_rate_limited_count'], 1.0)
        self.assertEqual(stats['queue_condition_ood_score_count'], 2.0)
        self.assertEqual(stats['queue_condition_ood_score_sum'], 2.0)
        self.assertEqual(stats['queue_condition_execution_mode_actor_duration_max_s'], 1.0)
        self.assertEqual(stats['queue_condition_execution_mode_low_speed_recovery_hold_duration_max_s'], 1.0)
        self.assertEqual(stats['queue_condition_execution_mode_actor_duration_run_count'], 1.0)
        self.assertEqual(stats['queue_condition_execution_mode_low_speed_recovery_hold_duration_run_count'], 1.0)

    def test_condition_summary_marks_trapezoid_unavailable_if_pid_reference_is_missing(self) -> None:
        class Reference:
            def reference_power(self, condition_id: int, sim_time_s: float) -> None:
                return None

        environment = type('Environment', (), {'reward_fn': Reference()})()
        stats = _queue_condition_stats()
        stats['queue_condition_eligible_transition_count'] = 1.0
        step_info = {
            'control_pre_send_execution_mode': 1.0,
            'control_low_speed_stage': 0.0,
            'control_actor_update_gate_open': 1.0,
            'control_action_projection_applied': 0.0,
            'control_action_rate_limited': 0.0,
            'control_ood_score': 0.0,
            'udp_sim_time': 0.0,
            'udp_next_sim_time': 1.0,
            'reward_power_t_raw': 100.0,
            'reward_power_raw': 200.0,
            'reward_pid_power_reference_raw': 0.0,
        }
        _record_queue_condition_step_stats(stats, step_info, environment, condition_id=3, transition_eligible=True)
        _finalize_queue_condition_step_stats(stats, condition_id=3)

        self.assertEqual(stats['queue_condition_rl_trapezoid_valid_segment_count'], 1.0)
        self.assertEqual(stats['queue_condition_pid_trapezoid_valid_segment_count'], 0.0)
        self.assertEqual(stats['queue_condition_trapezoid_missing_pid_reference_count'], 1.0)
        self.assertEqual(stats['queue_condition_trapezoid_inputs_complete'], 0.0)
        self.assertEqual(stats['queue_condition_rl_pdt_diagnostic_energy_j'], 200.0)

    def test_negative_power_transition_is_filtered_before_safety_stop(self) -> None:
        step_info = {
            'condition_boundary': 0.0,
            'transition_store_eligible': 1.0,
            'control_fallback_active': 0.0,
            'control_unsafe_stop': 0.0,
            'control_negative_power_steps': 1.0,
        }
        train_config = {'replay_filter': {'enabled': True}}

        self.assertFalse(_should_store_transition(step_info, train_config))
        self.assertEqual(
            _queue_rejection_metric_key(step_info, transition_finite=True),
            'queue_condition_rejected_negative_power_count',
        )

        unsafe_step_info = {**step_info, 'control_unsafe_stop': 1.0}
        self.assertEqual(
            _queue_rejection_metric_key(unsafe_step_info, transition_finite=True),
            'queue_condition_rejected_unsafe_count',
        )
        self.assertTrue(_should_store_transition(step_info, {'replay_filter': {'enabled': False}}))

    def test_queue_context_rejects_an_unconfigured_dll_condition(self) -> None:
        info = _QueueEnvironment._info(13, 103, boundary=False)

        with self.assertRaisesRegex(
            RuntimeError,
            r'Queue condition mismatch: received condition=13 instance=103, expected one of \[11, 12\]',
        ):
            _queue_context_from_info(info, expected_condition_ids=(11, 12))

    def test_queue_runner_stores_train_transitions_and_checkpoints_completed_round(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {
                'hidden_dims': [8],
                'td3_bc_enabled': False,
                'td3_bc_weight': 1.0,
            },
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_training_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            environment = _QueueEnvironment()
            summary = run_queue_training(
                agent,
                environment,
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            self.assertTrue((Path(temporary_directory) / 'checkpoints' / 'queue_round_1.pt').is_file())
            step_metrics_path = Path(temporary_directory) / 'queue_train_step_metrics.csv'
            self.assertTrue(step_metrics_path.is_file())
            self.assertIn('control_ood_feature_index', step_metrics_path.read_text(encoding='utf-8'))
            with step_metrics_path.open('r', encoding='utf-8', newline='') as handle:
                first_step_metrics = {
                    row['name']: float(row['value'])
                    for row in csv.DictReader(handle)
                    if row['stage'] == 'queue_train_step' and row['step'] == '1'
                }
            condition_metrics_path = Path(temporary_directory) / 'queue_condition_metrics.csv'
            with condition_metrics_path.open('r', encoding='utf-8', newline='') as handle:
                condition_metrics = {
                    row['name']: float(row['value'])
                    for row in csv.DictReader(handle)
                    if row['stage'] == 'queue_condition' and row['name'].startswith('queue_condition_batch_')
                }
            initial_action = environment.last_sent_action
            self.assertLessEqual(
                abs(first_step_metrics['control_action_torque'] - initial_action[0]),
                15_000.1,
            )
            self.assertLessEqual(
                abs(first_step_metrics['control_action_pitch'] - initial_action[1]),
                0.0003001,
            )
            self.assertIn('control_pre_safety_action_torque', first_step_metrics)
            self.assertIn('control_post_safety_action_torque', first_step_metrics)
            self.assertIn('control_projected_action_torque', first_step_metrics)
            self.assertEqual(
                first_step_metrics['control_rate_limited_action_torque'],
                first_step_metrics['control_sent_action_torque'],
            )
            self.assertEqual(
                first_step_metrics['control_rate_limited_action_pitch'],
                first_step_metrics['control_sent_action_pitch'],
            )

        self.assertEqual(summary['queue_completed_rounds'], 1.0)
        self.assertEqual(summary['queue_total_stored_transitions'], 2.0)
        self.assertEqual(replay.condition_counts(), {3: 1, 4: 1})
        self.assertEqual(condition_metrics['queue_condition_batch_count'], 0.0)
        self.assertEqual(condition_metrics['queue_condition_batch_sample_count'], 0.0)
        self.assertEqual(condition_metrics['queue_condition_batch_source_actor_sample_count'], 0.0)

    def test_condition_batch_source_metrics_report_observed_per_condition_counts(self) -> None:
        stats_by_round = {
            2: {
                'condition_sample_counts': {
                    3: {'actor': 12, 'low_speed_recovery_hold': 4},
                    4: {'actor': 8, 'low_speed_recovery_hold': 0},
                },
                'condition_batch_counts': {3: 2, 4: 1},
                'condition_quota_relaxed_count': {3: 2, 4: 0},
                'condition_quota_relaxed_by_source': {
                    3: {'actor': 0, 'low_speed_recovery_hold': 2},
                    4: {'actor': 0, 'low_speed_recovery_hold': 0},
                },
            },
        }

        metrics = _batch_source_metrics_for_condition(
            stats_by_round,
            2,
            3,
            ('actor', 'low_speed_recovery_hold'),
        )

        self.assertEqual(metrics['queue_condition_batch_count'], 2.0)
        self.assertEqual(metrics['queue_condition_batch_sample_count'], 16.0)
        self.assertEqual(metrics['queue_condition_batch_source_actor_sample_count'], 12.0)
        self.assertEqual(metrics['queue_condition_batch_source_low_speed_recovery_hold_sample_count'], 4.0)
        self.assertEqual(metrics['queue_condition_batch_quota_relaxed_count'], 2.0)
        self.assertEqual(metrics['queue_condition_batch_source_low_speed_recovery_hold_quota_relaxed_count'], 2.0)

    def test_queue_training_collects_bounded_load_predictor_pending_labels(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'env': {'reward': {}},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 20,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 128,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3],
                'total_condition_rounds': 1,
                'train_sim_s': 1.6,
                'train_sim_tolerance_s': 1.0e-6,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
            },
            'load_predictor': {
                'mode': 'shadow',
                'interface_path': 'RL/data/load_predictor_interface_v3.json',
                'replay_capacity': 256,
                'shadow_audit_capacity_per_group': 16,
                'shadow_audit_max_records': 1000,
                'shadow_audit_max_record_bytes': 64 * 1024,
                'shadow_audit_max_total_bytes': 2 * 1024 * 1024,
            },
            'offline_eval': {'enabled': False},
            'logging': {
                'mode': 'compact_streaming',
                'quantile_sample_capacity': 16,
                'event_ring_capacity': 16,
                'episode_ring_capacity': 2,
                'queue_condition_ring_capacity': 4,
                'queue_condition_max_record_bytes': 256 * 1024,
                'queue_condition_summaries_max_total_bytes': 2 * 1024 * 1024,
                'queue_event_pre_frames': 2,
                'queue_event_post_frames': 2,
                'queue_event_window_capacity': 16,
                'queue_event_window_max_record_bytes': 32 * 1024,
                'queue_event_windows_max_total_bytes': 2 * 1024 * 1024,
                'debug_long_form_enabled': False,
                'debug_max_rows': 0,
                'debug_max_bytes': 0,
            },
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(3, 2, config['agent'], torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(
            3,
            2,
            [3],
            capacity_per_condition=128,
            min_per_condition_in_batch=1,
        )

        with tempfile.TemporaryDirectory(prefix='queue_load_predictor_shadow_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory, config['logging'])
            summary = run_queue_training(
                agent,
                _PredictorShadowQueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            audit_path = Path(temporary_directory) / 'load_predictor_shadow_audit.jsonl'
            audit_rows = [json.loads(line) for line in audit_path.read_text(encoding='utf-8').splitlines()]
            condition_rows = [
                json.loads(line)
                for line in (Path(temporary_directory) / 'compact_queue_condition_summaries.jsonl')
                .read_text(encoding='utf-8')
                .splitlines()
                if json.loads(line)['stage'] == 'queue_condition'
            ]

        self.assertEqual(summary['queue_load_predictor_shadow_enabled'], 1.0)
        self.assertEqual(summary['queue_load_predictor_model_loaded'], 0.0)
        self.assertGreater(summary['queue_load_predictor_shadow_audit_records_written'], 0.0)
        self.assertEqual(len(condition_rows), 1)
        self.assertEqual(condition_rows[0]['queue_condition_load_predictor_observed_frame_count'], 79.0)
        self.assertEqual(condition_rows[0]['queue_condition_load_predictor_pending_sample_count'], 30.0)
        self.assertGreater(condition_rows[0]['queue_condition_load_predictor_completed_record_count'], 0.0)
        self.assertEqual(condition_rows[0]['queue_condition_load_predictor_dropped_pending_count'], 25.0)
        self.assertTrue(all(row['prediction_available'] is False for row in audit_rows))
        self.assertTrue(all(row['calibration_eligible'] is False for row in audit_rows))
        self.assertTrue(all(row['execution_mode'] for row in audit_rows))

    def test_queue_saves_four_train_time_checkpoints_per_condition(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 8,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'train_sim_s': 4.0,
                'train_sim_tolerance_s': 0.1,
                'condition_checkpoint_count': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=8, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_condition_checkpoints_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            run_queue_training(
                agent,
                _ConditionCheckpointQueueEnvironment(terminal_phase_elapsed_s=3.95),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            checkpoint_dir = Path(temporary_directory) / 'checkpoints'
            condition_checkpoints = sorted(checkpoint_dir.glob('queue_round_*_condition_*_instance_*_checkpoint_*_of_01.pt'))
            self.assertEqual(len(condition_checkpoints), 2)
            self.assertTrue((checkpoint_dir / 'queue_round_1.pt').is_file())
            condition_four_checkpoints = [
                torch.load(path, map_location='cpu')
                for path in condition_checkpoints
                if 'condition_04_instance_102' in path.name
            ]
            self.assertEqual(
                [checkpoint['queue_checkpoint']['target_train_elapsed_s'] for checkpoint in condition_four_checkpoints],
                [4.0],
            )
            self.assertEqual(
                [checkpoint['queue_checkpoint']['actual_train_elapsed_s'] for checkpoint in condition_four_checkpoints],
                [3.95],
            )
            checkpoint = condition_four_checkpoints[0]
            self.assertEqual(checkpoint['queue_checkpoint']['checkpoint_kind'], 'condition_progress')
            self.assertEqual(checkpoint['queue_checkpoint']['checkpoint_count'], 1)
            self.assertEqual(checkpoint['queue_checkpoint']['checkpoint_index'], 1)
            self.assertAlmostEqual(checkpoint['queue_checkpoint']['target_train_elapsed_s'], 4.0)
            self.assertAlmostEqual(checkpoint['queue_checkpoint']['actual_train_elapsed_s'], 3.95)
            self.assertEqual(checkpoint['queue_checkpoint']['total_steps'], checkpoint['total_steps'])
            checkpoint_metrics = (Path(temporary_directory) / 'queue_condition_checkpoint_metrics.csv').read_text(
                encoding='utf-8'
            )
            self.assertIn('queue_checkpoint_count,1.0', checkpoint_metrics)

    def test_queue_force_baseline_zeros_residual_actions_without_updates(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': True,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'force_baseline': True,
                'force_baseline_hold_initial': True,
            },
            'offline_eval': {'enabled': True, 'run_after_training': True},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_baseline_diagnosis_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            summary = run_queue_training(
                agent,
                _QueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            step_metrics = (Path(temporary_directory) / 'queue_train_step_metrics.csv').read_text(encoding='utf-8')
            self.assertFalse((Path(temporary_directory) / 'checkpoints' / 'queue_round_1.pt').exists())
            self.assertFalse((Path(temporary_directory) / 'offline_eval_jobs.jsonl').exists())

        self.assertEqual(summary['queue_total_stored_transitions'], 0.0)
        self.assertEqual(summary['queue_update_count'], 0.0)
        self.assertEqual(agent.update_step, 0)
        self.assertIn('control_force_baseline,1.0', step_metrics)
        self.assertIn('control_baseline_hold_initial,1.0', step_metrics)
        self.assertIn('control_baseline_action_torque,20000000.0', step_metrics)
        self.assertIn('control_baseline_action_pitch,0.05', step_metrics)
        self.assertIn('control_residual_torque,0.0', step_metrics)
        self.assertIn('control_residual_pitch,0.0', step_metrics)

    def test_queue_residual_scale_diagnosis_is_deterministic_and_non_training(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': True,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.5,
                'exploration_noise': 0.5,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.5,
                'residual_scale_diagnostic': 0.25,
            },
            'offline_eval': {'enabled': True, 'run_after_training': True},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = _FixedResidualAgent([1.0, -1.0])
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_residual_scale_diagnosis_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            summary = run_queue_training(
                agent,
                _QueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            step_metrics = (Path(temporary_directory) / 'queue_train_step_metrics.csv').read_text(encoding='utf-8')
            self.assertFalse((Path(temporary_directory) / 'checkpoints' / 'queue_round_1.pt').exists())
            self.assertFalse((Path(temporary_directory) / 'offline_eval_jobs.jsonl').exists())

        self.assertEqual(summary['queue_total_stored_transitions'], 0.0)
        self.assertEqual(summary['queue_update_count'], 0.0)
        self.assertTrue(all(noise_std == 0.0 for noise_std in agent.noise_stds))
        self.assertTrue(all(agent.deterministic_calls))
        self.assertIn('control_residual_scale_diagnostic,1.0', step_metrics)
        self.assertIn('control_residual_scale,0.25', step_metrics)
        self.assertIn('control_residual_torque,0.25', step_metrics)
        self.assertIn('control_residual_pitch,-0.25', step_metrics)

    def test_queue_force_bc_baseline_uses_frozen_actor_without_training(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': True,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'force_bc_baseline': True,
            },
            'offline_eval': {'enabled': True, 'run_after_training': True},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = _FixedBaselineAgent([1.0, -1.0])
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_bc_baseline_diagnosis_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            summary = run_queue_training(
                agent,
                _QueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            step_metrics = (Path(temporary_directory) / 'queue_train_step_metrics.csv').read_text(encoding='utf-8')
            self.assertFalse((Path(temporary_directory) / 'checkpoints' / 'queue_round_1.pt').exists())
            self.assertFalse((Path(temporary_directory) / 'offline_eval_jobs.jsonl').exists())

        self.assertEqual(summary['queue_total_stored_transitions'], 0.0)
        self.assertEqual(summary['queue_update_count'], 0.0)
        self.assertEqual(agent.select_action_calls, 0)
        self.assertIn('control_force_bc_baseline,1.0', step_metrics)
        self.assertIn('control_fallback_bc_baseline,1.0', step_metrics)
        self.assertIn('control_bc_baseline_action_torque,21000000.0', step_metrics)
        self.assertIn('control_bc_baseline_action_pitch,0.0', step_metrics)
        self.assertIn('control_bc_baseline_residual_torque,1.0', step_metrics)
        self.assertIn('control_bc_baseline_residual_pitch,-1.0', step_metrics)
        self.assertIn('control_residual_torque,0.0', step_metrics)
        self.assertIn('control_residual_pitch,0.0', step_metrics)

    def test_queue_force_pid_history_uses_configured_action_offset_without_training(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': True,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'pid_history_diagnostic_max_wind_error': 0.01,
                'pid_history_diagnostic_action_offset_steps': 1,
            },
            'offline_eval': {'enabled': True, 'run_after_training': True},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(
            np.zeros(len(OBSERVATION_VARIABLES), dtype=np.float32),
            np.ones(len(OBSERVATION_VARIABLES), dtype=np.float32),
        )
        agent = _NoActionAgent()
        replay = ConditionStratifiedReplayBuffer(
            len(OBSERVATION_VARIABLES),
            2,
            [3, 4],
            capacity_per_condition=4,
            min_per_condition_in_batch=1,
        )

        with tempfile.TemporaryDirectory(prefix='queue_pid_history_diagnosis_') as temporary_directory:
            history_path = Path(temporary_directory) / 'pid_history.csv'
            fieldnames = ['O_PcsTorqueDemand', 'O_PitchPositionDemand1', *OBSERVATION_VARIABLES]
            history_rows = []
            for torque, pitch in ((21_000_000.0, 0.0), (20_000_000.0, 0.1), (19_000_000.0, 0.2)):
                history_row = {fieldname: '0.0' for fieldname in fieldnames}
                history_row['O_PcsTorqueDemand'] = str(torque)
                history_row['O_PitchPositionDemand1'] = str(pitch)
                history_rows.append(','.join(history_row[fieldname] for fieldname in fieldnames))
            history_path.write_text(
                ','.join(fieldnames) + '\n'
                + '\n'.join(history_rows) + '\n',
                encoding='utf-8',
            )
            config['queue']['pid_history_diagnostic_path'] = str(history_path)
            logger = ExperimentLogger(temporary_directory)
            environment = _PidHistoryQueueEnvironment()
            summary = run_queue_training(
                agent,
                environment,
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            step_metrics_path = Path(temporary_directory) / 'queue_train_step_metrics.csv'
            step_metrics = step_metrics_path.read_text(encoding='utf-8')
            with step_metrics_path.open('r', encoding='utf-8', newline='') as handle:
                first_step_metrics = {
                    row['name']: float(row['value'])
                    for row in csv.DictReader(handle)
                    if row['stage'] == 'queue_train_step' and row['step'] == '1'
                }
            self.assertFalse((Path(temporary_directory) / 'checkpoints' / 'queue_round_1.pt').exists())
            self.assertFalse((Path(temporary_directory) / 'offline_eval_jobs.jsonl').exists())

        self.assertEqual(summary['queue_total_stored_transitions'], 0.0)
        self.assertEqual(summary['queue_update_count'], 0.0)
        self.assertEqual(agent.select_action_calls, 0)
        self.assertAlmostEqual(first_step_metrics['control_action_torque'], 20_385_000.0, places=1)
        self.assertAlmostEqual(first_step_metrics['control_action_pitch'], 0.049761, places=6)
        self.assertIn('control_force_pid_history,1.0', step_metrics)
        self.assertIn('control_fallback_pid_history,1.0', step_metrics)
        self.assertIn('control_pid_history_index,0.0', step_metrics)
        self.assertIn('control_pid_history_action_index,1.0', step_metrics)
        self.assertIn('control_pid_history_action_offset_steps,1.0', step_metrics)
        self.assertIn('control_pid_history_action_torque,20000000.0', step_metrics)
        self.assertIn('control_pid_history_action_pitch,0.1', step_metrics)
        self.assertIn('control_pid_history_wind_error,0.0', step_metrics)
        self.assertIn('control_pid_history_observation_available,1.0', step_metrics)
        self.assertIn('control_pid_history_obs_max_abs_error,0.0', step_metrics)
        self.assertIn('control_pid_history_obs_error_I_RotorSpeed,0.0', step_metrics)

    def test_queue_runs_a_critic_update_after_the_first_completed_round(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {
                'hidden_dims': [8],
                'td3_bc_enabled': False,
                'td3_bc_weight': 1.0,
                'policy_delay': 4,
            },
            'train': {
                'online_rl_enabled': True,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 2,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'critic_exploration_noise': 0.0,
                'actor_exploration_noise': 0.0,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_critic_update_') as temporary_directory:
            logger = ExperimentLogger(
                temporary_directory,
                {
                    'mode': 'compact_streaming',
                    'quantile_sample_capacity': 16,
                    'event_ring_capacity': 16,
                    'episode_ring_capacity': 4,
                    'queue_condition_ring_capacity': 16,
                    'debug_long_form_enabled': False,
                    'debug_max_rows': 0,
                    'debug_max_bytes': 0,
                },
            )
            summary = run_queue_training(
                agent,
                _CriticUpdateQueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            condition_rows = [
                json.loads(line)
                for line in (
                    Path(temporary_directory) / 'compact_queue_condition_summaries.jsonl'
                ).read_text(encoding='utf-8').splitlines()
                if json.loads(line)['stage'] == 'queue_condition'
            ]

        self.assertEqual(summary['queue_completed_rounds'], 2.0)
        self.assertGreater(summary['queue_update_count'], 0.0)
        self.assertGreater(agent.update_step, 0)
        round_two = [
            row for row in condition_rows
            if row.get('condition_instance_id') in {103.0, 104.0}
        ]
        self.assertEqual(len(round_two), 2)
        for row in round_two:
            self.assertGreater(row['queue_condition_batch_count'], 0.0)
            self.assertGreater(row['queue_condition_batch_sample_count'], 0.0)
            sampled_source_total = sum(
                value
                for key, value in row.items()
                if key.startswith('queue_condition_batch_source_') and key.endswith('_sample_count')
            )
            self.assertEqual(sampled_source_total, row['queue_condition_batch_sample_count'])

    def test_queue_skip_quarantines_updates_despite_existing_coverage(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {
                'hidden_dims': [8],
                'td3_bc_enabled': False,
                'td3_bc_weight': 1.0,
                'policy_delay': 4,
            },
            'train': {
                'online_rl_enabled': True,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 2,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'critic_exploration_noise': 0.0,
                'actor_exploration_noise': 0.0,
                'skip_failed_condition_on_train_timeout': True,
                'require_zero_skipped_conditions_for_updates': True,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)
        raw_observation = np.zeros(3, dtype=np.float32)
        raw_action = np.zeros(2, dtype=np.float32)
        for condition_id in (3, 4):
            replay.add(
                raw_observation,
                raw_action,
                0.0,
                raw_observation,
                False,
                condition_id=condition_id,
            )

        with tempfile.TemporaryDirectory(prefix='queue_skip_raw_coverage_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            summary = run_queue_training(
                agent,
                _SkippedCriticUpdateQueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )

        self.assertEqual(summary['queue_completed_rounds'], 2.0)
        self.assertEqual(summary['queue_all_conditions_covered'], 1.0)
        self.assertEqual(summary['queue_skipped_condition_count'], 1.0)
        self.assertEqual(summary['queue_updates_quarantined_after_skip'], 1.0)
        self.assertEqual(summary['queue_training_valid'], 0.0)
        self.assertEqual(summary['queue_update_count'], 0.0)
        self.assertEqual(agent.update_step, 0)

    def test_queue_safety_abort_persists_diagnostics(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
                'low_power_threshold': None,
                'negative_power_patience': 1,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_safety_abort_') as temporary_directory:
            output_dir = Path(temporary_directory)
            logger = ExperimentLogger(output_dir)
            with self.assertRaisesRegex(RuntimeError, 'negative_power_steps=1/1'):
                run_queue_training(
                    agent,
                    _NegativePowerQueueEnvironment(),
                    controller,
                    replay,
                    config,
                    logger,
                    torch.device('cpu'),
                )
            with (output_dir / 'queue_abort_diagnostics.json').open(encoding='utf-8') as file:
                diagnostics = json.load(file)

            self.assertEqual(diagnostics['queue_abort_reason_code'], 1.0)
            self.assertEqual(diagnostics['queue_abort_power'], -1.0)
            self.assertEqual(diagnostics['queue_abort_negative_power_steps'], 1.0)
            self.assertTrue((output_dir / 'queue_train_step_metrics.csv').is_file())

    def test_queue_timeout_log_reports_the_current_log_window(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_timeout_log_') as temporary_directory:
            logger = ExperimentLogger(temporary_directory)
            console = io.StringIO()
            with redirect_stdout(console), self.assertRaisesRegex(TimeoutError, 'condition=4'):
                run_queue_training(
                    agent,
                    _DelayedTimeoutQueueEnvironment(),
                    controller,
                    replay,
                    config,
                    logger,
                    torch.device('cpu'),
                )

        self.assertIn('train_timeouts_window=1', console.getvalue())
        self.assertGreaterEqual(console.getvalue().count('train_timeouts_window=0'), 2)

    def test_queue_skips_failed_condition_when_bladed_advances_to_next_instance(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'skip_failed_condition_on_train_timeout': True,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_skip_failed_condition_') as temporary_directory:
            output_dir = Path(temporary_directory)
            logger = ExperimentLogger(output_dir)
            summary = run_queue_training(
                agent,
                _SkippedConditionQueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            skip_path = output_dir / 'queue_skipped_condition_03_instance_101.json'
            self.assertTrue(skip_path.is_file())
            with skip_path.open(encoding='utf-8') as handle:
                diagnostics = json.load(handle)
            skip_metrics = (output_dir / 'queue_condition_skip_metrics.csv').read_text(encoding='utf-8')

        self.assertEqual(summary['queue_completed_rounds'], 1.0)
        self.assertEqual(summary['queue_skipped_condition_count'], 1.0)
        self.assertEqual(diagnostics['queue_skip_last_train_elapsed_s'], 0.02)
        self.assertEqual(diagnostics['queue_skip_recovered_condition_id'], 4.0)
        self.assertIn('queue_skip_reason_code,1.0', skip_metrics)

    def test_queue_skips_short_train_when_bladed_directly_switches_instance(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'train_sim_s': 0.04,
                'train_sim_tolerance_s': 1.0e-6,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'skip_failed_condition_on_train_timeout': True,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_direct_skip_failed_condition_') as temporary_directory:
            output_dir = Path(temporary_directory)
            logger = ExperimentLogger(output_dir)
            summary = run_queue_training(
                agent,
                _DirectSkippedConditionQueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            with (output_dir / 'queue_skipped_condition_03_instance_101.json').open(encoding='utf-8') as handle:
                diagnostics = json.load(handle)

        self.assertEqual(summary['queue_completed_rounds'], 1.0)
        self.assertEqual(summary['queue_skipped_condition_count'], 1.0)
        self.assertEqual(diagnostics['queue_skip_trigger_code'], 2.0)
        self.assertEqual(diagnostics['queue_skip_last_train_elapsed_s'], 0.02)

    def test_queue_retries_same_condition_without_consuming_lifecycle_slot(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'skip_failed_condition_on_train_timeout': True,
                'require_zero_skipped_conditions_for_updates': True,
                'require_zero_retried_condition_attempts_for_updates': True,
                'abort_on_retried_condition_attempt': False,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)

        with tempfile.TemporaryDirectory(prefix='queue_retry_failed_condition_') as temporary_directory:
            output_dir = Path(temporary_directory)
            logger = ExperimentLogger(output_dir)
            summary = run_queue_training(
                agent,
                _RetriedConditionQueueEnvironment(),
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            retry_path = output_dir / 'queue_retried_condition_03_instance_101_to_102.json'
            self.assertTrue(retry_path.is_file())
            self.assertFalse((output_dir / 'queue_skipped_condition_03_instance_101.json').exists())
            with retry_path.open(encoding='utf-8') as handle:
                diagnostics = json.load(handle)

        self.assertEqual(summary['queue_completed_rounds'], 1.0)
        self.assertEqual(summary['queue_skipped_condition_count'], 0.0)
        self.assertEqual(summary['queue_retried_condition_attempt_count'], 1.0)
        self.assertEqual(summary['queue_updates_quarantined_after_retry'], 1.0)
        self.assertEqual(summary['queue_training_valid'], 0.0)
        self.assertEqual(diagnostics['queue_retry_condition_id'], 3.0)
        self.assertEqual(diagnostics['queue_retry_failed_condition_instance_id'], 101.0)
        self.assertEqual(diagnostics['queue_retry_recovered_condition_instance_id'], 102.0)

    def test_strict_retry_aborts_before_sending_an_action_to_the_retry_instance(self) -> None:
        config = {
            'data_schema_version': 'bladed_udp_19_si_my_v1',
            'algorithm': 'td3',
            'agent': {'hidden_dims': [8], 'td3_bc_enabled': False, 'td3_bc_weight': 1.0},
            'train': {
                'online_rl_enabled': False,
                'online_eval_enabled': False,
                'eval_every': 0,
                'batch_size': 2,
                'updates_per_step': 1,
                'update_every_steps': 1,
                'step_log_every': 10,
                'warmup_exploration_noise': 0.0,
                'exploration_noise': 0.0,
                'condition_replay': {
                    'sampling': 'condition_stratified',
                    'capacity_per_condition': 4,
                    'min_valid_transitions_per_condition': 1,
                    'min_per_condition_in_batch': 1,
                },
            },
            'queue': {
                'condition_ids': [3, 4],
                'total_condition_rounds': 1,
                'critic_start_after_rounds': 1,
                'actor_start_after_rounds': 2,
                'collect_exploration_noise': 0.0,
                'skip_failed_condition_on_train_timeout': True,
                'require_zero_retried_condition_attempts_for_updates': True,
                'abort_on_retried_condition_attempt': True,
            },
            'offline_eval': {'enabled': False},
        }
        controller = SafeResidualController(
            {
                'action_low': [13_000_000.0, 0.0],
                'action_high': [21_000_000.0, 0.33],
                'action_limiter_owner': 'controller',
                'action_delta_low': [-15_000.0, -0.0003],
                'action_delta_high': [15_000.0, 0.0003],
            },
            {
                'baseline_action': [20_000_000.0, 0.05],
                'residual_limit': [1_000_000.0, 0.05],
                'safe_action_low': [13_000_000.0, 0.0],
                'safe_action_high': [21_000_000.0, 0.33],
                'ood_threshold': 100.0,
            },
        )
        controller.set_observation_distribution(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        agent = TD3Agent(obs_dim=3, action_dim=2, config=config['agent'], device=torch.device('cpu'))
        agent.set_actor_obs_normalizer(torch.zeros(3), torch.ones(3))
        agent.set_critic_obs_normalizer(torch.zeros(3), torch.ones(3))
        replay = ConditionStratifiedReplayBuffer(3, 2, [3, 4], capacity_per_condition=4, min_per_condition_in_batch=1)
        environment = _RetriedConditionQueueEnvironment()

        with tempfile.TemporaryDirectory(prefix='queue_strict_retry_abort_') as temporary_directory:
            output_dir = Path(temporary_directory)
            logger = ExperimentLogger(output_dir)
            with self.assertRaisesRegex(RuntimeError, 'strict retry safety is enabled'):
                run_queue_training(
                    agent,
                    environment,
                    controller,
                    replay,
                    config,
                    logger,
                    torch.device('cpu'),
                )
            logger.flush()
            retry_path = output_dir / 'queue_retried_condition_03_instance_101_to_102.json'
            abort_path = output_dir / 'queue_abort_diagnostics.json'
            self.assertTrue(retry_path.is_file())
            self.assertTrue(abort_path.is_file())
            with abort_path.open(encoding='utf-8') as handle:
                diagnostics = json.load(handle)

        self.assertEqual(environment._step_index, 2)
        self.assertEqual(diagnostics['queue_abort_reason_code'], 4.0)
        self.assertEqual(diagnostics['queue_abort_condition_id'], 3.0)
        self.assertEqual(diagnostics['queue_abort_condition_instance_id'], 101.0)

    def test_verified_physical_skip_relabels_the_next_manifest_case_without_retrying(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_verified_physical_skip_') as temporary_directory:
            root = Path(temporary_directory)
            config, controller, agent, replay = _build_verified_physical_skip_fixture(root)
            environment = _VerifiedPhysicalSkipQueueEnvironment()
            output_dir = root / 'output'
            logger = ExperimentLogger(output_dir)

            summary = run_queue_training(
                agent,
                environment,
                controller,
                replay,
                config,
                logger,
                torch.device('cpu'),
            )
            logger.flush()
            with (output_dir / 'queue_physical_skip_probe_condition_03_instance_101.json').open(
                encoding='utf-8'
            ) as handle:
                probe_diagnostics = json.load(handle)
            with (output_dir / 'queue_skipped_condition_03_instance_101.json').open(encoding='utf-8') as handle:
                skip_diagnostics = json.load(handle)

        self.assertEqual(environment.probe_calls, 2)
        self.assertEqual(environment.advance_calls, 1)
        self.assertEqual(environment.step_metadata, [(3, 101), (4, 103)])
        self.assertEqual(summary['queue_completed_rounds'], 1.0)
        self.assertEqual(summary['queue_skipped_condition_count'], 1.0)
        self.assertEqual(summary['queue_retried_condition_attempt_count'], 0.0)
        self.assertEqual(summary['queue_training_valid'], 0.0)
        self.assertEqual(probe_diagnostics['queue_physical_skip_probe_approved'], 1.0)
        self.assertEqual(probe_diagnostics['queue_case_identity_probe_matched_condition_id'], 4.0)
        self.assertEqual(probe_diagnostics['queue_physical_skip_remapped_condition_id'], 4.0)
        self.assertEqual(skip_diagnostics['queue_skip_trigger_code'], 3.0)

    def test_verified_physical_skip_rejects_retry_matching_the_current_slot(self) -> None:
        with tempfile.TemporaryDirectory(prefix='queue_unapproved_physical_skip_') as temporary_directory:
            root = Path(temporary_directory)
            config, controller, agent, replay = _build_verified_physical_skip_fixture(root)
            environment = _VerifiedPhysicalSkipQueueEnvironment((3.0, 3.1, 3.2))
            output_dir = root / 'output'
            logger = ExperimentLogger(output_dir)

            with self.assertRaisesRegex(RuntimeError, 'strict retry safety is enabled'):
                run_queue_training(
                    agent,
                    environment,
                    controller,
                    replay,
                    config,
                    logger,
                    torch.device('cpu'),
                )
            logger.flush()
            probe_path = output_dir / 'queue_physical_skip_probe_condition_03_instance_101.json'
            retry_path = output_dir / 'queue_retried_condition_03_instance_101_to_102.json'
            self.assertTrue(probe_path.is_file())
            self.assertTrue(retry_path.is_file())
            with probe_path.open(encoding='utf-8') as handle:
                probe_diagnostics = json.load(handle)

        self.assertEqual(environment.probe_calls, 2)
        self.assertEqual(environment.advance_calls, 0)
        self.assertEqual(environment.step_metadata, [(3, 101)])
        self.assertEqual(probe_diagnostics['queue_physical_skip_probe_approved'], 0.0)
        self.assertEqual(probe_diagnostics['queue_case_identity_probe_matched_condition_id'], 3.0)


if __name__ == '__main__':
    unittest.main()