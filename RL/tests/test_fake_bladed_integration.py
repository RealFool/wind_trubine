from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import torch

from RL.envs.wind_turbine_env import BLADED_CSV_HEADER, WindTurbineEnv


class FakeBladedUdpServer:
    def __init__(
        self,
        target_host: str,
        target_port: int,
        pause_after_frame_id: int | None = None,
        pause_duration_s: float = 0.0,
        pause_after_first_accepted_action: bool = False,
        use_queue_envelope: bool = False,
        queue_protocol_version: str = 'bladed_udp_19_si_my_queue_v1',
        switch_after_accepted_actions: int | None = None,
        queue_emit_end_before_switch: bool = False,
        queue_load_pause_s: float = 0.0,
        queue_failure_after_accepted_actions: int | None = None,
        queue_failure_pause_s: float = 0.0,
        queue_failure_next_condition_id: int = 3,
    ) -> None:
        self.target = (target_host, int(target_port))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._torque_cmd = 20_000_000.0
        self._pitch_cmd = 0.05
        self._sim_time = 0.0
        self._frame_id = 0
        self._pause_after_frame_id = pause_after_frame_id
        self._pause_duration_s = float(pause_duration_s)
        self._pause_after_first_accepted_action = bool(pause_after_first_accepted_action)
        self._pause_completed = False
        self._use_queue_envelope = bool(use_queue_envelope)
        self._queue_protocol_version = queue_protocol_version
        if self._use_queue_envelope and self._queue_protocol_version not in {
            'bladed_udp_19_si_my_queue_v1',
            'bladed_udp_19_si_my_queue_v2',
        }:
            raise ValueError('Fake queue protocol version must be v1 or v2')
        self._switch_after_accepted_actions = switch_after_accepted_actions
        self._queue_emit_end_before_switch = bool(queue_emit_end_before_switch)
        self._queue_load_pause_s = float(queue_load_pause_s)
        self._queue_failure_after_accepted_actions = queue_failure_after_accepted_actions
        self._queue_failure_pause_s = float(queue_failure_pause_s)
        self._queue_failure_next_condition_id = int(queue_failure_next_condition_id)
        self._queue_epoch = 1
        self._condition_instance_id = 1
        self._condition_id = 12
        self._phase = 'TRAIN'
        self._phase_elapsed_s = 0.0
        self._queue_switch_completed = False
        self._queue_switch_pending = False
        self._queue_failure_completed = False
        self.accepted_action_frame_ids: list[int] = []
        self.accepted_action_keys: list[tuple[int, int, int]] = []
        self.accepted_actions: list[tuple[float, float]] = []
        self.accepted_queue_commands: list[str] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name='FakeBladedUdpServer', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('127.0.0.1', 0))
        sock.settimeout(0.05)
        try:
            while not self._stop.is_set():
                payload = self._build_payload()
                sock.sendto(payload.encode('utf-8'), self.target)
                reply_accepted = False
                try:
                    data, _ = sock.recvfrom(1024)
                    reply_accepted = self._update_action(data.decode('utf-8', errors='ignore'))
                except socket.timeout:
                    pass
                except ConnectionResetError:
                    # On Windows UDP, remote close can surface as WSAECONNRESET.
                    pass
                if not reply_accepted:
                    continue
                if (
                    self._use_queue_envelope
                    and not self._queue_failure_completed
                    and self._queue_failure_after_accepted_actions is not None
                    and len(self.accepted_action_frame_ids) >= self._queue_failure_after_accepted_actions
                ):
                    if self._queue_failure_pause_s > 0.0:
                        time.sleep(self._queue_failure_pause_s)
                    self._queue_epoch += 1
                    self._condition_instance_id += 1
                    self._condition_id = self._queue_failure_next_condition_id
                    self._phase = 'TRAIN'
                    self._phase_elapsed_s = 0.0
                    self._frame_id = 0
                    self._sim_time = 0.0
                    self._queue_failure_completed = True
                    continue
                if (
                    self._use_queue_envelope
                    and self._queue_switch_pending
                    and not self._queue_switch_completed
                ):
                    if self._queue_load_pause_s > 0.0:
                        time.sleep(self._queue_load_pause_s)
                    self._queue_epoch += 1
                    self._condition_instance_id += 1
                    self._condition_id = 3
                    self._phase = 'SETTLE'
                    self._phase_elapsed_s = 0.0
                    self._frame_id = 0
                    self._sim_time = 0.0
                    self._queue_switch_completed = True
                    self._queue_switch_pending = False
                    continue
                if (
                    self._use_queue_envelope
                    and not self._queue_switch_completed
                    and self._switch_after_accepted_actions is not None
                    and len(self.accepted_action_frame_ids) >= self._switch_after_accepted_actions
                ):
                    if self._queue_emit_end_before_switch:
                        self._phase = 'END'
                        self._queue_switch_pending = True
                        self._phase_elapsed_s = 0.0
                        self._sim_time += 0.1
                        self._frame_id += 1
                        time.sleep(0.01)
                        continue
                    self._queue_epoch += 1
                    self._condition_instance_id += 1
                    self._condition_id = 3
                    self._phase = 'SETTLE'
                    self._phase_elapsed_s = 0.0
                    self._frame_id = 0
                    self._sim_time = 0.0
                    self._queue_switch_completed = True
                    continue
                if (
                    reply_accepted
                    and not self._pause_completed
                    and (
                        self._pause_after_first_accepted_action
                        or self._pause_after_frame_id == self._frame_id
                    )
                    and self._pause_duration_s > 0.0
                ):
                    time.sleep(self._pause_duration_s)
                    self._pause_completed = True
                self._sim_time += 0.1
                self._frame_id += 1
                self._phase_elapsed_s += 0.1
                time.sleep(0.01)
        finally:
            sock.close()

    def _update_action(self, text: str) -> bool:
        parts = [segment.strip() for segment in text.split(',')]
        try:
            if self._use_queue_envelope:
                if len(parts) not in {4, 5}:
                    return False
                queue_epoch = int(float(parts[0]))
                condition_instance_id = int(float(parts[1]))
                frame_id = int(float(parts[2]))
                if (
                    queue_epoch != self._queue_epoch
                    or condition_instance_id != self._condition_instance_id
                    or frame_id != self._frame_id
                ):
                    return False
                if len(parts) == 4:
                    if self._queue_protocol_version != 'bladed_udp_19_si_my_queue_v2':
                        return False
                    command = parts[3]
                    if command not in {'QUEUE_PROBE', 'QUEUE_ADVANCE_INCOMPLETE_SLOT'}:
                        return False
                    self.accepted_queue_commands.append(command)
                    if command == 'QUEUE_ADVANCE_INCOMPLETE_SLOT':
                        self._queue_epoch += 1
                        self._condition_instance_id += 1
                        self._condition_id += 1
                        self._phase = 'TRAIN'
                        self._phase_elapsed_s = 0.0
                        self._frame_id = -1
                        self._sim_time = -0.1
                    return True
                torque_index, pitch_index = 3, 4
                self.accepted_action_keys.append((queue_epoch, condition_instance_id, frame_id))
            else:
                if len(parts) < 3:
                    return False
                frame_id = int(float(parts[0]))
                if frame_id != self._frame_id:
                    return False
                torque_index, pitch_index = 1, 2
            self._torque_cmd = float(parts[torque_index])
            self._pitch_cmd = float(parts[pitch_index])
            self.accepted_actions.append((self._torque_cmd, self._pitch_cmd))
            self.accepted_action_frame_ids.append(frame_id)
            return True
        except ValueError:
            return False

    def _build_payload(self) -> str:
        idx = {name: i for i, name in enumerate(BLADED_CSV_HEADER)}
        row = [0.0] * len(BLADED_CSV_HEADER)

        rotor_speed = 0.08 + 0.004 * ((self._sim_time % 5.0) / 5.0)
        generator_speed = 0.75 + 0.01 * ((self._sim_time % 3.0) / 3.0)
        wind_speed = 12.8 + 0.5 * ((self._sim_time % 7.0) / 7.0)
        real_power = max(0.0, min(15.0e6, self._torque_cmd * generator_speed))

        def setv(name: str, value: float) -> None:
            row[idx[name]] = float(value)

        setv('Time', self._sim_time)
        setv('I_RotorAzimuth', (self._sim_time * 0.2) % (2.0 * 3.1415926))
        setv('I_RotorSpeed', rotor_speed)
        setv('I_AverageWindSpeed', wind_speed)
        setv('I_YawError', 0.1)
        setv('I_ElectricalTorque', self._torque_cmd)
        setv('I_GeneratorSpeed', generator_speed)
        setv('I_RealPower', real_power)
        setv('I_PitchPosition1', self._pitch_cmd)
        setv('I_PitchPosition2', self._pitch_cmd)
        setv('I_PitchPosition3', self._pitch_cmd)
        setv('I_NacelleForeAftAcceleration', 0.15)
        setv('I_NacelleSideSideAcceleration', 0.08)

        setv('Electrical power', real_power)
        setv('Blade root 1 My (Nm)', 45.0e6)
        setv('Blade root 2 My (Nm)', 44.5e6)
        setv('Blade root 3 My (Nm)', 45.5e6)
        setv('O_PcsTorqueDemand', self._torque_cmd)
        setv('O_PitchPositionDemand1', self._pitch_cmd)

        payload = ','.join(f'{value:.6f}' for value in row)
        if self._use_queue_envelope:
            prefix = (
                f'{self._queue_epoch},{self._condition_instance_id},{self._condition_id},'
                f'{self._phase},{self._phase_elapsed_s:.6f},{self._frame_id},{self._sim_time:.6f},'
            )
            if self._queue_protocol_version == 'bladed_udp_19_si_my_queue_v2':
                prefix = f'{self._queue_protocol_version},' + prefix
            return (
                f'{prefix}{payload}'
            )
        return f'{self._frame_id},{self._sim_time:.6f},{payload}'


def _pick_free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class TestFakeBladedIntegration(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        # The subprocess BC smoke test uses a stable logger path. Remove only
        # that test-owned output so it cannot be mistaken for a real run.
        test_output = Path(__file__).resolve().parents[1] / 'experiments' / 'it_outputs' / 'it_fake_bladed_bc_rl'
        shutil.rmtree(test_output, ignore_errors=True)

    @staticmethod
    def _queue_env(udp_port: int, protocol_version: str) -> WindTurbineEnv:
        return WindTurbineEnv(
            {
                'backend': 'bladed_udp',
                'udp_host': '127.0.0.1',
                'udp_port': udp_port,
                'udp_timeout_s': 0.1,
                'udp_reset_wait_s': 2.0,
                'udp_inter_frame_wait_s': 2.0,
                'condition_load_timeout_s': 2.0,
                'udp_require_frame_id': True,
                'udp_protocol_version': protocol_version,
                'require_condition_metadata': True,
                'udp_strict_field_count': True,
                'episode_length': 2,
                'rollout_segment_steps': 2,
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {'power_weight': 0.0, 'blade_root_weight': 0.0},
            },
            {},
            torch.device('cpu'),
        )

    def test_v2_queue_envelope_accepts_actions_and_verified_commands(self) -> None:
        udp_port = _pick_free_udp_port()
        env = self._queue_env(udp_port, 'bladed_udp_19_si_my_queue_v2')
        server = FakeBladedUdpServer(
            '127.0.0.1',
            udp_port,
            use_queue_envelope=True,
            queue_protocol_version='bladed_udp_19_si_my_queue_v2',
        )
        server.start()
        try:
            env.reset()
            env.probe_pending_condition()
            env.advance_pending_incomplete_slot()
            result = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
        finally:
            server.stop()
            env.close()

        self.assertEqual(server.accepted_queue_commands, ['QUEUE_PROBE', 'QUEUE_ADVANCE_INCOMPLETE_SLOT'])
        self.assertEqual(server.accepted_action_keys, [(2, 2, 0)])
        self.assertEqual(result.info['udp_queue_protocol_v2'], 1.0)
        self.assertEqual(result.info['udp_queue_epoch'], 2.0)
        self.assertEqual(result.info['udp_condition_instance_id'], 2.0)
    def test_queue_epoch_accepts_frame_time_reset_and_marks_boundary(self) -> None:
        udp_port = _pick_free_udp_port()
        env = WindTurbineEnv(
            {
                'backend': 'bladed_udp',
                'udp_host': '127.0.0.1',
                'udp_port': udp_port,
                'udp_timeout_s': 0.1,
                'udp_reset_wait_s': 2.0,
                'udp_inter_frame_wait_s': 2.0,
                'udp_require_frame_id': True,
                'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
                'require_condition_metadata': True,
                'udp_strict_field_count': True,
                'episode_length': 2,
                'rollout_segment_steps': 2,
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {
                    'power_reward_mode': 'energy_increment',
                    'energy_reference_j': 750_000.0,
                    'power_weight': 1.0,
                    'blade_root_weight': 0.0,
                    'acc_weight': 0.0,
                    'shaft_weight': 0.0,
                    'pitch_rate_weight': 0.0,
                    'torque_rate_weight': 0.0,
                    'command_excess_weight': 0.0,
                    'power_tracking_weight': 0.0,
                },
            },
            {},
            torch.device('cpu'),
        )
        server = FakeBladedUdpServer(
            '127.0.0.1',
            udp_port,
            use_queue_envelope=True,
            switch_after_accepted_actions=1,
        )
        server.start()
        try:
            env.reset()
            result = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
            next_obs, next_info = env.start_pending_condition()
        finally:
            server.stop()
            env.close()

        self.assertTrue(np.isfinite(next_obs).all())
        self.assertTrue(result.done)
        self.assertEqual(result.info['condition_boundary'], 1.0)
        self.assertEqual(result.info['transition_store_eligible'], 0.0)
        self.assertEqual(result.info['reward_pid_power_reference_raw'], 0.0)
        self.assertEqual(result.info['reward_relative_power_raw'], 0.0)
        self.assertEqual(result.info['reward_relative_energy_increment_j'], 0.0)
        self.assertEqual(result.info['udp_next_queue_epoch'], 2.0)
        self.assertEqual(result.info['udp_next_condition_instance_id'], 2.0)
        self.assertEqual(result.info['udp_next_condition_id'], 3.0)
        self.assertEqual(result.info['udp_next_phase_code'], 1.0)
        self.assertEqual(next_info['udp_queue_epoch'], 2.0)
        self.assertEqual(next_info['condition_train_active'], 0.0)
        self.assertEqual(server.accepted_action_keys, [(1, 1, 0)])

    def test_start_pending_condition_acknowledges_end_before_next_case(self) -> None:
        udp_port = _pick_free_udp_port()
        env = WindTurbineEnv(
            {
                'backend': 'bladed_udp',
                'udp_host': '127.0.0.1',
                'udp_port': udp_port,
                'udp_timeout_s': 0.01,
                'udp_reset_wait_s': 2.0,
                'udp_inter_frame_wait_s': 0.05,
                'condition_load_timeout_s': 0.2,
                'udp_require_frame_id': True,
                'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
                'require_condition_metadata': True,
                'udp_strict_field_count': True,
                'episode_length': 2,
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {'power_weight': 0.0, 'blade_root_weight': 0.0},
            },
            {},
            torch.device('cpu'),
        )
        server = FakeBladedUdpServer(
            '127.0.0.1',
            udp_port,
            use_queue_envelope=True,
            switch_after_accepted_actions=1,
            queue_emit_end_before_switch=True,
            queue_load_pause_s=0.08,
        )
        server.start()
        try:
            env.reset()
            train_to_end = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
            next_obs, next_info = env.start_pending_condition()
        finally:
            server.stop()
            env.close()

        self.assertTrue(np.isfinite(next_obs).all())
        self.assertTrue(train_to_end.done)
        self.assertEqual(next_info['udp_queue_epoch'], 2.0)
        self.assertEqual(next_info['udp_condition_instance_id'], 2.0)
        self.assertEqual(next_info['udp_condition_id'], 3.0)
        self.assertEqual(next_info['udp_phase_code'], 1.0)
        self.assertEqual(server.accepted_action_keys, [(1, 1, 0), (1, 1, 1)])

    def test_queue_train_timeout_recovers_the_next_condition_instance(self) -> None:
        udp_port = _pick_free_udp_port()
        env = WindTurbineEnv(
            {
                'backend': 'bladed_udp',
                'udp_host': '127.0.0.1',
                'udp_port': udp_port,
                'udp_timeout_s': 0.01,
                'udp_reset_wait_s': 2.0,
                'udp_inter_frame_wait_s': 0.02,
                'condition_load_timeout_s': 0.5,
                'udp_require_frame_id': True,
                'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
                'require_condition_metadata': True,
                'udp_strict_field_count': True,
                'episode_length': 2,
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {'power_weight': 0.0, 'blade_root_weight': 0.0},
            },
            {},
            torch.device('cpu'),
        )
        server = FakeBladedUdpServer(
            '127.0.0.1',
            udp_port,
            use_queue_envelope=True,
            queue_failure_after_accepted_actions=1,
            queue_failure_pause_s=0.08,
            queue_failure_next_condition_id=15,
        )
        server.start()
        try:
            env.reset()
            timeout_result = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
            recovered_obs, recovery_info = env.recover_after_train_timeout()
        finally:
            server.stop()
            env.close()

        self.assertTrue(timeout_result.done)
        self.assertEqual(timeout_result.info['udp_inter_frame_timeout'], 1.0)
        self.assertTrue(np.isfinite(recovered_obs).all())
        self.assertEqual(recovery_info['udp_timeout_recovery_active'], 1.0)
        self.assertEqual(recovery_info['udp_timeout_recovery_condition_changed'], 1.0)
        self.assertEqual(recovery_info['udp_timeout_recovery_from_queue_epoch'], 1.0)
        self.assertEqual(recovery_info['udp_timeout_recovery_from_condition_instance_id'], 1.0)
        self.assertEqual(recovery_info['udp_queue_epoch'], 2.0)
        self.assertEqual(recovery_info['udp_condition_instance_id'], 2.0)
        self.assertEqual(recovery_info['udp_condition_id'], 15.0)
        self.assertGreater(recovery_info['udp_timeout_recovery_wait_ms'], 20.0)

    def test_queue_train_timeout_resumes_the_same_condition_instance(self) -> None:
        udp_port = _pick_free_udp_port()
        env = WindTurbineEnv(
            {
                'backend': 'bladed_udp',
                'udp_host': '127.0.0.1',
                'udp_port': udp_port,
                'udp_timeout_s': 0.01,
                'udp_reset_wait_s': 2.0,
                'udp_inter_frame_wait_s': 0.02,
                'condition_load_timeout_s': 0.5,
                'udp_require_frame_id': True,
                'udp_protocol_version': 'bladed_udp_19_si_my_queue_v1',
                'require_condition_metadata': True,
                'udp_strict_field_count': True,
                'episode_length': 2,
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {'power_weight': 0.0, 'blade_root_weight': 0.0},
            },
            {},
            torch.device('cpu'),
        )
        server = FakeBladedUdpServer(
            '127.0.0.1',
            udp_port,
            use_queue_envelope=True,
            pause_after_first_accepted_action=True,
            pause_duration_s=0.08,
        )
        server.start()
        try:
            env.reset()
            timeout_result = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
            recovered_obs, recovery_info = env.recover_after_train_timeout()
        finally:
            server.stop()
            env.close()

        self.assertTrue(timeout_result.done)
        self.assertEqual(timeout_result.info['udp_inter_frame_timeout'], 1.0)
        self.assertTrue(np.isfinite(recovered_obs).all())
        self.assertEqual(recovery_info['udp_timeout_recovery_active'], 1.0)
        self.assertEqual(recovery_info['udp_timeout_recovery_condition_changed'], 0.0)
        self.assertEqual(recovery_info['udp_queue_epoch'], 1.0)
        self.assertEqual(recovery_info['udp_condition_instance_id'], 1.0)
        self.assertEqual(recovery_info['udp_condition_id'], 12.0)
        self.assertEqual(recovery_info['condition_train_active'], 1.0)
        self.assertGreater(recovery_info['udp_timeout_recovery_wait_ms'], 20.0)

    def test_energy_reward_uses_consecutive_udp_frame_values(self) -> None:
        udp_port = _pick_free_udp_port()
        env = WindTurbineEnv(
            {
                'backend': 'bladed_udp',
                'udp_host': '127.0.0.1',
                'udp_port': udp_port,
                'udp_timeout_s': 0.1,
                'udp_reset_wait_s': 2.0,
                'udp_inter_frame_wait_s': 2.0,
                'udp_pause_threshold_s': 1.0,
                'udp_require_frame_id': True,
                'udp_strict_field_count': True,
                'episode_length': 2,
                'action_low': [0.0, 0.0],
                'action_high': [21_000_000.0, 0.43],
                'rated_power': 15_000_000.0,
                'bladed_step_duration_ms': 20.0,
                'reward': {
                    'power_reward_mode': 'energy_increment',
                    'energy_reference_j': 750_000.0,
                    'power_weight': 1.0,
                    'blade_root_weight': 0.0,
                    'acc_weight': 0.0,
                    'shaft_weight': 0.0,
                    'pitch_rate_weight': 0.0,
                    'torque_rate_weight': 0.0,
                    'command_excess_weight': 0.0,
                    'power_tracking_weight': 0.0,
                },
            },
            {},
            torch.device('cpu'),
        )
        server = FakeBladedUdpServer('127.0.0.1', udp_port)
        server.start()
        try:
            env.reset()
            result = env.step(np.asarray([-1.0, -1.0], dtype=np.float32))
        finally:
            server.stop()
            env.close()

        self.assertEqual(result.info['reward_energy_increment_valid'], 1.0)
        self.assertAlmostEqual(result.info['reward_energy_delta_t_s'], 0.1)
        self.assertAlmostEqual(result.info['reward_energy_increment_j'], 750_000.0, places=1)
        self.assertAlmostEqual(result.reward, 1.0, places=6)
        self.assertGreaterEqual(result.info['udp_action_sent_monotonic_s'], result.info['udp_receive_monotonic_s'])

if __name__ == '__main__':
    unittest.main()
