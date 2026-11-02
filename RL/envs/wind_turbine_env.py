from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import select
import socket
import time
from typing import Any, Callable

import numpy as np
import torch

from RL.utils.constants import (
    ACTION_DIM,
    ACTION_VARIABLES,
    BLADED_UDP_VARIABLES,
    CONTROL_OBS_DIM,
    OBS_DIM,
    REWARD_VARIABLES,
    STATE_DIM,
    STATE_VARIABLES,
)
from RL.utils.control_contract import PhysicalActionTransform, action_array_float64
from RL.utils.normalize import ActionScaler
from RL.utils.observation import append_previous_action_observation, build_observation_from_mapping
from RL.utils.operating_state import (
    CONTROL_REGION_ENCODING_ORDER,
    OPTIMIZATION_STATE_ENCODING_ORDER,
    OperatingPoint,
)
from RL.utils.power_fields import finite_float_or_none, synchronize_power_fields
from RL.utils.reward import RewardFunction
from RL.utils.state_action_artifact import (
    STATE_AWARE_CONTROL_OBS_DIM,
    StateActionEnvelopeArtifact,
    append_state_regime_observation,
)


BLADED_CSV_HEADER = list(BLADED_UDP_VARIABLES)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_UDP_PROTOCOL_VERSION = 'bladed_udp_19_si_my_v1'
QUEUE_UDP_PROTOCOL_VERSION_V1 = 'bladed_udp_19_si_my_queue_v1'
QUEUE_UDP_PROTOCOL_VERSION = 'bladed_udp_19_si_my_queue_v2'
QUEUE_UDP_PROTOCOL_VERSIONS = frozenset({QUEUE_UDP_PROTOCOL_VERSION_V1, QUEUE_UDP_PROTOCOL_VERSION})
QUEUE_PROBE_COMMAND = 'QUEUE_PROBE'
QUEUE_ADVANCE_INCOMPLETE_SLOT_COMMAND = 'QUEUE_ADVANCE_INCOMPLETE_SLOT'
QUEUE_PHASE_CODES = {
    None: 0.0,
    'SETTLE': 1.0,
    'TRAIN': 2.0,
    'END': 3.0,
}


class UdpProtocolVersionMismatch(ValueError):
    pass


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, float]


@dataclass
class UdpPacket:
    payload: dict[str, float]
    frame_id: int | None
    sim_time: float | None
    recv_monotonic_s: float
    dropped_stale_frames: int
    queue_epoch: int | None = None
    condition_instance_id: int | None = None
    condition_id: int | None = None
    phase: str | None = None
    phase_elapsed_s: float | None = None
    queue_protocol_version: str | None = None
    # Local parse provenance only. No new UDP, observation or reward fields.
    invalid_payload_fields: tuple[str, ...] = ()


class WindTurbineEnv:
    supports_physical_action = True

    def __init__(self, env_config: dict[str, Any], data_config: dict[str, Any], device: torch.device) -> None:
        del data_config
        try:
            self.initialization_config_sha256 = hashlib.sha256(json.dumps(
                env_config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        except (TypeError, ValueError):
            self.initialization_config_sha256 = None
        self.device = device
        self.backend = str(env_config.get('backend', 'bladed_udp')).lower()
        if self.backend != 'bladed_udp':
            raise ValueError(f'Unsupported env backend: {self.backend}. Expected bladed_udp.')

        self.requires_single_instance = True
        self.episode_length = int(env_config.get('episode_length', 512))
        self.rollout_segment_steps = int(env_config.get('rollout_segment_steps', self.episode_length))
        if self.rollout_segment_steps <= 0:
            raise ValueError('env.rollout_segment_steps must be positive')
        self.action_scaler = ActionScaler(env_config['action_low'], env_config['action_high'])
        self.observation_action_low = self._optional_array(
            env_config.get('observation_action_low'),
            ACTION_DIM,
        )
        self.observation_action_high = self._optional_array(
            env_config.get('observation_action_high'),
            ACTION_DIM,
        )
        if self.observation_action_low is None:
            self.observation_action_low = self.action_scaler.low.copy()
        if self.observation_action_high is None:
            self.observation_action_high = self.action_scaler.high.copy()
        if (
            not np.isfinite(self.observation_action_low).all()
            or not np.isfinite(self.observation_action_high).all()
            or np.any(self.observation_action_low > self.observation_action_high)
        ):
            raise ValueError('env observation action bounds must be finite and ordered')
        self.include_previous_action_observation = bool(env_config.get('include_previous_action_observation', False))
        self.include_operating_state_observation = bool(
            env_config.get('include_operating_state_observation', False)
        )
        self.state_action_artifact: StateActionEnvelopeArtifact | None = None
        self.state_action_artifact_sha256: str | None = None
        self._state_classifier = None
        self._last_state_classification = None
        if self.include_operating_state_observation:
            if not self.include_previous_action_observation:
                raise ValueError('env.include_operating_state_observation requires previous-action observations')
            state_artifact_path = env_config.get('state_action_artifact_path')
            if not isinstance(state_artifact_path, str) or not state_artifact_path.strip():
                raise ValueError('env.include_operating_state_observation requires env.state_action_artifact_path')
            path = Path(state_artifact_path.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            self.state_action_artifact = StateActionEnvelopeArtifact.from_path(path)
            self.state_action_artifact_sha256 = self.state_action_artifact.source_sha256
        self.reward_fn = RewardFunction(env_config)
        self.error_clip = float(env_config.get('error_metric_clip', 1.0e6))

        self.action_delta_low = self._optional_array(env_config.get('action_delta_low'), ACTION_DIM)
        self.action_delta_high = self._optional_array(env_config.get('action_delta_high'), ACTION_DIM)
        self.action_limiter_owner = str(env_config.get('action_limiter_owner', 'env')).lower()
        if self.action_limiter_owner not in {'env', 'controller'}:
            raise ValueError('env.action_limiter_owner must be env or controller')
        self.safe_action_low = self._optional_array(env_config.get('safe_action_low'), ACTION_DIM)
        self.safe_action_high = self._optional_array(env_config.get('safe_action_high'), ACTION_DIM)
        if (self.safe_action_low is None) != (self.safe_action_high is None):
            raise ValueError('env.safe_action_low and env.safe_action_high must be configured together')
        physical_action_low = self.action_scaler.low if self.safe_action_low is None else self.safe_action_low
        physical_action_high = self.action_scaler.high if self.safe_action_high is None else self.safe_action_high
        self.action_transform = PhysicalActionTransform(
            action_low=physical_action_low,
            action_high=physical_action_high,
            action_delta_low=self.action_delta_low,
            action_delta_high=self.action_delta_high,
        )

        self.udp_host = str(env_config.get('udp_host', '0.0.0.0'))
        self.udp_port = int(env_config.get('udp_port', 8888))
        self.udp_timeout_s = float(env_config.get('udp_timeout_s', 1.0))
        self.reset_wait_s = float(env_config.get('udp_reset_wait_s', 30.0))
        self.step_wait_s = float(env_config.get('udp_step_wait_s', 10.0))
        self.inter_frame_wait_s = float(env_config.get('udp_inter_frame_wait_s', self.step_wait_s))
        self.condition_load_timeout_s = float(
            env_config.get('condition_load_timeout_s', self.inter_frame_wait_s)
        )
        self.pause_threshold_s = float(env_config.get('udp_pause_threshold_s', 1.0))
        self.require_frame_id = bool(env_config.get('udp_require_frame_id', False))
        self.udp_protocol_version = str(
            env_config.get('udp_protocol_version', LEGACY_UDP_PROTOCOL_VERSION)
        ).strip()
        if self.udp_protocol_version not in {LEGACY_UDP_PROTOCOL_VERSION, *QUEUE_UDP_PROTOCOL_VERSIONS}:
            raise ValueError(
                'env.udp_protocol_version must be '
                f'{LEGACY_UDP_PROTOCOL_VERSION!r}, {QUEUE_UDP_PROTOCOL_VERSION_V1!r}, '
                f'or {QUEUE_UDP_PROTOCOL_VERSION!r}'
            )
        configured_queue_lifecycle = bool(env_config.get('queue_lifecycle_enabled', False))
        self.queue_lifecycle_enabled = (
            configured_queue_lifecycle or self.udp_protocol_version in QUEUE_UDP_PROTOCOL_VERSIONS
        )
        self.require_condition_metadata = bool(
            env_config.get('require_condition_metadata', self.queue_lifecycle_enabled)
        )
        if self.queue_lifecycle_enabled:
            self.require_condition_metadata = True
        self.recv_buffer_size = int(env_config.get('udp_recv_buffer_size', 8192))
        self.strict_field_count = bool(env_config.get('udp_strict_field_count', True))
        self.timeout_penalty = float(env_config.get('udp_timeout_penalty', -10.0))
        if self.inter_frame_wait_s <= 0.0:
            raise ValueError('env.udp_inter_frame_wait_s must be positive')
        if self.condition_load_timeout_s <= 0.0:
            raise ValueError('env.condition_load_timeout_s must be positive')
        if self.pause_threshold_s < 0.0:
            raise ValueError('env.udp_pause_threshold_s must be non-negative')

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.udp_host, self.udp_port))
        self._socket.settimeout(self.udp_timeout_s)

        self._remote_addr: tuple[str, int] | None = None
        self._state = np.zeros(STATE_DIM, dtype=np.float32)
        self._last_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._prev_sent_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._last_sent_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._time_index = 0
        self._last_recv_frame_id: int | None = None
        self._last_sent_frame_id: int | None = None
        self._pending_packet: UdpPacket | None = None
        self._pending_action_sent = False

    def _optional_array(self, values: list[float] | None, dim: int) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (dim,):
            raise ValueError(f'Expected shape {(dim,)}, got {array.shape}')
        return array

    @staticmethod
    def _parse_nonnegative_int(value: str, field_name: str) -> int:
        try:
            parsed_float = float(value)
        except ValueError as exc:
            raise ValueError(f'Invalid {field_name}: {value!r}') from exc
        if not math.isfinite(parsed_float) or parsed_float < 0.0 or not parsed_float.is_integer():
            raise ValueError(f'Invalid {field_name}: {value!r}')
        return int(parsed_float)

    @staticmethod
    def _parse_phase(value: str) -> str:
        phase = value.strip().upper()
        if phase not in {'SETTLE', 'TRAIN', 'END'}:
            raise ValueError(f'Invalid queue phase: {value!r}')
        return phase

    @staticmethod
    def _packet_stream_key(packet: UdpPacket) -> tuple[int, int] | None:
        if packet.queue_epoch is None or packet.condition_instance_id is None:
            return None
        return packet.queue_epoch, packet.condition_instance_id

    @classmethod
    def _packet_condition_key(cls, packet: UdpPacket) -> tuple[int, int, int] | None:
        stream_key = cls._packet_stream_key(packet)
        if stream_key is None or packet.condition_id is None:
            return None
        return stream_key[0], stream_key[1], packet.condition_id

    @staticmethod
    def _packet_is_train(packet: UdpPacket) -> bool:
        return packet.phase == 'TRAIN'

    @classmethod
    def _is_stale_packet(cls, candidate: UdpPacket, reference: UdpPacket) -> bool:
        candidate_stream = cls._packet_stream_key(candidate)
        reference_stream = cls._packet_stream_key(reference)
        if candidate_stream is not None and reference_stream is not None:
            if candidate_stream != reference_stream:
                return candidate_stream < reference_stream
            if candidate.frame_id is None or reference.frame_id is None:
                return False
            return candidate.frame_id <= reference.frame_id
        if reference_stream is not None and candidate_stream is None:
            return True
        if candidate_stream is not None and reference_stream is None:
            return False
        if candidate.frame_id is None or reference.frame_id is None:
            return False
        return candidate.frame_id <= reference.frame_id

    @classmethod
    def _is_condition_boundary(cls, current_packet: UdpPacket, next_packet: UdpPacket) -> bool:
        current_condition = cls._packet_condition_key(current_packet)
        next_condition = cls._packet_condition_key(next_packet)
        if current_condition is None and next_condition is None:
            return False
        if current_condition != next_condition:
            return True
        return current_packet.phase != next_packet.phase

    @staticmethod
    def _packet_info(packet: UdpPacket, prefix: str) -> dict[str, float]:
        return {
            f'{prefix}_queue_epoch': float(packet.queue_epoch if packet.queue_epoch is not None else -1),
            f'{prefix}_condition_instance_id': float(
                packet.condition_instance_id if packet.condition_instance_id is not None else -1
            ),
            f'{prefix}_condition_id': float(packet.condition_id if packet.condition_id is not None else -1),
            f'{prefix}_phase_code': QUEUE_PHASE_CODES.get(packet.phase, 0.0),
            f'{prefix}_phase_elapsed_s': float(packet.phase_elapsed_s if packet.phase_elapsed_s is not None else -1.0),
            f'{prefix}_queue_protocol_v2': float(
                packet.queue_protocol_version == QUEUE_UDP_PROTOCOL_VERSION
            ),
        }

    def _parse_payload(
        self,
        raw_text: str,
        *,
        include_queue_metadata: bool = False,
        include_raw_validity: bool = False,
    ) -> tuple[Any, ...]:
        parts = [cell.strip() for cell in raw_text.strip().split(',')]
        expected = len(BLADED_CSV_HEADER)

        frame_id: int | None = None
        sim_time: float | None = None
        queue_epoch: int | None = None
        condition_instance_id: int | None = None
        condition_id: int | None = None
        phase: str | None = None
        phase_elapsed_s: float | None = None
        queue_protocol_version: str | None = None
        data_fields = parts
        if len(parts) == expected + 8:
            queue_protocol_version = parts[0]
            if queue_protocol_version != QUEUE_UDP_PROTOCOL_VERSION:
                raise ValueError(f'Unsupported queue protocol envelope: {queue_protocol_version!r}')
            queue_epoch = self._parse_nonnegative_int(parts[1], 'queue_epoch')
            condition_instance_id = self._parse_nonnegative_int(parts[2], 'condition_instance_id')
            condition_id = self._parse_nonnegative_int(parts[3], 'condition_id')
            phase = self._parse_phase(parts[4])
            phase_elapsed_s = float(parts[5])
            if not math.isfinite(phase_elapsed_s) or phase_elapsed_s < 0.0:
                raise ValueError(f'Invalid phase_elapsed_s: {parts[5]!r}')
            frame_id = self._parse_nonnegative_int(parts[6], 'frame_id')
            sim_time = float(parts[7])
            if not math.isfinite(sim_time):
                raise ValueError(f'Invalid sim_time: {parts[7]!r}')
            data_fields = parts[8:]
        elif len(parts) == expected + 7:
            queue_protocol_version = QUEUE_UDP_PROTOCOL_VERSION_V1
            queue_epoch = self._parse_nonnegative_int(parts[0], 'queue_epoch')
            condition_instance_id = self._parse_nonnegative_int(parts[1], 'condition_instance_id')
            condition_id = self._parse_nonnegative_int(parts[2], 'condition_id')
            phase = self._parse_phase(parts[3])
            phase_elapsed_s = float(parts[4])
            if not math.isfinite(phase_elapsed_s) or phase_elapsed_s < 0.0:
                raise ValueError(f'Invalid phase_elapsed_s: {parts[4]!r}')
            frame_id = self._parse_nonnegative_int(parts[5], 'frame_id')
            sim_time = float(parts[6])
            if not math.isfinite(sim_time):
                raise ValueError(f'Invalid sim_time: {parts[6]!r}')
            data_fields = parts[7:]
        elif len(parts) == expected + 2:
            frame_id = self._parse_nonnegative_int(parts[0], 'frame_id')
            sim_time = float(parts[1])
            if not math.isfinite(sim_time):
                raise ValueError(f'Invalid sim_time: {parts[1]!r}')
            data_fields = parts[2:]

        configured_protocol_version = getattr(self, 'udp_protocol_version', LEGACY_UDP_PROTOCOL_VERSION)
        if configured_protocol_version in QUEUE_UDP_PROTOCOL_VERSIONS:
            if queue_protocol_version != configured_protocol_version:
                if queue_protocol_version is None:
                    raise UdpProtocolVersionMismatch(
                        f'Queue protocol mismatch: expected {configured_protocol_version!r}, got legacy frame. '
                        'The Windows DLL sent frame_id,sim_time,payload without queue metadata. '
                        'Rebuild/deploy controller_old.cpp as controller.cpp, set '
                        'BLADED_RL_UDP_PROTOCOL=bladed_udp_19_si_my_queue_v2 as a User environment variable, '
                        'then restart Explorer or sign in again before manually launching Bladed.'
                    )
                raise UdpProtocolVersionMismatch(
                    f'Queue protocol mismatch: expected {configured_protocol_version!r}, '
                    f'got {queue_protocol_version!r}'
                )
        elif queue_protocol_version is not None:
            raise UdpProtocolVersionMismatch(
                f'UDP protocol mismatch: configured {configured_protocol_version!r}, '
                f'got queue frame {queue_protocol_version!r}'
            )

        if self.strict_field_count and len(data_fields) != expected:
            raise ValueError(f'Invalid field count: got {len(data_fields)}, expected {expected}')
        if getattr(self, 'require_condition_metadata', False) and queue_epoch is None:
            raise ValueError('Queue condition metadata is required by env.require_condition_metadata')
        missing_fields = tuple(BLADED_CSV_HEADER[len(data_fields):])
        if len(data_fields) < expected:
            data_fields.extend(['0'] * (expected - len(data_fields)))
        if len(data_fields) > expected:
            data_fields = data_fields[:expected]

        raw_payload: dict[str, float | None] = {}
        for idx, key in enumerate(BLADED_CSV_HEADER):
            raw_payload[key] = finite_float_or_none(data_fields[idx])
        invalid_fields = tuple(dict.fromkeys((*missing_fields,
            *(key for key, value in raw_payload.items() if value is None))))
        synchronize_power_fields(raw_payload)
        payload = {key: 0.0 if value is None else float(value) for key, value in raw_payload.items()}
        if sim_time is None:
            sim_time = float(payload.get('Time', 0.0))
        if include_queue_metadata:
            parsed = (
                payload,
                frame_id,
                sim_time,
                queue_epoch,
                condition_instance_id,
                condition_id,
                phase,
                phase_elapsed_s,
                queue_protocol_version,
            )
        else:
            parsed = (payload, frame_id, sim_time)
        return (*parsed, invalid_fields) if include_raw_validity else parsed

    def _recv_payload(
        self,
        stage: str,
        max_wait_s: float,
        after_packet: UdpPacket | None = None,
        after_frame_id: int | None = None,
    ) -> UdpPacket:
        if after_packet is not None and after_frame_id is not None:
            raise ValueError('Specify after_packet or after_frame_id, not both')
        if after_packet is None and after_frame_id is not None:
            after_packet = UdpPacket(
                payload={},
                frame_id=int(after_frame_id),
                sim_time=None,
                recv_monotonic_s=0.0,
                dropped_stale_frames=0,
            )
        deadline = time.monotonic() + max_wait_s
        last_error: Exception | None = None
        dropped_stale_frames = 0
        while time.monotonic() < deadline:
            try:
                data, addr = self._socket.recvfrom(self.recv_buffer_size)
            except socket.timeout:
                continue
            except OSError as exc:
                raise RuntimeError(f'UDP receive failed during {stage}: {exc}') from exc
            received_monotonic_s = time.monotonic()

            text = data.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
            try:
                (
                    payload,
                    frame_id,
                    sim_time,
                    queue_epoch,
                    condition_instance_id,
                    condition_id,
                    phase,
                    phase_elapsed_s,
                    queue_protocol_version,
                    invalid_payload_fields,
                ) = self._parse_payload(text, include_queue_metadata=True, include_raw_validity=True)
            except UdpProtocolVersionMismatch as exc:
                raise RuntimeError(str(exc)) from exc
            except ValueError as exc:
                last_error = exc
                continue
            if self.require_frame_id and frame_id is None:
                last_error = ValueError('Frame ID is required by env.udp_require_frame_id')
                continue
            latest_packet = UdpPacket(
                payload=payload,
                frame_id=frame_id,
                sim_time=sim_time,
                recv_monotonic_s=received_monotonic_s,
                dropped_stale_frames=dropped_stale_frames,
                queue_epoch=queue_epoch,
                condition_instance_id=condition_instance_id,
                condition_id=condition_id,
                phase=phase,
                phase_elapsed_s=phase_elapsed_s,
                queue_protocol_version=queue_protocol_version,
                invalid_payload_fields=invalid_payload_fields,
            )
            if after_packet is not None and self._is_stale_packet(latest_packet, after_packet):
                dropped_stale_frames += 1
                last_error = ValueError(f'Stale packet during {stage}')
                continue
            latest_addr = addr

            # Drain queued UDP packets and keep only the newest frame to avoid chasing stale history.
            while True:
                readable, _, _ = select.select([self._socket], [], [], 0.0)
                if not readable:
                    break
                try:
                    queued_data, queued_addr = self._socket.recvfrom(self.recv_buffer_size)
                except (socket.timeout, OSError):
                    break
                queued_received_monotonic_s = time.monotonic()
                queued_text = queued_data.decode('utf-8', errors='ignore').strip()
                if not queued_text:
                    continue
                try:
                    (
                        queued_payload,
                        queued_frame_id,
                        queued_sim_time,
                        queued_queue_epoch,
                        queued_condition_instance_id,
                        queued_condition_id,
                        queued_phase,
                        queued_phase_elapsed_s,
                        queued_queue_protocol_version,
                        queued_invalid_payload_fields,
                    ) = self._parse_payload(queued_text, include_queue_metadata=True, include_raw_validity=True)
                except UdpProtocolVersionMismatch as exc:
                    raise RuntimeError(str(exc)) from exc
                except ValueError:
                    continue
                if self.require_frame_id and queued_frame_id is None:
                    continue
                queued_packet = UdpPacket(
                    payload=queued_payload,
                    frame_id=queued_frame_id,
                    sim_time=queued_sim_time,
                    recv_monotonic_s=queued_received_monotonic_s,
                    dropped_stale_frames=dropped_stale_frames,
                    queue_epoch=queued_queue_epoch,
                    condition_instance_id=queued_condition_instance_id,
                    condition_id=queued_condition_id,
                    phase=queued_phase,
                    phase_elapsed_s=queued_phase_elapsed_s,
                    queue_protocol_version=queued_queue_protocol_version,
                    invalid_payload_fields=queued_invalid_payload_fields,
                )
                if after_packet is not None and self._is_stale_packet(queued_packet, after_packet):
                    dropped_stale_frames += 1
                    continue
                if self._is_stale_packet(queued_packet, latest_packet):
                    dropped_stale_frames += 1
                    continue

                latest_packet = queued_packet
                latest_packet.dropped_stale_frames = dropped_stale_frames
                latest_addr = queued_addr

            self._remote_addr = latest_addr
            return latest_packet

        if last_error is not None:
            raise TimeoutError(f'Timeout waiting UDP payload during {stage}; last parse error: {last_error}')
        raise TimeoutError(f'Timeout waiting UDP payload during {stage}; no valid packet received')

    def _send_action(
        self,
        action_real: np.ndarray,
        packet: UdpPacket | None = None,
        *,
        frame_id: int | None = None,
    ) -> float | None:
        if packet is None:
            packet = UdpPacket(
                payload={},
                frame_id=frame_id,
                sim_time=None,
                recv_monotonic_s=0.0,
                dropped_stale_frames=0,
            )
        elif frame_id is not None:
            raise ValueError('Specify packet or frame_id, not both')
        action_transform = getattr(self, 'action_transform', None)
        if action_transform is None:
            wire_torque, wire_pitch = float(action_real[0]), float(action_real[1])
        else:
            wire_torque, wire_pitch = action_transform.serialize_for_wire(action_real)
        if self._remote_addr is None:
            return None
        if packet.frame_id is None:
            message = f'{wire_torque:.6f},{wire_pitch:.6f}'
        elif self._packet_stream_key(packet) is not None:
            message = (
                f'{packet.queue_epoch},{packet.condition_instance_id},{int(packet.frame_id)},'
                f'{wire_torque:.6f},{wire_pitch:.6f}'
            )
        else:
            message = f'{int(packet.frame_id)},{wire_torque:.6f},{wire_pitch:.6f}'
        if packet.frame_id is not None:
            self._last_sent_frame_id = int(packet.frame_id)
        self._socket.sendto(message.encode('utf-8'), self._remote_addr)
        sent_monotonic_s = time.monotonic()
        # Read-only provenance for diagnostics. The UDP bytes and control
        # action/state remain unchanged; record the actual six-decimal wire.
        self._last_successful_action_send = {
            'action_real': np.asarray([float(f'{wire_torque:.6f}'), float(f'{wire_pitch:.6f}')], dtype=np.float64),
            'frame_id': int(packet.frame_id) if packet.frame_id is not None else -1,
            'condition_id': packet.condition_id,
            'condition_instance_id': packet.condition_instance_id,
            'queue_epoch': packet.queue_epoch,
            'invalid_payload_fields': packet.invalid_payload_fields,
        }
        return sent_monotonic_s

    def _send_queue_command(self, command: str, packet: UdpPacket) -> float:
        if self.udp_protocol_version != QUEUE_UDP_PROTOCOL_VERSION:
            raise RuntimeError(
                f'Queue command {command} requires {QUEUE_UDP_PROTOCOL_VERSION}, '
                f'not {self.udp_protocol_version}'
            )
        if packet.queue_protocol_version != QUEUE_UDP_PROTOCOL_VERSION:
            raise RuntimeError('Queue command requires a v2 queue frame')
        if self._packet_stream_key(packet) is None or packet.frame_id is None:
            raise RuntimeError('Queue command requires epoch, condition instance, and frame identity')
        if self._remote_addr is None:
            raise RuntimeError('No UDP peer is available to receive the queue command')
        message = (
            f'{packet.queue_epoch},{packet.condition_instance_id},{int(packet.frame_id)},{command}'
        )
        self._last_sent_frame_id = int(packet.frame_id)
        self._socket.sendto(message.encode('ascii'), self._remote_addr)
        return time.monotonic()

    def _send_pending_queue_command_and_receive(
        self,
        command: str,
        *,
        stage: str,
        require_same_stream: bool,
        require_new_stream: bool,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self._pending_packet is None:
            raise RuntimeError(f'Queue command {command} requires a pending frame')
        if self._pending_action_sent:
            raise RuntimeError(f'Queue command {command} cannot reuse an acknowledged pending frame')
        current_packet = self._pending_packet
        current_stream = self._packet_stream_key(current_packet)
        if current_stream is None:
            raise RuntimeError(f'Queue command {command} requires a queue stream identity')
        self._send_queue_command(command, current_packet)
        self._pending_action_sent = True
        packet = self._recv_payload(
            stage=stage,
            max_wait_s=self.condition_load_timeout_s,
            after_packet=current_packet,
        )
        packet_stream = self._packet_stream_key(packet)
        if packet_stream is None:
            raise RuntimeError(f'Queue command {command} received a frame without queue identity')
        if require_same_stream and packet_stream != current_stream:
            raise RuntimeError(f'Queue command {command} unexpectedly changed queue stream')
        if require_new_stream and packet_stream == current_stream:
            raise RuntimeError(f'Queue command {command} did not advance queue stream')
        self._pending_packet = packet
        self._pending_action_sent = False
        self._last_recv_frame_id = packet.frame_id
        return self._activate_packet_as_condition(packet)

    def probe_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        return self._send_pending_queue_command_and_receive(
            QUEUE_PROBE_COMMAND,
            stage='queue_physical_case_probe',
            require_same_stream=True,
            require_new_stream=False,
        )

    def advance_pending_incomplete_slot(self) -> tuple[np.ndarray, dict[str, float]]:
        return self._send_pending_queue_command_and_receive(
            QUEUE_ADVANCE_INCOMPLETE_SLOT_COMMAND,
            stage='queue_verified_physical_skip',
            require_same_stream=False,
            require_new_stream=True,
        )

    def _state_from_payload(self, payload: dict[str, float]) -> np.ndarray:
        return np.asarray([payload.get(name, 0.0) for name in STATE_VARIABLES], dtype=np.float32)

    def _obs_from_payload(
        self,
        payload: dict[str, float],
        previous_action_real: np.ndarray | None = None,
    ) -> np.ndarray:
        observations = build_observation_from_mapping(payload)
        if not self.include_previous_action_observation:
            return observations
        if previous_action_real is None:
            previous_action_real = self._last_sent_action
        control_observation = append_previous_action_observation(
            observations,
            previous_action_real,
            self.observation_action_low,
            self.observation_action_high,
        )
        if not self.include_operating_state_observation:
            return control_observation
        assert self.state_action_artifact is not None
        if self._state_classifier is None:
            self._state_classifier = self.state_action_artifact.classifier()
        try:
            classification = self._state_classifier.classify(OperatingPoint(
                timestamp_s=float(payload['Time']),
                wind_speed_mps=float(payload['I_AverageWindSpeed']),
                rotor_speed_rad_s=float(payload['I_RotorSpeed']),
                real_power_w=float(payload['I_RealPower']),
                pitch_rad=float(payload['I_PitchPosition1']),
                previous_torque_nm=float(np.asarray(previous_action_real, dtype=np.float64)[0]),
                previous_pitch_rad=float(np.asarray(previous_action_real, dtype=np.float64)[1]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('State-aware observations require finite current Time/control payload values') from exc
        self._last_state_classification = classification
        return append_state_regime_observation(control_observation, classification)

    def _state_observation_metrics(self) -> dict[str, float]:
        classification = self._last_state_classification
        if classification is None:
            return {
                'control_state_observation_enabled': 0.0,
                'control_optimization_state_code': -1.0,
                'control_region_code': -1.0,
                'control_filtered_wind_speed_mps': 0.0,
                'control_operating_state_changed': 0.0,
                'control_region_changed': 0.0,
            }
        return {
            'control_state_observation_enabled': 1.0,
            'control_optimization_state_code': float(
                OPTIMIZATION_STATE_ENCODING_ORDER.index(classification.optimization_state)
            ),
            'control_region_code': float(CONTROL_REGION_ENCODING_ORDER.index(classification.control_region)),
            'control_filtered_wind_speed_mps': float(classification.filtered_wind_speed_mps),
            'control_operating_state_changed': float(classification.operating_state_changed),
            'control_region_changed': float(classification.control_region_changed),
        }

    def _reward_metrics_from_payload(self, payload: dict[str, float]) -> dict[str, float]:
        return {name: float(payload.get(name, 0.0)) for name in REWARD_VARIABLES}

    def _sanitize_info(self, info: dict[str, float]) -> dict[str, float]:
        sanitized: dict[str, float] = {}
        for key, value in info.items():
            numeric = float(np.nan_to_num(value, nan=0.0, posinf=self.error_clip, neginf=-self.error_clip))
            if np.isfinite(numeric):
                sanitized[key] = numeric
        return sanitized

    def _default_action_from_payload(self, payload: dict[str, float]) -> np.ndarray:
        default_action = np.asarray([
            payload.get(ACTION_VARIABLES[0], payload.get('I_ElectricalTorque', 0.0)),
            payload.get(ACTION_VARIABLES[1], payload.get('I_PitchPosition1', 0.0)),
        ], dtype=np.float64)
        return self.action_transform.project(default_action, apply_rate_limit=False).action_real

    def _activate_packet_as_condition(self, packet: UdpPacket) -> tuple[np.ndarray, dict[str, float]]:
        payload = packet.payload
        self._state = self._state_from_payload(payload)
        self._time_index = 0
        self.reward_fn.reset_condition(packet.condition_id)
        self._state_classifier = (
            None if self.state_action_artifact is None else self.state_action_artifact.classifier()
        )
        self._last_state_classification = None

        default_action = self._default_action_from_payload(payload)
        self._prev_sent_action = default_action.copy()
        self._last_sent_action = default_action.copy()
        self._last_obs = self._obs_from_payload(payload, previous_action_real=default_action)
        info = {
            'wind_source_mode': 2.0,
            'initial_wind_speed': float(payload.get('I_AverageWindSpeed', 0.0)),
            'udp_connected': 1.0,
            'udp_frame_id': float(packet.frame_id if packet.frame_id is not None else -1),
            'udp_sim_time': float(packet.sim_time if packet.sim_time is not None else 0.0),
            'udp_dropped_stale_frames': float(packet.dropped_stale_frames),
            'condition_boundary': 0.0,
            'condition_start': float(self._packet_is_train(packet)),
            'condition_end': float(packet.phase == 'END'),
            'condition_train_active': float(self._packet_is_train(packet) or not self.queue_lifecycle_enabled),
        }
        info.update(self._packet_info(packet, 'udp'))
        info.update(self._state_observation_metrics())
        return self._last_obs.copy(), self._sanitize_info(info)

    def start_pending_condition(self) -> tuple[np.ndarray, dict[str, float]]:
        """Activate a successor packet, completing a pending END handshake when necessary."""
        if self._pending_packet is None:
            raise RuntimeError('No pending packet is available to start a condition')
        if self._pending_action_sent:
            raise RuntimeError('The pending packet already has an action; it cannot start a new condition')
        if self.queue_lifecycle_enabled and self._pending_packet.phase == 'END':
            return self._complete_pending_end_and_receive_successor()
        return self._activate_packet_as_condition(self._pending_packet)

    def _complete_pending_end_and_receive_successor(self) -> tuple[np.ndarray, dict[str, float]]:
        if self._pending_packet is None or self._pending_packet.phase != 'END':
            raise RuntimeError('Queue END completion requires a pending END packet')
        completed_stream = self._packet_stream_key(self._pending_packet)
        if completed_stream is None:
            raise RuntimeError('Queue END completion requires queue stream identity')

        deadline = time.monotonic() + self.condition_load_timeout_s
        packet = self._pending_packet
        while True:
            self._send_action(self._last_sent_action, packet)
            self._pending_action_sent = True
            remaining_wait_s = deadline - time.monotonic()
            if remaining_wait_s <= 0.0:
                raise TimeoutError('Queue condition load timed out while waiting for a successor instance after END')
            packet = self._recv_payload(
                stage='condition_load',
                max_wait_s=remaining_wait_s,
                after_packet=packet,
            )
            packet_stream = self._packet_stream_key(packet)
            if packet_stream is None:
                raise RuntimeError('Queue condition load received a frame without queue stream identity')
            self._pending_packet = packet
            self._pending_action_sent = False
            self._last_recv_frame_id = packet.frame_id
            if packet_stream != completed_stream:
                return self._activate_packet_as_condition(packet)
            if packet.phase != 'END':
                raise RuntimeError(
                    'Queue condition load received a non-END frame from the completed condition stream'
                )

    def recover_after_train_timeout(self) -> tuple[np.ndarray, dict[str, float]]:
        """Wait for a resumed TRAIN frame or the next queue instance after a TRAIN timeout."""
        if not self.queue_lifecycle_enabled:
            raise RuntimeError('Queue timeout recovery requires queue_lifecycle_enabled=true')
        if self._pending_packet is None:
            raise RuntimeError('Queue timeout recovery requires the timed-out packet')
        failed_packet = self._pending_packet
        failed_stream = self._packet_stream_key(failed_packet)
        if failed_stream is None:
            raise RuntimeError('Queue timeout recovery requires queue epoch and condition instance metadata')

        recovery_wait_start = time.monotonic()
        packet = self._recv_payload(
            stage='condition_recovery',
            max_wait_s=self.condition_load_timeout_s,
            after_packet=failed_packet,
        )
        recovered_stream = self._packet_stream_key(packet)
        if recovered_stream is None:
            raise RuntimeError('Queue timeout recovery received a packet without queue identity metadata')
        condition_changed = recovered_stream != failed_stream

        self._pending_packet = packet
        self._pending_action_sent = False
        self._last_recv_frame_id = packet.frame_id
        if condition_changed:
            obs, info = self._activate_packet_as_condition(packet)
        else:
            if packet.phase != 'TRAIN':
                raise RuntimeError(
                    'Queue timeout recovery received the same condition instance outside TRAIN; '
                    'expected a resumed TRAIN frame or a new queue instance'
                )
            self._state = self._state_from_payload(packet.payload)
            self._last_obs = self._obs_from_payload(packet.payload)
            obs = self._last_obs.copy()
            info = {
                'wind_source_mode': 2.0,
                'udp_connected': 1.0,
                'udp_frame_id': float(packet.frame_id if packet.frame_id is not None else -1),
                'udp_sim_time': float(packet.sim_time if packet.sim_time is not None else 0.0),
                'udp_dropped_stale_frames': float(packet.dropped_stale_frames),
                'condition_boundary': 0.0,
                'condition_start': 0.0,
                'condition_end': 0.0,
                'condition_train_active': 1.0,
            }
            info.update(self._packet_info(packet, 'udp'))

        info.update(
            {
                'udp_timeout_recovery_active': 1.0,
                'udp_timeout_recovery_wait_ms': float((time.monotonic() - recovery_wait_start) * 1000.0),
                'udp_timeout_recovery_condition_changed': float(condition_changed),
                'udp_timeout_recovery_from_queue_epoch': float(failed_stream[0]),
                'udp_timeout_recovery_from_condition_instance_id': float(failed_stream[1]),
            }
        )
        return obs, self._sanitize_info(info)

    def reset(self) -> tuple[np.ndarray, dict[str, float]]:
        after_packet = self._pending_packet
        if self._pending_packet is not None and not self._pending_action_sent:
            # Complete the previous frame handshake before waiting for the next episode's observation.
            self._send_action(self._last_sent_action, self._pending_packet)
            self._pending_action_sent = True
        packet = self._recv_payload(
            stage='reset',
            max_wait_s=self.reset_wait_s,
            after_packet=after_packet,
        )
        self._pending_packet = packet
        self._pending_action_sent = False
        self._last_recv_frame_id = packet.frame_id
        return self._activate_packet_as_condition(packet)

    def step(
        self,
        action_norm: np.ndarray,
        on_action_sent: Callable[[float], None] | None = None,
        *,
        action_real: np.ndarray | None = None,
    ) -> StepResult:
        if self._pending_packet is None:
            raise RuntimeError('Environment step called before reset completed')
        if self._pending_action_sent:
            raise RuntimeError('Environment step called before the pending frame handshake completed')
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(ACTION_DIM)
        action_norm = np.nan_to_num(action_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        action_norm = np.clip(action_norm, -1.0, 1.0)

        current_packet = self._pending_packet
        current_payload = current_packet.payload
        current_frame_id = current_packet.frame_id
        action_ready_time = time.monotonic()

        requested_action = (
            self.action_scaler.denormalize(action_norm)
            if action_real is None
            else action_array_float64(action_real, 'action_real')
        )
        next_action = self.action_transform.project(
            requested_action,
            previous_action_real=self._last_sent_action,
            apply_rate_limit=self.action_limiter_owner == 'env',
        ).action_real
        action_sent_monotonic_s = self._send_action(next_action, current_packet)
        if action_sent_monotonic_s is None:
            raise RuntimeError('No UDP peer is available to receive the action')
        self._pending_action_sent = True
        action_apply_lag_ms = (action_sent_monotonic_s - current_packet.recv_monotonic_s) * 1000.0
        if on_action_sent is not None:
            on_action_sent(action_sent_monotonic_s)
        udp_receive_wait_start = time.monotonic()
        condition_load_active = bool(self.queue_lifecycle_enabled and current_packet.phase == 'END')
        receive_wait_limit_s = self.condition_load_timeout_s if condition_load_active else self.inter_frame_wait_s

        try:
            packet = self._recv_payload(
                stage='condition_load' if condition_load_active else 'step',
                max_wait_s=receive_wait_limit_s,
                after_packet=current_packet,
            )
        except TimeoutError:
            receive_wait_ms = float((time.monotonic() - udp_receive_wait_start) * 1000.0)
            info = self._sanitize_info(
                {
                    'env_invalid_prediction': 1.0,
                    'env_invalid_reward': 1.0,
                    'env_timeout': 1.0,
                    'env_input_clip_fraction': 0.0,
                    'env_output_clip_fraction': 0.0,
                    'env_model_clip_event': 0.0,
                    'env_model_clip_count': 0.0,
                    'udp_frame_id': -1.0,
                    'udp_sim_time': 0.0,
                    'udp_dropped_stale_frames': 0.0,
                    'udp_receive_monotonic_s': float(current_packet.recv_monotonic_s),
                    'udp_action_sent_monotonic_s': float(action_sent_monotonic_s),
                    'udp_frame_lag': 0.0,
                    'udp_action_apply_lag_ms': 0.0,
                    'udp_receive_wait_ms': receive_wait_ms,
                    'udp_handshake_complete': 0.0,
                    'udp_inter_frame_timeout': 1.0,
                    'udp_condition_load_active': float(condition_load_active),
                    'udp_condition_load_timeout': float(condition_load_active),
                    'udp_wait_limit_s': float(receive_wait_limit_s),
                    'udp_pause_active': 1.0,
                    'udp_pause_wait_ms': receive_wait_ms,
                }
            )
            return StepResult(obs=self._last_obs.copy(), reward=self.timeout_penalty, done=True, info=info)

        payload = packet.payload
        frame_id = packet.frame_id
        self._last_recv_frame_id = frame_id
        self._pending_packet = packet
        self._pending_action_sent = False
        receive_wait_ms = float((time.monotonic() - udp_receive_wait_start) * 1000.0)
        pause_threshold_ms = self.pause_threshold_s * 1000.0

        state_next = self._state_from_payload(payload)
        obs_next = self._obs_from_payload(payload, previous_action_real=next_action)
        reward_state_weights = (
            None
            if self.state_action_artifact is None or self._last_state_classification is None
            else self.state_action_artifact.reward_state_weights(self._last_state_classification)
        )
        reward_metrics = self._reward_metrics_from_payload(payload)
        wind_t = float(payload.get('I_AverageWindSpeed', 0.0))
        condition_boundary = self._is_condition_boundary(current_packet, packet)
        current_train = self._packet_is_train(current_packet) or not self.queue_lifecycle_enabled
        next_train = self._packet_is_train(packet) or not self.queue_lifecycle_enabled
        transition_store_eligible = current_train and next_train and not condition_boundary
        if transition_store_eligible:
            current_power = (
                float(self._state[self.reward_fn.real_power_index])
                if self.reward_fn.power_source == 'state_next_real_power'
                else float(current_payload.get('Electrical power', 0.0))
            )
            reward, reward_breakdown = self.reward_fn.compute(
                state_t=self._state,
                state_next=state_next,
                reward_metrics=reward_metrics,
                action_t=next_action,
                prev_action_t=self._prev_sent_action,
                wind_t=wind_t,
                condition_id=current_packet.condition_id,
                reward_state_weights=reward_state_weights,
                power_t=current_power,
                sim_time_t=current_packet.sim_time,
                sim_time_next=packet.sim_time,
                power_reference_raw=self.reward_fn.reference_power(packet.condition_id, packet.sim_time),
            )
        else:
            reward = 0.0
            reward_breakdown = {
                'reward_power_raw': 0.0,
                'reward_power_metric_raw': 0.0,
                'reward_power_state_raw': 0.0,
                'reward_power_t_raw': 0.0,
                'reward_power_term': 0.0,
                'reward_energy_delta_t_s': 0.0,
                'reward_energy_increment_j': 0.0,
                'reward_energy_increment_valid': 0.0,
                'reward_pid_power_reference_raw': 0.0,
                'reward_relative_power_raw': 0.0,
                'reward_relative_energy_increment_j': 0.0,
                'reward_blade_root_raw': 0.0,
                'reward_blade_root_ratio': 0.0,
                'reward_blade_root_term': 0.0,
                'reward_blade_root_peak_raw': 0.0,
                'reward_blade_root_peak_excess': 0.0,
                'reward_blade_root_peak_excess_increment': 0.0,
                'reward_acc_raw': 0.0,
                'reward_acc_term': 0.0,
                'reward_shaft_raw': 0.0,
                'reward_shaft_term': 0.0,
                'reward_pitch_rate_raw': 0.0,
                'reward_pitch_rate_term': 0.0,
                'reward_torque_rate_raw': 0.0,
                'reward_torque_rate_term': 0.0,
                'reward_power_available_raw': 0.0,
                'reward_power_available_fraction': 0.0,
                'reward_command_power_raw': 0.0,
                'reward_command_excess_raw': 0.0,
                'reward_command_excess_term': 0.0,
                'reward_power_tracking_raw': 0.0,
                'reward_power_tracking_term': 0.0,
                'reward_power_floor_raw': 0.0,
                'reward_power_floor_term': 0.0,
                'reward_blade_root_continuous_term': 0.0,
                'reward_total': 0.0,
            }
        transition_action_delta = next_action - self._last_sent_action

        self._prev_sent_action = self._last_sent_action.copy()
        self._last_sent_action = next_action

        self._state = state_next
        self._last_obs = obs_next
        self._time_index += 1
        done = condition_boundary if self.queue_lifecycle_enabled else self._time_index >= self.episode_length
        condition_start = condition_boundary and next_train
        condition_end = condition_boundary and (current_train or packet.phase == 'END')

        info = {
            **reward_breakdown,
            'wind_t': wind_t,
            'wind_next': wind_t,
            'power_t_metric_raw': float(current_payload.get(REWARD_VARIABLES[0], 0.0)),
            'power_metric_raw': float(reward_metrics.get(REWARD_VARIABLES[0], 0.0)),
            'action_torque': float(self._last_sent_action[0]),
            'action_pitch': float(self._last_sent_action[1]),
            'action_torque_delta': float(transition_action_delta[0]),
            'action_pitch_delta': float(transition_action_delta[1]),
            'power': float(reward_breakdown.get('reward_power_raw', 0.0)),
            'power_reward_metric': float(reward_metrics.get('Electrical power', 0.0)),
            'power_state_real': float(state_next[5]),
            'blade_root_metrics_available': float(all(name in payload for name in REWARD_VARIABLES[1:])),
            'blade_root_metric_nonzero': float(abs(float(reward_breakdown.get('reward_blade_root_raw', 0.0))) > 1e-9),
            'blade_root_metric_abs': abs(float(reward_breakdown.get('reward_blade_root_raw', 0.0))),
            'env_invalid_prediction': 0.0,
            'env_invalid_reward': 0.0,
            'env_input_clip_fraction': 0.0,
            'env_output_clip_fraction': 0.0,
            'env_model_clip_event': 0.0,
            'env_model_clip_count': 0.0,
            'udp_connected': 1.0,
            'udp_frame_time': float(payload.get('Time', 0.0)),
            'udp_frame_id': float(current_frame_id if current_frame_id is not None else -1),
            'udp_next_frame_id': float(frame_id if frame_id is not None else -1),
            'udp_sim_time': float(current_packet.sim_time if current_packet.sim_time is not None else 0.0),
            'udp_next_sim_time': float(packet.sim_time if packet.sim_time is not None else 0.0),
            'udp_receive_monotonic_s': float(current_packet.recv_monotonic_s),
            'udp_action_sent_monotonic_s': float(action_sent_monotonic_s),
            'udp_frame_lag': float(
                frame_id - current_frame_id - 1
                if frame_id is not None and current_frame_id is not None
                else 0
            ),
            'udp_dropped_stale_frames': float(packet.dropped_stale_frames),
            'udp_action_apply_lag_ms': float(action_apply_lag_ms),
            'udp_action_ready_lag_ms': float((action_ready_time - current_packet.recv_monotonic_s) * 1000.0),
            'udp_receive_wait_ms': receive_wait_ms,
            'udp_handshake_complete': 1.0,
            'udp_inter_frame_timeout': 0.0,
            'udp_condition_load_active': float(condition_load_active),
            'udp_condition_load_timeout': 0.0,
            'udp_wait_limit_s': float(receive_wait_limit_s),
            'udp_pause_active': float(receive_wait_ms >= pause_threshold_ms),
            'udp_pause_wait_ms': max(0.0, receive_wait_ms - pause_threshold_ms),
            'udp_action_next_torque': float(next_action[0]),
            'udp_action_next_pitch': float(next_action[1]),
            'udp_action_next_torque_delta': float(transition_action_delta[0]),
            'udp_action_next_pitch_delta': float(transition_action_delta[1]),
            'condition_boundary': float(condition_boundary),
            'condition_start': float(condition_start),
            'condition_end': float(condition_end),
            'condition_train_active': float(next_train),
            'transition_store_eligible': float(transition_store_eligible),
            'transition_store_reason_code': float(
                1.0 if transition_store_eligible else 2.0 if condition_boundary else 3.0
            ),
        }
        info.update(self._packet_info(current_packet, 'udp'))
        info.update(self._packet_info(packet, 'udp_next'))
        info.update(self._state_observation_metrics())
        return StepResult(obs=obs_next, reward=float(reward), done=done, info=self._sanitize_info(info))

    @property
    def obs_dim(self) -> int:
        if self.include_operating_state_observation:
            return STATE_AWARE_CONTROL_OBS_DIM
        return CONTROL_OBS_DIM if self.include_previous_action_observation else OBS_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def current_payload(self) -> dict[str, float]:
        """Return a copy of the Bladed frame used for the next control decision."""
        if self._pending_packet is None:
            raise RuntimeError('No Bladed payload is available before reset completed')
        return dict(self._pending_packet.payload)

    @property
    def current_queue_metadata(self) -> dict[str, float]:
        """Identity/phase of the actual pending parsed packet (including END)."""
        if self._pending_packet is None:
            raise RuntimeError('No Bladed payload is available before reset completed')
        return self._packet_info(self._pending_packet, 'udp')

    @property
    def current_payload_invalid_fields(self) -> tuple[str, ...]:
        """Raw invalidity before the existing control-side zero filling."""
        if self._pending_packet is None:
            raise RuntimeError('No Bladed payload is available before reset completed')
        return self._pending_packet.invalid_payload_fields

    @property
    def last_sent_action(self) -> np.ndarray:
        """Return the Bladed command active when the current condition was activated."""
        return self._last_sent_action.copy()

    @property
    def last_successful_action_send(self) -> dict:
        """Copy of the last successfully serialized wire command, not an alias."""
        snapshot = dict(self._last_successful_action_send)
        snapshot['action_real'] = snapshot['action_real'].copy()
        return snapshot

    def close(self) -> None:
        udp_socket = getattr(self, '_socket', None)
        if udp_socket is None:
            return
        try:
            udp_socket.close()
        except OSError:
            pass

    def __del__(self) -> None:
        self.close()
