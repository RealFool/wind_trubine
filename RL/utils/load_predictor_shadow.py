from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import base64
import math
import random
from typing import Mapping, Sequence
import zlib

import numpy as np
import torch

from RL.replay.load_prediction_replay import LoadPredictionFrame, LoadPredictionRecord, LoadPredictionReplay
from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
    make_zoh_candidate_plan,
)


BLADE_ROOT_MY_FIELDS = (
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
)
PHYSICAL_FEATURE_FIELDS = (
    'I_RotorSpeed',
    'I_GeneratorSpeed',
    'I_RealPower',
    'I_ElectricalTorque',
    'I_PitchPosition1',
    'I_NacelleForeAftAcceleration',
    'I_NacelleSideSideAcceleration',
    'I_AverageWindSpeed',
)
STATE_ENCODING_WIDTH = 8
FEATURE_NAMES = (
    *PHYSICAL_FEATURE_FIELDS,
    'sin_azimuth',
    'cos_azimuth',
    *BLADE_ROOT_MY_FIELDS,
    'Blade root 1 My rate (Nm/s)',
    'Blade root 2 My rate (Nm/s)',
    'Blade root 3 My rate (Nm/s)',
    'sent_action_torque',
    'sent_action_pitch',
    'projected_candidate_action_torque',
    'projected_candidate_action_pitch',
    *(f'state_encoding_{index}' for index in range(STATE_ENCODING_WIDTH)),
    'pid_peak_headroom_blade_1',
    'pid_peak_headroom_blade_2',
    'pid_peak_headroom_blade_3',
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ShadowStepResult:
    action_real: np.ndarray
    feature: np.ndarray
    history_ready: bool
    pending_sample_id: int | None
    completed_records: tuple[LoadPredictionRecord, ...]
    predictor_output: np.ndarray | None
    completed_audit_record_count: int
    completed_audit_calibration_eligible_count: int
    diagnostics: dict[str, float]


class LoadPredictorShadow:
    """Causal shadow path; it never changes the action passed to the plant."""

    def __init__(
        self,
        interface: LoadPredictorInterface,
        *,
        action_low: Sequence[float],
        action_high: Sequence[float],
        interface_sha256: str | None = None,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        replay_capacity: int = 100_000,
        record_capacity: int = 512,
        audit_capacity_per_group: int = 35,
        audit_seed: int = 42,
    ) -> None:
        self.interface = interface
        if interface.schema_version != 'causal_load_predictor_interface_v3':
            raise ValueError('Live load predictor shadow requires the candidate-conditioned v3 interface')
        self.interface_sha256 = '' if interface_sha256 is None else str(interface_sha256).lower()
        if self.interface_sha256 and (
            len(self.interface_sha256) != 64
            or any(character not in '0123456789abcdef' for character in self.interface_sha256)
        ):
            raise ValueError('interface_sha256 must be a lowercase SHA-256 digest when provided')
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        if self.action_low.shape != (2,) or self.action_high.shape != (2,) or np.any(self.action_high <= self.action_low):
            raise ValueError('shadow action bounds must contain two increasing values')
        self.model = model
        self.device = device or torch.device('cpu')
        if audit_capacity_per_group <= 0:
            raise ValueError('audit_capacity_per_group must be positive')
        self.audit_capacity_per_group = int(audit_capacity_per_group)
        self.minimum_audit_sample_spacing_s = (
            interface.history_frames - 1 + max(interface.horizon_frames)
        ) * interface.step_duration_s
        self._audit_random = random.Random(int(audit_seed))
        self._pending_prediction_metadata: dict[int, dict[str, object]] = {}
        self._audit_records_by_group: dict[tuple[int, str, float], list[dict[str, object]]] = {}
        self._audit_seen_by_group: dict[tuple[int, str, float], int] = {}
        self._audit_last_candidate_time_by_group: dict[tuple[int, str, float], float] = {}
        self._audit_records_seen = 0
        self._audit_records_retained_total = 0
        self._audit_records_dropped = 0
        self._pending_audit_dropped_count = 0
        self._audit_condition_records_seen = 0
        self._audit_condition_records_dropped = 0
        self._resolved_record_count_at_condition_start = 0
        self._discontinuity_count_at_condition_start = 0
        if self.model is not None:
            if getattr(self.model, 'input_dim', FEATURE_DIM) != FEATURE_DIM:
                raise ValueError('load predictor model input_dim does not match shadow feature schema')
            if not callable(getattr(self.model, 'predict_candidate', None)):
                raise ValueError('load predictor model must expose candidate-conditioned predict_candidate()')
            self.model.to(self.device).eval()
        self.replay = LoadPredictionReplay(
            interface,
            action_low=self.action_low,
            action_high=self.action_high,
            max_pending=replay_capacity,
            max_records=record_capacity,
        )
        self._history: deque[np.ndarray] = deque(maxlen=interface.history_frames)
        self._previous_blade_root_my: np.ndarray | None = None
        self._condition_key: tuple[int, int] | None = None

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def history_count(self) -> int:
        return len(self._history)

    @property
    def last_history(self) -> np.ndarray | None:
        if not self._history:
            return None
        return np.stack(tuple(self._history), axis=0).copy()

    def reset_condition(self, condition_id: int, condition_instance_id: int) -> dict[str, int]:
        if self._audit_records_by_group:
            dropped_rows = sum(len(rows) for rows in self._audit_records_by_group.values())
            self._audit_records_dropped += dropped_rows
            self._audit_condition_records_dropped += dropped_rows
            self._audit_records_by_group.clear()
            self._audit_seen_by_group.clear()
            self._audit_last_candidate_time_by_group.clear()
        self._pending_audit_dropped_count += len(self._pending_prediction_metadata)
        self._pending_prediction_metadata.clear()
        self._history.clear()
        self._previous_blade_root_my = None
        self._condition_key = (int(condition_id), int(condition_instance_id))
        stats = self.replay.start_condition(*self._condition_key)
        replay_state = self.replay.state_dict()
        self._resolved_record_count_at_condition_start = int(replay_state['resolved_record_count'])
        self._discontinuity_count_at_condition_start = int(replay_state['discontinuity_count'])
        self._audit_condition_records_seen = 0
        self._audit_condition_records_dropped = 0
        return stats

    def observe(
        self,
        *,
        condition_id: int,
        condition_instance_id: int,
        payload: Mapping[str, float],
        sent_action_real: Sequence[float],
        projected_candidate_action_real: Sequence[float],
        state_encoding: Sequence[float],
        pid_peak_abs_nm: Sequence[float],
        optimization_state: str | None = None,
        execution_mode: str = 'actor',
        outcome_tag: str = 'complete',
        actor_policy_version: int = 0,
        actor_policy_sha256: str = '',
        predictor_artifact_sha256: str = '',
    ) -> ShadowStepResult:
        actor_policy_sha256 = str(actor_policy_sha256).lower()
        predictor_artifact_sha256 = str(predictor_artifact_sha256).lower()
        if int(actor_policy_version) < 0:
            raise ValueError('actor_policy_version must be non-negative')
        if actor_policy_sha256 and (
            len(actor_policy_sha256) != 64
            or any(character not in '0123456789abcdef' for character in actor_policy_sha256)
        ):
            raise ValueError('actor_policy_sha256 must be a lowercase SHA-256 digest when provided')
        if predictor_artifact_sha256 and (
            len(predictor_artifact_sha256) != 64
            or any(character not in '0123456789abcdef' for character in predictor_artifact_sha256)
        ):
            raise ValueError('predictor_artifact_sha256 must be a lowercase SHA-256 digest when provided')
        key = (int(condition_id), int(condition_instance_id))
        if self._condition_key != key:
            self.reset_condition(*key)
        feature, blade_root_my = self._feature_from_payload(
            payload,
            sent_action_real=sent_action_real,
            projected_candidate_action_real=projected_candidate_action_real,
            state_encoding=state_encoding,
            pid_peak_abs_nm=pid_peak_abs_nm,
        )
        self._history.append(feature)
        frame = LoadPredictionFrame(
            condition_id=key[0],
            condition_instance_id=key[1],
            timestamp_s=float(payload['Time']),
            sent_action_real=np.asarray(sent_action_real, dtype=np.float64),
            actual_electrical_torque_nm=float(payload['I_ElectricalTorque']),
            actual_pitch_rad=float(payload['I_PitchPosition1']),
            blade_root_my_nm=blade_root_my,
            pid_peak_abs_nm=np.asarray(pid_peak_abs_nm, dtype=np.float64),
            actor_policy_version=int(actor_policy_version),
        )
        completed_records = self.replay.observe_frame(frame)
        completed_audit_record_count, completed_audit_calibration_eligible_count = (
            self._capture_completed_audit_records(completed_records)
        )
        pending_ids = set(self.replay.pending_sample_ids)
        completed_ids = {record.sample_id for record in completed_records}
        flushed_ids = [
            sample_id
            for sample_id in self._pending_prediction_metadata
            if sample_id not in pending_ids and sample_id not in completed_ids
        ]
        for sample_id in flushed_ids:
            self._pending_prediction_metadata.pop(sample_id, None)
        self._pending_audit_dropped_count += len(flushed_ids)
        history_ready = len(self._history) == self.interface.history_frames
        pending_sample_id: int | None = None
        predictor_output: np.ndarray | None = None
        predictor_output_invalid = False
        candidate = np.asarray(projected_candidate_action_real, dtype=np.float64)
        if history_ready:
            history = self.last_history
            assert history is not None
            pending_sample_id = self.replay.add_pending(
                history=history,
                candidate_action_real=np.asarray(projected_candidate_action_real, dtype=np.float64),
                plan_kind=PLAN_KIND_ZOH_CANDIDATE,
                online=True,
            )
            selected_state = str(optimization_state or 'unknown')
            prediction_metadata: dict[str, object] = {
                'condition_id': key[0],
                'condition_instance_id': key[1],
                'timestamp_s': float(payload['Time']),
                'optimization_state': selected_state,
                'execution_mode': str(execution_mode),
                'outcome_tag': str(outcome_tag),
                'actor_policy_version': int(actor_policy_version),
                'actor_policy_sha256': str(actor_policy_sha256).lower(),
                'predictor_artifact_sha256': predictor_artifact_sha256,
                'candidate_action_real': candidate.copy(),
                'history': history.copy(),
                'predictor_output': None,
            }
            if self.model is not None:
                candidate_horizons = self.interface.soft_guard_horizons_s
                max_horizon_frames = max(self.interface.horizon_frames)
                plan_batch = np.zeros(
                    (len(candidate_horizons), max_horizon_frames, 2),
                    dtype=np.float32,
                )
                model_horizon_indices: list[int] = []
                for candidate_index, horizon_s in enumerate(candidate_horizons):
                    horizon_index = self.interface.horizons_s.index(horizon_s)
                    model_horizon_indices.append(horizon_index)
                    horizon_frames = self.interface.horizon_frames[horizon_index]
                    plan = make_zoh_candidate_plan(
                        projected_candidate_action_real,
                        horizon_frames,
                    )
                    plan_batch[candidate_index, :horizon_frames] = plan.astype(np.float32)
                with torch.inference_mode():
                    predictions = self.model.predict_candidate(
                        torch.from_numpy(np.repeat(history[None, ...], len(candidate_horizons), axis=0)).to(
                            self.device
                        ),
                        torch.from_numpy(plan_batch).to(self.device),
                        torch.as_tensor(model_horizon_indices, dtype=torch.long, device=self.device),
                        torch.ones(len(candidate_horizons), dtype=torch.long, device=self.device),
                    )
                candidate_predictions = predictions.detach().cpu().numpy()
                if candidate_predictions.shape == (len(candidate_horizons), 5) and np.isfinite(
                    candidate_predictions
                ).all():
                    predictor_output = candidate_predictions
                    prediction_metadata['predictor_output'] = predictor_output.copy()
                else:
                    predictor_output_invalid = True
            self._pending_prediction_metadata[pending_sample_id] = prediction_metadata
        diagnostics = {
            'load_predictor_shadow_active': 1.0,
            'load_predictor_shadow_history_count': float(len(self._history)),
            'load_predictor_shadow_history_ready': float(history_ready),
            'load_predictor_shadow_pending_count': float(self.replay.pending_count),
            'load_predictor_shadow_completed_record_count': float(len(completed_records)),
            'load_predictor_shadow_action_changed': 0.0,
            'load_predictor_shadow_zoh_candidate_submitted': float(pending_sample_id is not None),
            'load_predictor_shadow_prediction_available': float(predictor_output is not None),
            'load_predictor_shadow_prediction_invalid': float(predictor_output_invalid),
            'load_predictor_shadow_actor_policy_version': float(actor_policy_version),
        }
        if predictor_output is not None:
            for horizon_index, horizon_s in enumerate(self.interface.soft_guard_horizons_s):
                horizon_ms = int(round(horizon_s * 1000.0))
                prediction = predictor_output[horizon_index]
                diagnostics[f'load_predictor_shadow_predicted_future_torque_nm_h{horizon_ms:03d}'] = float(
                    prediction[0]
                )
                diagnostics[f'load_predictor_shadow_predicted_future_pitch_rad_h{horizon_ms:03d}'] = float(
                    prediction[1]
                )
                for blade_index in range(3):
                    diagnostics[
                        f'load_predictor_shadow_predicted_peak_ratio_blade_{blade_index + 1}_h{horizon_ms:03d}'
                    ] = float(prediction[blade_index + 2])
        return ShadowStepResult(
            action_real=candidate.copy(),
            feature=feature.copy(),
            history_ready=history_ready,
            pending_sample_id=pending_sample_id,
            completed_records=tuple(completed_records),
            predictor_output=predictor_output,
            completed_audit_record_count=completed_audit_record_count,
            completed_audit_calibration_eligible_count=completed_audit_calibration_eligible_count,
            diagnostics=diagnostics,
        )

    def _capture_completed_audit_records(
        self,
        records: Sequence[LoadPredictionRecord],
    ) -> tuple[int, int]:
        completed = [record for record in records if record.plan_kind == 'realized_plan']
        audit_record_count = 0
        calibration_eligible_count = 0
        for record in completed:
            metadata = self._pending_prediction_metadata.get(record.sample_id)
            if metadata is None:
                continue
            output = metadata.get('predictor_output')
            online_horizon = record.horizon_s in self.interface.soft_guard_horizons_s
            prediction_index = (
                self.interface.soft_guard_horizons_s.index(record.horizon_s)
                if online_horizon
                else None
            )
            predicted = (
                output[prediction_index]
                if isinstance(output, np.ndarray) and prediction_index is not None
                else None
            )
            plan_kind = PLAN_KIND_ZOH_CANDIDATE if online_horizon else PLAN_KIND_REALIZED
            candidate_plan = (
                make_zoh_candidate_plan(metadata['candidate_action_real'], record.horizon_frames)
                if online_horizon
                else record.actual_future_action_plan
            )
            prediction_available = predicted is not None
            calibration_eligible = bool(
                prediction_available
                and online_horizon
                and record.plan_mismatch is not None
                and record.plan_mismatch <= self.interface.max_plan_mismatch
            )
            audit_record: dict[str, object] = {
                'schema_version': 'load_predictor_shadow_audit_record_v1',
                'sample_id': record.sample_id,
                'condition_id': record.condition_id,
                'condition_instance_id': record.condition_instance_id,
                'feature_dim': FEATURE_DIM,
                'predictor_interface_sha256': self.interface_sha256,
                'predictor_artifact_sha256': str(metadata['predictor_artifact_sha256']),
                'timestamp_s': float(metadata['timestamp_s']),
                'optimization_state': str(metadata['optimization_state']),
                'execution_mode': str(metadata['execution_mode']),
                'outcome_tag': str(metadata['outcome_tag']),
                'actor_policy_version': int(metadata['actor_policy_version']),
                'actor_policy_sha256': str(metadata['actor_policy_sha256']),
                'actual_future_actor_policy_versions': list(record.actual_future_actor_policy_versions),
                'actor_policy_window_frozen': bool(
                    record.actual_future_actor_policy_versions
                    and all(
                        version == int(metadata['actor_policy_version'])
                        for version in record.actual_future_actor_policy_versions
                    )
                ),
                'horizon_s': float(record.horizon_s),
                'horizon_frames': int(record.horizon_frames),
                'plan_kind': plan_kind,
                'online_authorized': bool(online_horizon),
                'prediction_available': prediction_available,
                'candidate_action_real': np.asarray(metadata['candidate_action_real']).tolist(),
                'model_input_plan': candidate_plan.tolist(),
                'history_schema_version': 'load_predictor_shadow_feature_v1',
                'history_shape': list(np.asarray(metadata['history']).shape),
                'history_dtype': '<f4',
                'history_zlib_base64': base64.b64encode(
                    zlib.compress(
                        np.asarray(metadata['history'], dtype='<f4').tobytes(order='C'),
                        level=1,
                    )
                ).decode('ascii'),
                'actual_future_action_plan': record.actual_future_action_plan.tolist(),
                'predicted_response_available': prediction_available,
                'predicted_future_torque_nm': None if predicted is None else float(predicted[0]),
                'predicted_future_pitch_rad': None if predicted is None else float(predicted[1]),
                'predicted_peak_ratio_by_blade': (
                    None if predicted is None else [float(value) for value in predicted[2:]]
                ),
                'actual_future_torque_nm': float(record.future_electrical_torque_nm),
                'actual_future_pitch_rad': float(record.future_pitch_rad),
                'actual_future_response': [
                    float(record.future_electrical_torque_nm),
                    float(record.future_pitch_rad),
                ],
                'actual_future_peak_ratio_by_blade': [
                    float(value) for value in record.future_blade_root_peak_ratio
                ],
                'plan_mismatch': record.plan_mismatch,
                'plan_match_complete': bool(
                    record.plan_mismatch is not None
                    and math.isfinite(float(record.plan_mismatch))
                ),
                'calibration_eligible': calibration_eligible,
            }
            state = str(metadata['optimization_state'])
            group = (record.condition_id, state, float(record.horizon_s))
            sample_time_s = float(metadata['timestamp_s'])
            previous_audit_time_s = self._audit_last_candidate_time_by_group.get(group)
            if (
                previous_audit_time_s is not None
                and sample_time_s - previous_audit_time_s < self.minimum_audit_sample_spacing_s - 1.0e-9
            ):
                self._audit_records_dropped += 1
                self._audit_condition_records_dropped += 1
                if record.horizon_s == max(self.interface.horizons_s):
                    self._pending_prediction_metadata.pop(record.sample_id, None)
                continue
            self._audit_last_candidate_time_by_group[group] = sample_time_s
            audit_record_count += 1
            calibration_eligible_count += int(calibration_eligible)
            bucket = self._audit_records_by_group.setdefault(group, [])
            seen = self._audit_seen_by_group.get(group, 0) + 1
            self._audit_seen_by_group[group] = seen
            self._audit_records_seen += 1
            self._audit_condition_records_seen += 1
            if len(bucket) < self.audit_capacity_per_group:
                bucket.append(audit_record)
                self._audit_records_retained_total += 1
            else:
                replacement_index = self._audit_random.randrange(seen)
                self._audit_records_dropped += 1
                self._audit_condition_records_dropped += 1
                if replacement_index < self.audit_capacity_per_group:
                    bucket[replacement_index] = audit_record
            if record.horizon_s == max(self.interface.horizons_s):
                self._pending_prediction_metadata.pop(record.sample_id, None)
        return audit_record_count, calibration_eligible_count

    def drain_audit_records(self) -> tuple[dict[str, object], ...]:
        records = [
            record
            for group in sorted(self._audit_records_by_group)
            for record in sorted(
                self._audit_records_by_group[group],
                key=lambda value: (float(value['timestamp_s']), int(value['sample_id'])),
            )
        ]
        self._audit_records_by_group.clear()
        self._audit_seen_by_group.clear()
        self._audit_last_candidate_time_by_group.clear()
        self._audit_records_retained_total = 0
        return tuple(records)

    @property
    def audit_summary(self) -> dict[str, int]:
        return {
            'audit_records_seen': self._audit_records_seen,
            'audit_records_retained': sum(len(rows) for rows in self._audit_records_by_group.values()),
            'audit_records_dropped': self._audit_records_dropped,
            'audit_capacity_per_group': self.audit_capacity_per_group,
            'audit_pending_prediction_count': len(self._pending_prediction_metadata),
            'audit_pending_prediction_dropped_count': self._pending_audit_dropped_count,
            'audit_condition_records_seen': self._audit_condition_records_seen,
            'audit_condition_records_dropped': self._audit_condition_records_dropped,
            'pending_capacity': self.replay.max_pending,
            'completed_record_capacity': self.replay.max_records,
            'pending_current_frame_count': self.replay.frame_count,
            'minimum_audit_sample_spacing_s': self.minimum_audit_sample_spacing_s,
        }

    def finish_condition(self) -> dict[str, int]:
        stats = self.replay.finish_condition()
        stats['resolved_record_count'] = max(
            0,
            int(stats.get('resolved_record_count', 0)) - self._resolved_record_count_at_condition_start,
        )
        stats['discontinuity_count'] = max(
            0,
            int(stats.get('discontinuity_count', 0)) - self._discontinuity_count_at_condition_start,
        )
        stats['pending_prediction_dropped_count'] = len(self._pending_prediction_metadata)
        self._pending_audit_dropped_count += len(self._pending_prediction_metadata)
        self._pending_prediction_metadata.clear()
        self._history.clear()
        self._previous_blade_root_my = None
        self._condition_key = None
        return stats

    def _feature_from_payload(
        self,
        payload: Mapping[str, float],
        *,
        sent_action_real: Sequence[float],
        projected_candidate_action_real: Sequence[float],
        state_encoding: Sequence[float],
        pid_peak_abs_nm: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        def value(name: str) -> float:
            try:
                parsed = float(payload[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'load predictor payload is missing finite {name}') from exc
            if not np.isfinite(parsed):
                raise ValueError(f'load predictor payload is missing finite {name}')
            return parsed

        sent = np.asarray(sent_action_real, dtype=np.float64)
        candidate = np.asarray(projected_candidate_action_real, dtype=np.float64)
        state = np.asarray(state_encoding, dtype=np.float64)
        pid_peak = np.asarray(pid_peak_abs_nm, dtype=np.float64)
        if sent.shape != (2,) or candidate.shape != (2,) or state.shape != (STATE_ENCODING_WIDTH,) or pid_peak.shape != (3,):
            raise ValueError('load predictor shadow action/state/PID vectors have incompatible shapes')
        if not np.isfinite(sent).all() or not np.isfinite(candidate).all() or not np.isfinite(state).all() or not np.isfinite(pid_peak).all() or np.any(pid_peak <= 0.0):
            raise ValueError('load predictor shadow action/state/PID vectors must be finite')
        blade_root_my = np.asarray([value(name) for name in BLADE_ROOT_MY_FIELDS], dtype=np.float64)
        if self._previous_blade_root_my is None:
            blade_root_rate = np.zeros(3, dtype=np.float64)
        else:
            blade_root_rate = (blade_root_my - self._previous_blade_root_my) / self.interface.step_duration_s
        self._previous_blade_root_my = blade_root_my.copy()
        azimuth = value('I_RotorAzimuth')
        physical = np.asarray([value(name) for name in PHYSICAL_FEATURE_FIELDS], dtype=np.float64)
        feature = np.concatenate((
            physical,
            np.asarray([np.sin(azimuth), np.cos(azimuth)], dtype=np.float64),
            blade_root_my,
            blade_root_rate,
            sent,
            candidate,
            state,
            1.0 - np.abs(blade_root_my) / pid_peak,
        ))
        if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
            raise ValueError('load predictor shadow feature construction produced invalid values')
        return feature.astype(np.float32), blade_root_my
