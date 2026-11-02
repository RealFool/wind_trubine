from __future__ import annotations

from dataclasses import dataclass
import math
from collections import deque
from typing import Any, Sequence

import numpy as np

from RL.utils.load_predictor_contract import (
    PLAN_KIND_REALIZED,
    PLAN_KIND_ZOH_CANDIDATE,
    LoadPredictorInterface,
    make_zoh_candidate_plan,
    normalized_action_plan_mismatch,
)


@dataclass(frozen=True)
class LoadPredictionFrame:
    condition_id: int
    condition_instance_id: int
    timestamp_s: float
    sent_action_real: np.ndarray
    actual_electrical_torque_nm: float
    actual_pitch_rad: float
    blade_root_my_nm: np.ndarray
    pid_peak_abs_nm: np.ndarray
    actor_policy_version: int = 0

    def __post_init__(self) -> None:
        action = np.asarray(self.sent_action_real, dtype=np.float64)
        blade_root = np.asarray(self.blade_root_my_nm, dtype=np.float64)
        pid_peak = np.asarray(self.pid_peak_abs_nm, dtype=np.float64)
        if action.shape != (2,) or blade_root.shape != (3,) or pid_peak.shape != (3,):
            raise ValueError('Load prediction frame action/load vectors have incompatible shapes')
        scalar_values = (
            self.timestamp_s,
            self.actual_electrical_torque_nm,
            self.actual_pitch_rad,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError('Load prediction frame scalar values must be finite')
        if not np.isfinite(action).all() or not np.isfinite(blade_root).all() or not np.isfinite(pid_peak).all():
            raise ValueError('Load prediction frame vectors must be finite')
        if np.any(pid_peak <= 0.0):
            raise ValueError('Load prediction frame PID peaks must be positive')
        if int(self.actor_policy_version) < 0:
            raise ValueError('Load prediction frame Actor policy version must be non-negative')
        object.__setattr__(self, 'sent_action_real', action.copy())
        object.__setattr__(self, 'blade_root_my_nm', blade_root.copy())
        object.__setattr__(self, 'pid_peak_abs_nm', pid_peak.copy())


@dataclass(frozen=True)
class PendingLoadPredictionSample:
    sample_id: int
    condition_id: int
    condition_instance_id: int
    start_frame_index: int
    history: np.ndarray
    candidate_action_real: np.ndarray | None
    plan_kind: str


@dataclass(frozen=True)
class LoadPredictionRecord:
    sample_id: int
    condition_id: int
    condition_instance_id: int
    horizon_s: float
    horizon_frames: int
    plan_kind: str
    online_authorized: bool
    model_input_plan: np.ndarray
    actual_future_action_plan: np.ndarray
    future_electrical_torque_nm: float
    future_pitch_rad: float
    future_blade_root_peak_ratio: np.ndarray
    plan_mismatch: float | None
    calibration_eligible: bool
    history: np.ndarray
    actual_future_actor_policy_versions: tuple[int, ...] = ()


class LoadPredictionReplay:
    """Time-ordered pending replay for causal future-window labels.

    The future actual action plan is retained as a label/diagnostic only. The
    online model input plan is explicitly stored separately and is ZOH-only.
    """

    def __init__(
        self,
        interface: LoadPredictorInterface,
        *,
        action_low: Sequence[float],
        action_high: Sequence[float],
        max_pending: int = 100_000,
        max_records: int = 2_048,
        timestamp_tolerance_s: float = 1.0e-6,
    ) -> None:
        if max_pending <= 0:
            raise ValueError('max_pending must be positive')
        if max_records <= 0:
            raise ValueError('max_records must be positive')
        if not math.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s < 0.0:
            raise ValueError('timestamp_tolerance_s must be finite and non-negative')
        self.interface = interface
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        if self.action_low.shape != (2,) or self.action_high.shape != (2,) or np.any(self.action_high <= self.action_low):
            raise ValueError('action bounds must contain two increasing values')
        self.max_pending = int(max_pending)
        self.max_records = int(max_records)
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self._frames: list[LoadPredictionFrame] = []
        self._pending: list[PendingLoadPredictionSample] = []
        self._records: deque[LoadPredictionRecord] = deque(maxlen=self.max_records)
        self._active_condition: tuple[int, int] | None = None
        self._next_sample_id = 1
        self._dropped_pending_count = 0
        self._resolved_record_count = 0
        self._discontinuity_count = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_sample_ids(self) -> tuple[int, ...]:
        return tuple(sample.sample_id for sample in self._pending)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start_condition(self, condition_id: int, condition_instance_id: int) -> dict[str, int]:
        dropped = len(self._pending)
        self._dropped_pending_count += dropped
        self._pending.clear()
        self._frames.clear()
        self._active_condition = (int(condition_id), int(condition_instance_id))
        return {
            'dropped_pending_count': dropped,
            'pending_count': 0,
            'frame_count': 0,
        }

    def observe_frame(self, frame: LoadPredictionFrame) -> list[LoadPredictionRecord]:
        context = (int(frame.condition_id), int(frame.condition_instance_id))
        if self._active_condition is None:
            self.start_condition(*context)
        elif context != self._active_condition:
            self.start_condition(*context)
        if self._frames:
            delta_s = frame.timestamp_s - self._frames[-1].timestamp_s
            if delta_s <= 0.0 or not math.isclose(
                delta_s,
                self.interface.step_duration_s,
                rel_tol=0.0,
                abs_tol=self.timestamp_tolerance_s,
            ):
                self._discontinuity_count += 1
                self._dropped_pending_count += len(self._pending)
                self._pending.clear()
                self._frames.clear()
        self._frames.append(frame)
        current_index = len(self._frames) - 1
        resolved: list[LoadPredictionRecord] = []
        remaining: list[PendingLoadPredictionSample] = []
        max_horizon_frames = max(self.interface.horizon_frames)
        for sample in self._pending:
            if current_index - sample.start_frame_index < max_horizon_frames:
                remaining.append(sample)
                continue
            sample_records = self._resolve_sample(sample)
            self._records.extend(sample_records)
            resolved.extend(sample_records)
            self._resolved_record_count += 1
        self._pending = remaining
        minimum_history_start = max(0, current_index - self.interface.history_frames - max_horizon_frames)
        if minimum_history_start > 0:
            self._frames = self._frames[minimum_history_start:]
            self._pending = [
                PendingLoadPredictionSample(
                    sample_id=sample.sample_id,
                    condition_id=sample.condition_id,
                    condition_instance_id=sample.condition_instance_id,
                    start_frame_index=sample.start_frame_index - minimum_history_start,
                    history=sample.history,
                    candidate_action_real=sample.candidate_action_real,
                    plan_kind=sample.plan_kind,
                )
                for sample in self._pending
            ]
        return resolved

    def add_pending(
        self,
        *,
        history: np.ndarray,
        candidate_action_real: Sequence[float] | None,
        plan_kind: str,
        online: bool = False,
    ) -> int:
        self.interface.validate_plan_kind(plan_kind, online=online)
        if self._active_condition is None or not self._frames:
            raise RuntimeError('Cannot add a pending predictor sample before the first condition frame')
        history_array = np.asarray(history, dtype=np.float32)
        if history_array.ndim != 2 or history_array.shape[0] != self.interface.history_frames:
            raise ValueError(
                f'Predictor history must have {self.interface.history_frames} frames, got {history_array.shape}'
            )
        if not np.isfinite(history_array).all():
            raise ValueError('Predictor history must be finite')
        candidate = None
        if candidate_action_real is not None:
            candidate = np.asarray(candidate_action_real, dtype=np.float64)
            if candidate.shape != (2,) or not np.isfinite(candidate).all():
                raise ValueError('candidate_action_real must contain two finite values')
        if len(self._pending) >= self.max_pending:
            raise RuntimeError('Load prediction pending buffer is full')
        sample = PendingLoadPredictionSample(
            sample_id=self._next_sample_id,
            condition_id=self._active_condition[0],
            condition_instance_id=self._active_condition[1],
            start_frame_index=len(self._frames) - 1,
            history=history_array.copy(),
            candidate_action_real=None if candidate is None else candidate.copy(),
            plan_kind=str(plan_kind),
        )
        self._pending.append(sample)
        self._next_sample_id += 1
        return sample.sample_id

    def finish_condition(self) -> dict[str, int]:
        dropped = len(self._pending)
        self._dropped_pending_count += dropped
        self._pending.clear()
        self._frames.clear()
        self._active_condition = None
        return {
            'dropped_pending_count': dropped,
            'pending_count': 0,
            'resolved_record_count': self._resolved_record_count,
            'discontinuity_count': self._discontinuity_count,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            'next_sample_id': self._next_sample_id,
            'dropped_pending_count': self._dropped_pending_count,
            'resolved_record_count': self._resolved_record_count,
            'discontinuity_count': self._discontinuity_count,
            'pending_count': len(self._pending),
            'active_condition': self._active_condition,
        }

    def _resolve_sample(self, sample: PendingLoadPredictionSample) -> list[LoadPredictionRecord]:
        records: list[LoadPredictionRecord] = []
        actual_actions = np.asarray([
            self._frames[sample.start_frame_index + offset].sent_action_real
            for offset in range(1, max(self.interface.horizon_frames) + 1)
        ], dtype=np.float64)
        for horizon_s, horizon_frames in zip(self.interface.horizons_s, self.interface.horizon_frames, strict=True):
            future_frames = self._frames[
                sample.start_frame_index + 1:sample.start_frame_index + horizon_frames + 1
            ]
            future_action_plan = actual_actions[:horizon_frames].copy()
            future_my = np.asarray([frame.blade_root_my_nm for frame in future_frames], dtype=np.float64)
            future_peak_ratio = np.max(np.abs(future_my), axis=0) / future_frames[-1].pid_peak_abs_nm
            future_torque = float(future_frames[-1].actual_electrical_torque_nm)
            future_pitch = float(future_frames[-1].actual_pitch_rad)
            future_actor_policy_versions = tuple(
                int(frame.actor_policy_version)
                for frame in future_frames
            )
            candidate_plan = None
            mismatch = None
            calibration_eligible = False
            if sample.candidate_action_real is not None:
                candidate_plan = make_zoh_candidate_plan(sample.candidate_action_real, horizon_frames)
                mismatch = normalized_action_plan_mismatch(
                    future_action_plan,
                    candidate_plan,
                    self.action_low,
                    self.action_high,
                )
                calibration_eligible = mismatch <= self.interface.max_plan_mismatch
            if (
                sample.plan_kind == PLAN_KIND_ZOH_CANDIDATE
                and candidate_plan is not None
                and calibration_eligible
                and horizon_s in self.interface.soft_guard_horizons_s
            ):
                records.append(LoadPredictionRecord(
                    sample_id=sample.sample_id,
                    condition_id=sample.condition_id,
                    condition_instance_id=sample.condition_instance_id,
                    horizon_s=horizon_s,
                    horizon_frames=horizon_frames,
                    plan_kind=PLAN_KIND_ZOH_CANDIDATE,
                    online_authorized=True,
                    model_input_plan=candidate_plan,
                    actual_future_action_plan=future_action_plan,
                    future_electrical_torque_nm=future_torque,
                    future_pitch_rad=future_pitch,
                    future_blade_root_peak_ratio=future_peak_ratio,
                    plan_mismatch=mismatch,
                    calibration_eligible=calibration_eligible and horizon_s in self.interface.soft_guard_horizons_s,
                    history=sample.history.copy(),
                    actual_future_actor_policy_versions=future_actor_policy_versions,
                ))
            if sample.plan_kind in (PLAN_KIND_ZOH_CANDIDATE, PLAN_KIND_REALIZED):
                records.append(LoadPredictionRecord(
                    sample_id=sample.sample_id,
                    condition_id=sample.condition_id,
                    condition_instance_id=sample.condition_instance_id,
                    horizon_s=horizon_s,
                    horizon_frames=horizon_frames,
                    plan_kind=PLAN_KIND_REALIZED,
                    online_authorized=False,
                    model_input_plan=future_action_plan,
                    actual_future_action_plan=future_action_plan,
                    future_electrical_torque_nm=future_torque,
                    future_pitch_rad=future_pitch,
                    future_blade_root_peak_ratio=future_peak_ratio,
                    plan_mismatch=mismatch,
                    calibration_eligible=False,
                    history=sample.history.copy(),
                    actual_future_actor_policy_versions=future_actor_policy_versions,
                ))
        return records

    def samples(self, *, plan_kind: str | None = None) -> tuple[LoadPredictionRecord, ...]:
        if plan_kind is None:
            return tuple(self._records)
        self.interface.validate_plan_kind(plan_kind, online=False)
        return tuple(record for record in self._records if record.plan_kind == plan_kind)
