from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from RL.utils.constants import (
    ACTION_VARIABLES,
    CONTROL_OBSERVATION_SCALE_FLOORS,
    DATA_SCHEMA_VERSION,
    OBSERVATION_VARIABLES,
    RAW_OBSERVATION_VARIABLES,
)
from RL.utils.observation import append_previous_action_observation, build_observation_from_frame
from RL.utils.operating_state import OperatingPoint
from RL.utils.state_action_artifact import (
    STATE_AWARE_CONTROL_OBS_DIM,
    StateActionEnvelopeArtifact,
    append_state_regime_observation,
)


@dataclass(frozen=True)
class OfflineWindow:
    window_index: int
    source_file: str
    start_index: int
    end_index: int
    condition_id: int | None


class OfflineBCDataset(Dataset):
    CACHE_VERSION = 8

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        max_files: int | None = None,
        observation_wind_override: float | None = None,
        cache_enabled: bool = True,
        cache_dir: str | None = None,
        include_previous_action_observation: bool = False,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
        precomputed_action_target_columns: list[str] | None = None,
        include_operating_state_observation: bool = False,
        state_action_artifact_path: str | None = None,
    ) -> None:
        split_dir = Path(root_dir) / split
        self.file_paths = sorted(split_dir.glob('*.csv'))
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]
        if not self.file_paths:
            raise FileNotFoundError(f'No CSV files found in {split_dir}')
        self.include_previous_action_observation = bool(include_previous_action_observation)
        self.include_operating_state_observation = bool(include_operating_state_observation)
        self.state_action_artifact: StateActionEnvelopeArtifact | None = None
        self.state_action_artifact_sha256: str | None = None
        if self.include_operating_state_observation:
            if not self.include_previous_action_observation:
                raise ValueError('State-aware observations require previous-action observations')
            if not isinstance(state_action_artifact_path, str) or not state_action_artifact_path.strip():
                raise ValueError('State-aware observations require state_action_artifact_path')
            artifact_path = Path(state_action_artifact_path).expanduser()
            self.state_action_artifact = StateActionEnvelopeArtifact.from_path(artifact_path)
            self.state_action_artifact_sha256 = self.state_action_artifact.source_sha256
        self.precomputed_action_target_columns = (
            None
            if precomputed_action_target_columns is None
            else tuple(str(column) for column in precomputed_action_target_columns)
        )
        if self.precomputed_action_target_columns is not None and len(self.precomputed_action_target_columns) != 2:
            raise ValueError('precomputed_action_target_columns must contain exactly two column names')
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float64)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float64)
        if self.include_previous_action_observation:
            if self.action_low is None or self.action_high is None:
                raise ValueError('Previous-action observations require action_low and action_high')
            if (
                self.action_low.shape != (2,)
                or self.action_high.shape != (2,)
                or np.any(self.action_high <= self.action_low)
            ):
                raise ValueError('action_low/action_high must contain two increasing values')

        cache_paths = self._cache_paths(
            root_dir=Path(root_dir),
            split=split,
            max_files=max_files,
            observation_wind_override=observation_wind_override,
            cache_dir=cache_dir,
            include_previous_action_observation=self.include_previous_action_observation,
            action_low=self.action_low,
            action_high=self.action_high,
            precomputed_action_target_columns=self.precomputed_action_target_columns,
            include_operating_state_observation=self.include_operating_state_observation,
            state_action_artifact_sha256=self.state_action_artifact_sha256,
        )
        source_fingerprint = self._source_fingerprint()
        if cache_enabled and self._load_cache(cache_paths, source_fingerprint):
            print(
                f'[BC-Data] loaded cache split={split} samples={self.obs.shape[0]} '
                f'from {cache_paths["metadata"].parent}',
                flush=True,
            )
            return

        wind_override_msg = ''
        if observation_wind_override is not None:
            wind_override_msg = f' observation_wind_override={float(observation_wind_override):.3f}'
        print(
            f'[BC-Data] loading split={split} files={len(self.file_paths)} from {split_dir}{wind_override_msg}',
            flush=True,
        )

        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        action_target_chunks: list[np.ndarray] = []
        previous_action_chunks: list[np.ndarray] = []
        windows: list[OfflineWindow] = []
        total_rows = 0
        progress_every = max(1, len(self.file_paths) // 10)
        wind_index = OBSERVATION_VARIABLES.index('I_AverageWindSpeed')
        required_columns = list(dict.fromkeys(
            RAW_OBSERVATION_VARIABLES
            + ACTION_VARIABLES
            + (['Time'] if self.include_operating_state_observation else [])
            + ([] if self.precomputed_action_target_columns is None else list(self.precomputed_action_target_columns))
        ))
        for index, file_path in enumerate(self.file_paths, start=1):
            frame = pd.read_csv(file_path, usecols=required_columns)
            obs = build_observation_from_frame(frame)
            if observation_wind_override is not None:
                obs[:, wind_index] = float(observation_wind_override)
            actions_real = frame[ACTION_VARIABLES].to_numpy(dtype=np.float64)
            actions = actions_real.astype(np.float32)
            if self.precomputed_action_target_columns is not None:
                action_targets = frame[list(self.precomputed_action_target_columns)].to_numpy(dtype=np.float32)
                if not np.isfinite(action_targets).all() or np.any(np.abs(action_targets) > 1.0 + 1.0e-6):
                    raise ValueError(f'Precomputed action targets must be finite and within [-1, 1]: {file_path}')
                action_target_chunks.append(action_targets)
            previous_actions_real = actions_real.copy()
            if len(actions_real) > 1:
                previous_actions_real[1:] = actions_real[:-1]
            previous_actions = previous_actions_real.astype(np.float32)
            if self.include_previous_action_observation:
                obs = append_previous_action_observation(
                    obs,
                    previous_actions_real,
                    self.action_low,
                    self.action_high,
                )
            if self.include_operating_state_observation:
                assert self.state_action_artifact is not None
                classifier = self.state_action_artifact.classifier()
                state_observations: list[np.ndarray] = []
                for row_index, control_observation in enumerate(obs):
                    classification = classifier.classify(OperatingPoint(
                        timestamp_s=float(frame['Time'].iat[row_index]),
                        wind_speed_mps=float(control_observation[wind_index]),
                        rotor_speed_rad_s=float(frame['I_RotorSpeed'].iat[row_index]),
                        real_power_w=float(frame['I_RealPower'].iat[row_index]),
                        pitch_rad=float(frame['I_PitchPosition1'].iat[row_index]),
                        previous_torque_nm=float(previous_actions_real[row_index, 0]),
                        previous_pitch_rad=float(previous_actions_real[row_index, 1]),
                    ))
                    state_observations.append(
                        append_state_regime_observation(control_observation, classification)
                    )
                obs = np.stack(state_observations, axis=0).astype(np.float32)
            obs_chunks.append(obs)
            action_chunks.append(actions)
            previous_action_chunks.append(previous_actions)
            start_index = total_rows
            total_rows += int(len(frame))
            windows.append(
                OfflineWindow(
                    window_index=len(windows),
                    source_file=file_path.name,
                    start_index=start_index,
                    end_index=total_rows,
                    condition_id=self._condition_id_from_file_name(file_path.name),
                )
            )
            if index == 1 or index % progress_every == 0 or index == len(self.file_paths):
                print(
                    f'[BC-Data] loaded {index}/{len(self.file_paths)} files rows={total_rows}',
                    flush=True,
                )

        self.obs = np.concatenate(obs_chunks, axis=0)
        self.actions = np.concatenate(action_chunks, axis=0)
        self.action_targets = (
            None
            if self.precomputed_action_target_columns is None
            else np.concatenate(action_target_chunks, axis=0)
        )
        self.previous_actions = np.concatenate(previous_action_chunks, axis=0)
        self.windows = tuple(windows)
        self.obs_median = np.median(self.obs, axis=0).astype(np.float32)
        self.obs_iqr = np.subtract(*np.percentile(self.obs, [75, 25], axis=0)).astype(np.float32)
        scale_floors = np.asarray(
            CONTROL_OBSERVATION_SCALE_FLOORS
            if self.include_previous_action_observation
            else CONTROL_OBSERVATION_SCALE_FLOORS[: len(OBSERVATION_VARIABLES)],
            dtype=np.float32,
        )
        if self.include_operating_state_observation:
            scale_floors = np.concatenate((
                scale_floors,
                np.full(
                    STATE_AWARE_CONTROL_OBS_DIM - len(scale_floors),
                    0.01,
                    dtype=np.float32,
                ),
            ))
        self.obs_iqr = np.maximum(self.obs_iqr, scale_floors).astype(np.float32)
        if cache_enabled:
            self._save_cache(cache_paths, source_fingerprint)
        print(
            f'[BC-Data] ready samples={self.obs.shape[0]} obs_dim={self.obs.shape[1]} action_dim={self.actions.shape[1]}',
            flush=True,
        )

    def _cache_paths(
        self,
        root_dir: Path,
        split: str,
        max_files: int | None,
        observation_wind_override: float | None,
        cache_dir: str | None,
        include_previous_action_observation: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        precomputed_action_target_columns: tuple[str, ...] | None = None,
        include_operating_state_observation: bool = False,
        state_action_artifact_sha256: str | None = None,
    ) -> dict[str, Path]:
        cache_root = Path(cache_dir) if cache_dir else root_dir / '.rl_cache' / 'offline_bc'
        cache_key_data: dict[str, Any] = {
            'version': self.CACHE_VERSION,
            'data_schema_version': DATA_SCHEMA_VERSION,
            'split': split,
            'max_files': max_files,
            'observation_wind_override': observation_wind_override,
            'observation_variables': OBSERVATION_VARIABLES,
            'action_variables': ACTION_VARIABLES,
            'include_previous_action_observation': bool(include_previous_action_observation),
            'include_operating_state_observation': bool(include_operating_state_observation),
            'state_action_artifact_sha256': state_action_artifact_sha256,
            'action_low': None if action_low is None else action_low.astype(float).tolist(),
            'action_high': None if action_high is None else action_high.astype(float).tolist(),
            'precomputed_action_target_columns': precomputed_action_target_columns,
        }
        cache_key = hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode('utf-8')).hexdigest()[:16]
        prefix = cache_root / f'{split}_{cache_key}'
        return {
            'obs': prefix.with_suffix('.obs.npy'),
            'actions': prefix.with_suffix('.actions.npy'),
            'previous_actions': prefix.with_suffix('.previous_actions.npy'),
            'action_targets': prefix.with_suffix('.action_targets.npy'),
            'metadata': prefix.with_suffix('.json'),
        }

    @staticmethod
    def _condition_id_from_file_name(file_name: str) -> int | None:
        match = re.search(r'windspeed(?P<condition>\d+)', file_name, flags=re.IGNORECASE)
        return None if match is None else int(match.group('condition'))

    @staticmethod
    def _windows_from_metadata(rows: Any) -> tuple[OfflineWindow, ...]:
        if not isinstance(rows, list):
            raise ValueError('Offline dataset cache windows metadata must be a list')
        windows: list[OfflineWindow] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError('Offline dataset cache window metadata must contain objects')
            condition_id = row.get('condition_id')
            windows.append(
                OfflineWindow(
                    window_index=int(row.get('window_index', index)),
                    source_file=str(row['source_file']),
                    start_index=int(row['start_index']),
                    end_index=int(row['end_index']),
                    condition_id=None if condition_id is None else int(condition_id),
                )
            )
        return tuple(windows)

    def _source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for file_path in self.file_paths:
            stat = file_path.stat()
            digest.update(str(file_path.name).encode('utf-8'))
            digest.update(str(stat.st_size).encode('ascii'))
            digest.update(str(stat.st_mtime_ns).encode('ascii'))
        return digest.hexdigest()

    def _load_cache(self, cache_paths: dict[str, Path], source_fingerprint: str) -> bool:
        try:
            with cache_paths['metadata'].open('r', encoding='utf-8') as file:
                metadata = json.load(file)
            if metadata.get('data_schema_version') != DATA_SCHEMA_VERSION:
                return False
            if metadata.get('source_fingerprint') != source_fingerprint:
                return False
            if metadata.get('include_operating_state_observation', False) != self.include_operating_state_observation:
                return False
            if metadata.get('state_action_artifact_sha256') != self.state_action_artifact_sha256:
                return False
            self.obs = np.load(cache_paths['obs'], mmap_mode='r')
            self.actions = np.load(cache_paths['actions'], mmap_mode='r')
            self.previous_actions = np.load(cache_paths['previous_actions'], mmap_mode='r')
            target_columns = metadata.get('precomputed_action_target_columns')
            self.precomputed_action_target_columns = None if target_columns is None else tuple(target_columns)
            self.action_targets = (
                None
                if self.precomputed_action_target_columns is None
                else np.load(cache_paths['action_targets'], mmap_mode='r')
            )
            self.obs_median = np.asarray(metadata['obs_median'], dtype=np.float32)
            self.obs_iqr = np.asarray(metadata['obs_iqr'], dtype=np.float32)
            self.windows = self._windows_from_metadata(metadata['windows'])
            expected_obs_shape = (
                int(metadata['sample_count']),
                int(metadata.get(
                    'obs_dim',
                    STATE_AWARE_CONTROL_OBS_DIM
                    if self.include_operating_state_observation
                    else len(OBSERVATION_VARIABLES),
                )),
            )
            expected_action_shape = (int(metadata['sample_count']), len(ACTION_VARIABLES))
            return (
                self.obs.shape == expected_obs_shape
                and self.actions.shape == expected_action_shape
                and self.previous_actions.shape == expected_action_shape
                and (self.action_targets is None or self.action_targets.shape == expected_action_shape)
                and self.windows
                and self.windows[-1].end_index == int(metadata['sample_count'])
            )
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def _save_cache(self, cache_paths: dict[str, Path], source_fingerprint: str) -> None:
        cache_paths['metadata'].parent.mkdir(parents=True, exist_ok=True)
        self._atomic_save_array(cache_paths['obs'], self.obs)
        self._atomic_save_array(cache_paths['actions'], self.actions)
        self._atomic_save_array(cache_paths['previous_actions'], self.previous_actions)
        if self.action_targets is not None:
            self._atomic_save_array(cache_paths['action_targets'], self.action_targets)
        metadata = {
            'data_schema_version': DATA_SCHEMA_VERSION,
            'source_fingerprint': source_fingerprint,
            'sample_count': int(self.obs.shape[0]),
            'obs_dim': int(self.obs.shape[1]),
            'obs_median': self.obs_median.tolist(),
            'obs_iqr': self.obs_iqr.tolist(),
            'precomputed_action_target_columns': self.precomputed_action_target_columns,
            'include_operating_state_observation': self.include_operating_state_observation,
            'state_action_artifact_sha256': self.state_action_artifact_sha256,
            'windows': [
                {
                    'window_index': window.window_index,
                    'source_file': window.source_file,
                    'start_index': window.start_index,
                    'end_index': window.end_index,
                    'condition_id': window.condition_id,
                }
                for window in self.windows
            ],
        }
        metadata_tmp = cache_paths['metadata'].with_suffix('.json.tmp')
        with metadata_tmp.open('w', encoding='utf-8') as file:
            json.dump(metadata, file)
        metadata_tmp.replace(cache_paths['metadata'])
        print(f'[BC-Data] saved cache to {cache_paths["metadata"].parent}', flush=True)

    @staticmethod
    def _atomic_save_array(path: Path, array: np.ndarray) -> None:
        temporary_path = path.with_suffix(path.suffix + '.tmp')
        with temporary_path.open('wb') as file:
            np.save(file, array)
        temporary_path.replace(path)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        values = (
            torch.from_numpy(np.array(self.obs[index], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(self.actions[index], dtype=np.float32, copy=True)),
        )
        if self.action_targets is None:
            return values
        return (*values, torch.from_numpy(np.array(self.action_targets[index], dtype=np.float32, copy=True)))
