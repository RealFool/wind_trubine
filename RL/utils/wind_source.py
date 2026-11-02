from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from RL.utils.constants import ACTION_VARIABLES, DISTURBANCE_VARIABLES, REWARD_VARIABLES, STATE_VARIABLES


@dataclass
class EpisodeSlice:
    initial_state: np.ndarray
    wind: np.ndarray
    state_sequence: np.ndarray
    next_state_sequence: np.ndarray
    action_sequence: np.ndarray
    reward_sequence: np.ndarray


class _BaseWindSource:
    def __init__(self, data_root: str, split: str, seed: int = 42) -> None:
        self.mode = 'offline_sequence'
        self.split_dir = Path(data_root) / split
        self.file_paths = sorted(self.split_dir.glob('*.csv'))
        if not self.file_paths:
            raise FileNotFoundError(
                f'No CSV files found under {self.split_dir}. '
                'Set data.root_dir in config or RL_DATA_ROOT to your local dataset root.'
            )
        self.rng = random.Random(seed)

    def _sample_frame(self) -> pd.DataFrame:
        file_path = self.rng.choice(self.file_paths)
        return pd.read_csv(file_path)

    def sample_episode(self, episode_length: int) -> dict[str, np.ndarray]:
        frame = self._sample_frame()
        if len(frame) <= episode_length:
            start = 0
        else:
            start = self.rng.randint(0, len(frame) - episode_length)
        window = frame.iloc[start:start + episode_length].reset_index(drop=True)
        state_sequence = window.iloc[:-1][STATE_VARIABLES].to_numpy(dtype=np.float32)
        next_state_sequence = window.iloc[1:][STATE_VARIABLES].to_numpy(dtype=np.float32)
        action_sequence = window.iloc[:-1][ACTION_VARIABLES].to_numpy(dtype=np.float32)
        reward_sequence = window.iloc[1:][REWARD_VARIABLES].to_numpy(dtype=np.float32)
        state = state_sequence[0]
        wind = window[DISTURBANCE_VARIABLES[0]].to_numpy(dtype=np.float32)
        return {
            'initial_state': state,
            'wind': wind,
            'state_sequence': state_sequence,
            'next_state_sequence': next_state_sequence,
            'action_sequence': action_sequence,
            'reward_sequence': reward_sequence,
        }


class OfflineSequenceWindSource(_BaseWindSource):
    mode = 'offline_sequence'


class FixedSpeedWindSource(_BaseWindSource):
    def __init__(self, data_root: str, split: str, fixed_speed: float, seed: int = 42) -> None:
        super().__init__(data_root, split, seed=seed)
        self.mode = 'fixed_speed'
        self.fixed_speed = float(fixed_speed)

    def sample_episode(self, episode_length: int) -> dict[str, np.ndarray]:
        payload = super().sample_episode(episode_length)
        payload['wind'] = np.full((episode_length,), self.fixed_speed, dtype=np.float32)
        return payload


def build_wind_source(env_config: dict, data_config: dict):
    mode = env_config.get('wind_source_mode', 'offline_sequence')
    root_dir = os.environ.get('RL_DATA_ROOT', '').strip() or data_config['root_dir']
    split = data_config.get('train_split', 'train')
    seed = int(data_config.get('seed', 42))
    if mode == 'offline_sequence':
        return OfflineSequenceWindSource(root_dir, split, seed=seed)
    if mode == 'fixed_speed':
        return FixedSpeedWindSource(root_dir, split, fixed_speed=env_config.get('fixed_wind_speed', 12.0), seed=seed)
    raise ValueError(f'Unsupported wind source mode: {mode}')
