from __future__ import annotations

import json
from pathlib import Path


def normalize_user_path(path: str) -> Path:
    normalized = str(path).strip().replace('\\', '/')
    return Path(normalized).expanduser()


def load_config(path: str) -> dict:
    config_path = normalize_user_path(path)
    with config_path.open('r', encoding='utf-8') as file:
        if config_path.suffix.lower() == '.json':
            return json.load(file)
        if config_path.suffix.lower() in {'.yaml', '.yml'}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError('PyYAML is required to load YAML config files') from exc
            return yaml.safe_load(file)
    raise ValueError(f'Unsupported config file format: {config_path}')
