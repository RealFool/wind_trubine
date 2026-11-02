from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_CACHE: dict[str, ModuleType] = {}


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    cache_key = f'{module_name}:{file_path}'
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load module from {file_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def load_simulation_linear_model_class():
    module = _load_module(
        'simulation_environment_linear',
        _PROJECT_ROOT / 'simulation-environment' / 'model' / 'linear.py',
    )
    return module.LinearModel
