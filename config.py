"""YAML configuration handling.

Configurations are plain nested dictionaries wrapped in :class:`Config` for
attribute access. A YAML file may inherit from other files through a
``_base_`` key (a path or a list of paths relative to the file), and any value
can be overridden from the command line with ``--set key.sub=value``.
All values used in the paper are in ``configs/default.yaml`` (Tables A4-A8).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """A dict with attribute access and dotted-path helpers."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(item) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))

    # ------------------------------------------------------------------
    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        merged = deep_merge(merged, _load_yaml((path.parent / base).resolve()))
    return deep_merge(merged, data)


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(paths: str | Path | Iterable[str | Path] | None = None,
                overrides: Iterable[str] | None = None) -> Config:
    """Load one or more YAML files (later files win) and apply overrides.

    ``overrides`` are strings of the form ``"training.max_epochs=5"``.
    When no path is given, ``configs/default.yaml`` of the repository is used.
    """
    if paths is None:
        paths = [default_config_path()]
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, _load_yaml(Path(p).resolve()))
    cfg = _wrap(merged)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), _parse_scalar(value.strip()))
    return cfg


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """``configs/default.yaml`` of the source checkout (editable install), else of the working directory."""
    for candidate in (repo_root() / "configs" / "default.yaml", Path.cwd() / "configs" / "default.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("configs/default.yaml not found: run from the repository root or pass --config")


def save_config(cfg: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), fh, sort_keys=False)
