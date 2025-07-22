from __future__ import annotations

"""YAML helpers for persisting selected pipeline identifiers."""

from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # PyYAML is optional for core runtime


def _require_yaml():
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run `poetry add pyyaml` or `pip install pyyaml`.")


def read_pipelines_yaml(path: Path) -> list[str]:
    """Return list of pipeline identifiers from YAML file."""
    _require_yaml()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "pipelines" not in data or not isinstance(data["pipelines"], list):
        raise ValueError("Invalid pipelines YAML format – expected key 'pipelines: [list]'")
    return [str(x) for x in data["pipelines"]]


def write_pipelines_yaml(path: Path, pipelines: list[str]) -> None:
    """Write list of identifiers to YAML."""
    _require_yaml()
    content = {"pipelines": sorted(set(pipelines))}
    path.write_text(yaml.safe_dump(content, sort_keys=False))
