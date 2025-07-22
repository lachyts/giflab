from __future__ import annotations

"""Capability registry – maps compression variables to available tool wrappers.

Stage-3 requirement: enumerate every tool that can fulfil *one* variable slot so
that single-variable strategies can be generated automatically.
"""

import inspect
from collections.abc import Iterator

from .tool_interfaces import (
    ExternalTool,
)

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_all_wrapper_subclasses() -> Iterator[type[ExternalTool]]:
    """Yield every subclass defined in *giflab.tool_wrappers* (import side-effect)."""
    from . import tool_wrappers  # noqa: F401 – ensures module is imported

    for _, obj in inspect.getmembers(tool_wrappers, inspect.isclass):
        if issubclass(obj, ExternalTool) and obj is not ExternalTool:
            yield obj  # type: ignore[arg-type]


_REGISTRY: dict[str, list[type[ExternalTool]]] = {
    "color_reduction": [],
    "frame_reduction": [],
    "lossy_compression": [],
}

for cls in _iter_all_wrapper_subclasses():
    variable = getattr(cls, "VARIABLE", None)
    if variable in _REGISTRY and cls.available():
        _REGISTRY[variable].append(cls)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tools_for(variable: str) -> list[type[ExternalTool]]:
    """Return *available* tool wrapper classes for *variable*."""
    if variable not in _REGISTRY:
        raise ValueError(f"Unknown variable: {variable}")
    return list(_REGISTRY[variable])  # return copy


def all_single_variable_strategies() -> list[str]:
    """Return names like ``gifsicleColor`` representing each (variable, tool)."""
    names: list[str] = []
    for var, classes in _REGISTRY.items():
        for cls in classes:
            # Simple name scheme: <tool-name>|<variable-suffix> (camel-case)
            suffix = {
                "color_reduction": "Color",
                "frame_reduction": "Frame",
                "lossy_compression": "Lossy",
            }[var]
            names.append(f"{cls.NAME}_{suffix}")
    return names
