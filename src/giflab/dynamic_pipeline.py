from __future__ import annotations

"""Dynamic pipeline generator (Stage-4).

Builds all combinations of (frame → color → lossy) slots using the capability
registry.  Consecutive steps are merged when `combines_with()` returns *True*.
"""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Type

from .tool_interfaces import ExternalTool
from .capability_registry import tools_for

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PipelineStep:
    variable: str  # "frame_reduction" | "color_reduction" | "lossy_compression"
    tool_cls: Type[ExternalTool]

    def name(self) -> str:
        suffix = {
            "frame_reduction": "Frame",
            "color_reduction": "Color",
            "lossy_compression": "Lossy",
        }[self.variable]
        return f"{self.tool_cls.NAME}_{suffix}"


@dataclass(frozen=True)
class Pipeline:
    steps: Sequence[PipelineStep]

    def identifier(self) -> str:
        return "__".join(step.name() for step in self.steps)

    # For Stage-5 execution: we may add an `execute()` method later


# ---------------------------------------------------------------------------
# Generator logic
# ---------------------------------------------------------------------------

_VARIABLE_ORDER = ["frame_reduction", "color_reduction", "lossy_compression"]


def _collapse_steps(raw_steps: List[PipelineStep]) -> List[PipelineStep]:
    """Merge consecutive steps when their tool_cls can combine."""
    if not raw_steps:
        return []

    collapsed: List[PipelineStep] = [raw_steps[0]]
    for step in raw_steps[1:]:
        last = collapsed[-1]
        if last.tool_cls().combines_with(step.tool_cls()):  # type: ignore[arg-type]
            # Skip adding new step – treated as merged.
            continue
        collapsed.append(step)
    return collapsed


def generate_all_pipelines() -> List[Pipeline]:
    """Return *every* valid 3-slot pipeline (may be hundreds)."""

    frame_tools = tools_for("frame_reduction")
    color_tools = tools_for("color_reduction")
    lossy_tools = tools_for("lossy_compression")

    pipelines: List[Pipeline] = []
    for trio in itertools.product(frame_tools, color_tools, lossy_tools):
        raw_steps = [
            PipelineStep("frame_reduction", trio[0]),
            PipelineStep("color_reduction", trio[1]),
            PipelineStep("lossy_compression", trio[2]),
        ]
        steps = _collapse_steps(raw_steps)
        pipelines.append(Pipeline(steps))

    return pipelines 