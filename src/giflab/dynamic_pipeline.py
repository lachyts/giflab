from __future__ import annotations

"""Dynamic pipeline generator (Stage-4).

Builds all combinations of (frame → color → lossy) slots using the capability
registry.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .capability_registry import tools_for
from .tool_interfaces import ExternalTool

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineStep:
    variable: str  # "frame_reduction" | "color_reduction" | "lossy_compression"
    tool_cls: type[ExternalTool]

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


def generate_all_pipelines() -> list[Pipeline]:
    """Return *every* valid 3-slot pipeline (may be hundreds)."""

    frame_tools = tools_for("frame_reduction")
    color_tools = tools_for("color_reduction")
    lossy_tools = tools_for("lossy_compression")

    pipelines: list[Pipeline] = []
    for trio in itertools.product(frame_tools, color_tools, lossy_tools):
        raw_steps = [
            PipelineStep("frame_reduction", trio[0]),
            PipelineStep("color_reduction", trio[1]),
            PipelineStep("lossy_compression", trio[2]),
        ]
        pipelines.append(Pipeline(raw_steps))

    return pipelines
