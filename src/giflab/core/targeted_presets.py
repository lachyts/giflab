"""Targeted experiment preset system for efficient pipeline generation.

This module provides a preset-based approach for generating specific pipeline
combinations instead of creating all possible combinations then sampling.
Enables focused research studies with dramatic efficiency improvements.

Example:
    Frame removal study: 5 pipelines instead of 935 (99.5% reduction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


@dataclass
class SlotConfiguration:
    """Configuration for a single algorithm slot (frame/color/lossy).

    Each slot can be either:
    - Variable: Test multiple algorithms from a specified scope
    - Locked: Use a specific algorithm with fixed parameters

    Examples:
        # Variable slot: test all frame algorithms
        SlotConfiguration(
            type="variable",
            scope=["*"],  # All available tools
            parameters={"ratios": [1.0, 0.8, 0.5]}
        )

        # Locked slot: use specific implementation
        SlotConfiguration(
            type="locked",
            implementation="animately-advanced",
            parameters={"level": 40}
        )
    """

    type: Literal["variable", "locked"]
    implementation: (
        str | None
    ) = None  # For locked slots: "ffmpeg-color", "animately-advanced"
    scope: (
        list[str] | None
    ) = None  # For variable slots: ["*"] or ["gifski", "animately"]
    parameters: dict[str, Any] = field(
        default_factory=dict
    )  # Algorithm-specific parameters

    def __post_init__(self) -> None:
        """Validate slot configuration after initialization."""
        if self.type == "locked":
            if not self.implementation:
                raise ValueError("Locked slots must specify an implementation")
            if self.scope is not None:
                raise ValueError(
                    "Locked slots cannot have a scope (use implementation instead)"
                )
        elif self.type == "variable":
            if not self.scope:
                raise ValueError("Variable slots must specify a scope")
            if self.implementation is not None:
                raise ValueError(
                    "Variable slots cannot have an implementation (use scope instead)"
                )
        else:
            raise ValueError(
                f"Invalid slot type: {self.type}. Must be 'variable' or 'locked'"
            )


@dataclass
class ExperimentPreset:
    """Targeted experiment configuration for focused pipeline generation.

    Defines which algorithm slots are variable (tested) vs locked (fixed),
    enabling precise control over experiment scope without unnecessary
    pipeline generation.

    Example:
        # Frame removal focus study
        ExperimentPreset(
            name="Frame Removal Focus",
            description="Compare frame algorithms with locked color/lossy",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="animately-advanced")
        )
    """

    name: str
    description: str
    frame_slot: SlotConfiguration
    color_slot: SlotConfiguration
    lossy_slot: SlotConfiguration

    # Optional configuration overrides
    custom_sampling: str | None = None  # "quick", "representative", etc.
    max_combinations: int | None = None  # Hard limit on generated pipelines

    # Metadata
    tags: list[str] = field(
        default_factory=list
    )  # ["research", "comparison", "optimization"]
    author: str | None = None  # Preset creator
    version: str = "1.0"  # Preset version for compatibility

    def __post_init__(self) -> None:
        """Validate experiment preset after initialization."""
        # Ensure at least one variable slot
        variable_count = sum(
            1
            for slot in [self.frame_slot, self.color_slot, self.lossy_slot]
            if slot.type == "variable"
        )
        if variable_count == 0:
            raise ValueError(
                "At least one slot must be variable (otherwise use generate_all_pipelines)"
            )

        # Validate sampling strategy if specified
        if self.custom_sampling:
            from .sampling import SAMPLING_STRATEGIES

            if self.custom_sampling not in SAMPLING_STRATEGIES:
                valid_strategies = list(SAMPLING_STRATEGIES.keys())
                raise ValueError(
                    f"Invalid sampling strategy: {self.custom_sampling}. "
                    f"Valid options: {valid_strategies}"
                )

        # Validate max_combinations
        if self.max_combinations is not None and self.max_combinations <= 0:
            raise ValueError("max_combinations must be positive")

    def get_variable_slots(self) -> list[str]:
        """Return list of variable slot names (e.g., ['frame', 'color'])."""
        variables = []
        if self.frame_slot.type == "variable":
            variables.append("frame")
        if self.color_slot.type == "variable":
            variables.append("color")
        if self.lossy_slot.type == "variable":
            variables.append("lossy")
        return variables

    def get_locked_slots(self) -> dict[str, SlotConfiguration]:
        """Return mapping of locked slot names to their configurations."""
        locked = {}
        if self.frame_slot.type == "locked":
            locked["frame"] = self.frame_slot
        if self.color_slot.type == "locked":
            locked["color"] = self.color_slot
        if self.lossy_slot.type == "locked":
            locked["lossy"] = self.lossy_slot
        return locked

    def estimate_pipeline_count(self) -> int:
        """Estimate number of pipelines this preset will generate.

        Note: This is an approximation since actual tool availability
        depends on system configuration.
        """
        # Rough estimates based on typical tool counts
        TYPICAL_TOOL_COUNTS = {
            "frame": 5,  # animately, ffmpeg, gifsicle, imagemagick, none
            "color": 17,  # All color reduction variants
            "lossy": 11,  # All lossy compression variants
        }

        count = 1
        for slot_name, slot in [
            ("frame", self.frame_slot),
            ("color", self.color_slot),
            ("lossy", self.lossy_slot),
        ]:
            if slot.type == "variable":
                if slot.scope and "*" in slot.scope:
                    # Wildcard scope uses all available tools
                    count *= TYPICAL_TOOL_COUNTS[slot_name]
                elif slot.scope:
                    # Specific tool list
                    count *= len(slot.scope)
            # Locked slots contribute 1 to the count (already initialized)

        return min(count, self.max_combinations) if self.max_combinations else count


class PresetRegistry:
    """Registry for managing experiment presets with validation and conflict detection."""

    def __init__(self) -> None:
        self.presets: dict[str, ExperimentPreset] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, preset_id: str, preset: ExperimentPreset) -> None:
        """Register a preset with validation and conflict detection."""
        if not preset_id:
            raise ValueError("Preset ID cannot be empty")

        if preset_id in self.presets:
            self.logger.warning(f"Overwriting existing preset: {preset_id}")

        # Validate preset
        try:
            # This will call __post_init__ validation
            preset.__post_init__()
        except Exception as e:
            raise ValueError(f"Invalid preset configuration for '{preset_id}': {e}")

        self.presets[preset_id] = preset
        self.logger.info(f"Registered preset: {preset_id} ({preset.name})")

    def get(self, preset_id: str) -> ExperimentPreset:
        """Get a preset by ID with validation."""
        if preset_id not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Unknown preset: {preset_id}. Available: {available}")
        return self.presets[preset_id]

    def list_presets(self) -> dict[str, str]:
        """Return mapping of preset IDs to their descriptions."""
        return {pid: preset.description for pid, preset in self.presets.items()}

    def find_similar_presets(
        self, preset: ExperimentPreset, threshold: float = 0.8
    ) -> list[str]:
        """Find presets with similar configurations to detect potential conflicts."""
        similar = []

        for pid, existing in self.presets.items():
            similarity = self._calculate_preset_similarity(preset, existing)
            if similarity >= threshold:
                similar.append(pid)

        return similar

    def _calculate_preset_similarity(
        self, preset1: ExperimentPreset, preset2: ExperimentPreset
    ) -> float:
        """Calculate similarity between two presets (0.0 to 1.0)."""
        if preset1.name == preset2.name:
            return 1.0

        # Compare slot configurations
        matches = 0
        total = 3  # frame, color, lossy slots

        slots1 = [preset1.frame_slot, preset1.color_slot, preset1.lossy_slot]
        slots2 = [preset2.frame_slot, preset2.color_slot, preset2.lossy_slot]

        for slot1, slot2 in zip(slots1, slots2, strict=True):
            if slot1.type == slot2.type:
                if (
                    slot1.type == "locked"
                    and slot1.implementation == slot2.implementation
                ):
                    matches += 1
                elif slot1.type == "variable" and slot1.scope == slot2.scope:
                    matches += 1

        return matches / total


# Global preset registry instance
PRESET_REGISTRY = PresetRegistry()
