"""Targeted pipeline generator for efficient experiment design.

This module implements the core engine for generating specific pipeline
combinations based on experiment presets, replacing the generate_all_pipelines()
+ sampling approach with direct targeted creation.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from ..capability_registry import tools_for
from ..dynamic_pipeline import Pipeline, PipelineStep
from ..tool_interfaces import ExternalTool
from .targeted_presets import ExperimentPreset, SlotConfiguration


class TargetedPipelineGenerator:
    """Generates specific pipeline combinations based on experiment presets.

    This class replaces the inefficient generate_all_pipelines() + sampling
    approach by directly creating only the pipeline combinations needed for
    focused research studies.

    Example:
        # Frame removal study: 5 pipelines instead of 935 (99.5% reduction)
        generator = TargetedPipelineGenerator()
        preset = PRESET_REGISTRY.get("frame-focus")
        pipelines = generator.generate_targeted_pipelines(preset)
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the targeted pipeline generator.

        Args:
            logger: Optional logger instance for tracking generation progress
        """
        self.logger = logger or logging.getLogger(__name__)
        self._tool_cache: dict[str, list[type[ExternalTool]]] = {}

    def generate_targeted_pipelines(self, preset: ExperimentPreset) -> list[Pipeline]:
        """Generate specific pipeline combinations based on preset configuration.

        Args:
            preset: Experiment preset defining variable and locked slots

        Returns:
            List of Pipeline objects for the specific combinations needed

        Raises:
            ValueError: If preset configuration is invalid
            RuntimeError: If no valid tools found for variable slots
        """
        self.logger.info(f"🎯 Generating targeted pipelines for preset: {preset.name}")

        # Expand variable slots to actual tool classes
        frame_tools = self._resolve_slot_tools("frame_reduction", preset.frame_slot)
        color_tools = self._resolve_slot_tools("color_reduction", preset.color_slot)
        lossy_tools = self._resolve_slot_tools("lossy_compression", preset.lossy_slot)

        # Log generation plan
        self.logger.info("📊 Pipeline generation plan:")
        self.logger.info(
            f"   Frame tools: {len(frame_tools)} ({preset.frame_slot.type})"
        )
        self.logger.info(
            f"   Color tools: {len(color_tools)} ({preset.color_slot.type})"
        )
        self.logger.info(
            f"   Lossy tools: {len(lossy_tools)} ({preset.lossy_slot.type})"
        )

        total_combinations = len(frame_tools) * len(color_tools) * len(lossy_tools)
        self.logger.info(f"   Total combinations: {total_combinations}")

        if total_combinations == 0:
            raise RuntimeError(
                f"No valid tool combinations found for preset: {preset.name}"
            )

        # Generate pipeline combinations
        pipelines: list[Pipeline] = []

        for frame_tool, color_tool, lossy_tool in itertools.product(
            frame_tools, color_tools, lossy_tools
        ):
            pipeline_steps = [
                PipelineStep("frame_reduction", frame_tool),
                PipelineStep("color_reduction", color_tool),
                PipelineStep("lossy_compression", lossy_tool),
            ]
            pipelines.append(Pipeline(pipeline_steps))

        # Apply max_combinations limit if specified
        if preset.max_combinations and len(pipelines) > preset.max_combinations:
            self.logger.info(
                f"⚠️  Limiting pipelines to {preset.max_combinations} (from {len(pipelines)})"
            )
            pipelines = pipelines[: preset.max_combinations]

        self.logger.info(f"✅ Generated {len(pipelines)} targeted pipelines")
        return pipelines

    def _resolve_slot_tools(
        self, variable: str, slot_config: SlotConfiguration
    ) -> list[type[ExternalTool]]:
        """Resolve a slot configuration to actual tool classes.

        Args:
            variable: The variable type ("frame_reduction", "color_reduction", "lossy_compression")
            slot_config: Configuration for this slot (variable or locked)

        Returns:
            List of tool classes for this slot

        Raises:
            ValueError: If slot configuration is invalid
            RuntimeError: If no tools found for variable slot
        """
        if slot_config.type == "locked":
            # Locked slot: find specific implementation
            if slot_config.implementation is None:
                raise ValueError(f"Locked slot for {variable} missing implementation")
            return self._resolve_locked_implementation(
                variable, slot_config.implementation
            )
        elif slot_config.type == "variable":
            # Variable slot: expand scope to tool classes
            if slot_config.scope is None:
                raise ValueError(f"Variable slot for {variable} missing scope")
            return self._expand_variable_scope(variable, slot_config.scope)
        else:
            raise ValueError(f"Unknown slot type: {slot_config.type}")

    def _resolve_locked_implementation(
        self, variable: str, implementation: str
    ) -> list[type[ExternalTool]]:
        """Resolve a locked implementation name to the specific tool class.

        Args:
            variable: The variable type
            implementation: Implementation name (e.g., "animately-advanced", "ffmpeg-color")

        Returns:
            Single-element list containing the matched tool class

        Raises:
            ValueError: If implementation not found
        """
        if not implementation:
            raise ValueError("Locked slot must specify an implementation")

        # Get all available tools for this variable
        available_tools = self._get_tools_for_variable(variable)

        # Find tool with matching NAME
        for tool_cls in available_tools:
            if tool_cls.NAME == implementation:
                self.logger.debug(f"🔒 Locked {variable} to {implementation}")
                return [tool_cls]

        # Implementation not found
        available_names = [tool.NAME for tool in available_tools]
        raise ValueError(
            f"Implementation '{implementation}' not found for {variable}. "
            f"Available: {available_names}"
        )

    def _expand_variable_scope(
        self, variable: str, scope: list[str]
    ) -> list[type[ExternalTool]]:
        """Expand a variable scope to the actual tool classes.

        Args:
            variable: The variable type
            scope: Scope specification (["*"] for all, or list of specific tool names)

        Returns:
            List of tool classes matching the scope

        Raises:
            RuntimeError: If no tools found in scope
        """
        if not scope:
            raise ValueError("Variable slot must specify a scope")

        available_tools = self._get_tools_for_variable(variable)

        if "*" in scope:
            # Wildcard: use all available tools
            self.logger.debug(
                f"🔍 Variable {variable} expanded to all {len(available_tools)} tools"
            )
            return available_tools

        # Specific tool names: filter available tools
        matched_tools = []
        for tool_name in scope:
            for tool_cls in available_tools:
                if tool_cls.NAME == tool_name:
                    matched_tools.append(tool_cls)
                    break
            else:
                # Tool name not found
                available_names = [tool.NAME for tool in available_tools]
                self.logger.warning(
                    f"Tool '{tool_name}' not found for {variable}. "
                    f"Available: {available_names}"
                )

        if not matched_tools:
            available_names = [tool.NAME for tool in available_tools]
            raise RuntimeError(
                f"No tools found in scope {scope} for {variable}. "
                f"Available: {available_names}"
            )

        self.logger.debug(
            f"🔍 Variable {variable} matched {len(matched_tools)} tools from scope"
        )
        return matched_tools

    def _get_tools_for_variable(self, variable: str) -> list[type[ExternalTool]]:
        """Get available tools for a variable with caching.

        Args:
            variable: The variable type

        Returns:
            List of available tool classes for this variable
        """
        if variable not in self._tool_cache:
            self._tool_cache[variable] = tools_for(variable)

        return self._tool_cache[variable]

    def validate_preset_feasibility(self, preset: ExperimentPreset) -> dict[str, Any]:
        """Validate that a preset can generate pipelines and estimate resource requirements.

        Args:
            preset: Experiment preset to validate

        Returns:
            Dictionary with validation results and resource estimates
        """
        results: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "estimated_pipelines": 0,
            "tool_availability": {},
            "efficiency_gain": 0.0,
        }

        try:
            # Check tool availability for each slot
            for slot_name, variable in [
                ("frame", "frame_reduction"),
                ("color", "color_reduction"),
                ("lossy", "lossy_compression"),
            ]:
                slot_config = getattr(preset, f"{slot_name}_slot")

                try:
                    tools = self._resolve_slot_tools(variable, slot_config)
                    results["tool_availability"][slot_name] = {
                        "available": len(tools),
                        "tools": [t.NAME for t in tools],
                    }
                except (ValueError, RuntimeError) as e:
                    results["valid"] = False
                    results["errors"].append(f"{slot_name} slot: {str(e)}")

            if results["valid"]:
                # Calculate efficiency gains vs generate_all_pipelines
                estimated_count = preset.estimate_pipeline_count()
                results["estimated_pipelines"] = estimated_count

                # Rough estimate: generate_all_pipelines creates ~935 combinations
                baseline_count = 935  # 5 frame × 17 color × 11 lossy (typical)
                if estimated_count < baseline_count:
                    results["efficiency_gain"] = 1.0 - (
                        estimated_count / baseline_count
                    )

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {str(e)}")

        return results


def generate_targeted_pipelines(preset: ExperimentPreset) -> list[Pipeline]:
    """Generate targeted pipelines for an experiment preset.

    This is the main entry point function that replaces generate_all_pipelines()
    + sampling workflows with direct targeted generation.

    Args:
        preset: Experiment preset defining the pipeline generation strategy

    Returns:
        List of Pipeline objects for the specific combinations needed

    Example:
        # Frame removal study
        preset = PRESET_REGISTRY.get("frame-focus")
        pipelines = generate_targeted_pipelines(preset)
        # Returns ~5 pipelines instead of 935
    """
    generator = TargetedPipelineGenerator()
    return generator.generate_targeted_pipelines(preset)
