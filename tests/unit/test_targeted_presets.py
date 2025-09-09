"""Unit tests for the targeted experiment preset system.

This module tests the new slot-based preset system that replaces
generate_all_pipelines() + sampling with targeted pipeline generation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.core.targeted_generator import TargetedPipelineGenerator
from giflab.core.targeted_presets import (
    PRESET_REGISTRY,
    ExperimentPreset,
    PresetRegistry,
    SlotConfiguration,
)


class TestSlotConfiguration:
    """Test SlotConfiguration validation and functionality."""

    def test_valid_variable_slot_creation(self):
        """Test creating a valid variable slot succeeds."""
        slot = SlotConfiguration(
            type="variable", scope=["*"], parameters={"ratios": [1.0, 0.8, 0.5]}
        )
        assert slot.type == "variable"
        assert slot.scope == ["*"]
        assert slot.implementation is None
        assert slot.parameters == {"ratios": [1.0, 0.8, 0.5]}

    def test_valid_locked_slot_creation(self):
        """Test creating a valid locked slot succeeds."""
        slot = SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        )
        assert slot.type == "locked"
        assert slot.implementation == "ffmpeg-color"
        assert slot.scope is None
        assert slot.parameters == {"colors": 32}

    def test_variable_slot_missing_scope(self):
        """Test that variable slot without scope raises ValueError."""
        with pytest.raises(ValueError, match="Variable slots must specify a scope"):
            SlotConfiguration(type="variable", implementation="should-not-have-this")

    def test_variable_slot_with_implementation(self):
        """Test that variable slot with implementation raises ValueError."""
        with pytest.raises(
            ValueError, match="Variable slots cannot have an implementation"
        ):
            SlotConfiguration(
                type="variable", scope=["*"], implementation="should-not-have-this"
            )

    def test_locked_slot_missing_implementation(self):
        """Test that locked slot without implementation raises ValueError."""
        with pytest.raises(
            ValueError, match="Locked slots must specify an implementation"
        ):
            SlotConfiguration(type="locked", scope=["should-not-have-this"])

    def test_locked_slot_with_scope(self):
        """Test that locked slot with scope raises ValueError."""
        with pytest.raises(ValueError, match="Locked slots cannot have a scope"):
            SlotConfiguration(
                type="locked",
                implementation="ffmpeg-color",
                scope=["should-not-have-this"],
            )

    def test_invalid_slot_type(self):
        """Test that invalid slot type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid slot type: invalid"):
            SlotConfiguration(type="invalid", implementation="test")

    def test_default_parameters(self):
        """Test that parameters default to empty dict."""
        slot = SlotConfiguration(type="locked", implementation="test-tool")
        assert slot.parameters == {}


class TestExperimentPreset:
    """Test ExperimentPreset validation and functionality."""

    def test_valid_preset_creation(self):
        """Test creating a valid preset succeeds."""
        preset = ExperimentPreset(
            name="Test Preset",
            description="Test description",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        assert preset.name == "Test Preset"
        assert preset.description == "Test description"
        assert preset.frame_slot.type == "variable"
        assert preset.color_slot.type == "locked"
        assert preset.lossy_slot.type == "locked"

    def test_preset_with_all_slots_locked(self):
        """Test that preset with all locked slots raises ValueError."""
        with pytest.raises(ValueError, match="At least one slot must be variable"):
            ExperimentPreset(
                name="All Locked",
                description="Invalid preset",
                frame_slot=SlotConfiguration(
                    type="locked", implementation="animately-frame"
                ),
                color_slot=SlotConfiguration(
                    type="locked", implementation="ffmpeg-color"
                ),
                lossy_slot=SlotConfiguration(
                    type="locked", implementation="none-lossy"
                ),
            )

    def test_preset_with_invalid_sampling_strategy(self):
        """Test that invalid sampling strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid sampling strategy"):
            ExperimentPreset(
                name="Invalid Sampling",
                description="Test preset",
                frame_slot=SlotConfiguration(type="variable", scope=["*"]),
                color_slot=SlotConfiguration(
                    type="locked", implementation="ffmpeg-color"
                ),
                lossy_slot=SlotConfiguration(
                    type="locked", implementation="none-lossy"
                ),
                custom_sampling="invalid_strategy",
            )

    def test_preset_with_zero_max_combinations(self):
        """Test that zero max_combinations raises ValueError."""
        with pytest.raises(ValueError, match="max_combinations must be positive"):
            ExperimentPreset(
                name="Zero Max",
                description="Test preset",
                frame_slot=SlotConfiguration(type="variable", scope=["*"]),
                color_slot=SlotConfiguration(
                    type="locked", implementation="ffmpeg-color"
                ),
                lossy_slot=SlotConfiguration(
                    type="locked", implementation="none-lossy"
                ),
                max_combinations=0,
            )

    def test_preset_metadata_defaults(self):
        """Test preset metadata has correct defaults."""
        preset = ExperimentPreset(
            name="Test",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        assert preset.tags == []
        assert preset.author is None
        assert preset.version == "1.0"
        assert preset.custom_sampling is None
        assert preset.max_combinations is None

    def test_get_variable_slots(self):
        """Test get_variable_slots method."""
        preset = ExperimentPreset(
            name="Test",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(
                type="variable", scope=["ffmpeg-color", "gifsicle-color"]
            ),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        variable_slots = preset.get_variable_slots()
        assert set(variable_slots) == {"frame", "color"}

    def test_get_locked_slots(self):
        """Test get_locked_slots method."""
        preset = ExperimentPreset(
            name="Test",
            description="Test",
            frame_slot=SlotConfiguration(
                type="locked", implementation="animately-frame"
            ),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        locked_slots = preset.get_locked_slots()
        assert set(locked_slots.keys()) == {"frame", "lossy"}
        assert locked_slots["frame"].implementation == "animately-frame"
        assert locked_slots["lossy"].implementation == "none-lossy"

    def test_estimate_pipeline_count_single_variable(self):
        """Test pipeline count estimation for single variable slot."""
        preset = ExperimentPreset(
            name="Frame Focus",
            description="Test frame algorithms",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        count = preset.estimate_pipeline_count()
        assert count == 5  # 5 frame tools × 1 color × 1 lossy

    def test_estimate_pipeline_count_multiple_variables(self):
        """Test pipeline count estimation for multiple variable slots."""
        preset = ExperimentPreset(
            name="Multi Variable",
            description="Test multiple variables",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        count = preset.estimate_pipeline_count()
        assert count == 85  # 5 frame × 17 color × 1 lossy

    def test_estimate_with_max_combinations_limit(self):
        """Test estimation respects max_combinations limit."""
        preset = ExperimentPreset(
            name="Limited",
            description="Test with limit",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="variable", scope=["*"]),
            max_combinations=100,
        )
        count = preset.estimate_pipeline_count()
        assert count == 100  # Limited to 100 instead of 5×17×11=935

    def test_estimate_with_specific_scope(self):
        """Test estimation with specific tool scope."""
        preset = ExperimentPreset(
            name="Specific Tools",
            description="Test specific tools",
            frame_slot=SlotConfiguration(
                type="variable", scope=["animately-frame", "ffmpeg-frame"]
            ),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        count = preset.estimate_pipeline_count()
        assert count == 2  # 2 specific frame tools × 1 color × 1 lossy


class TestPresetRegistry:
    """Test PresetRegistry functionality."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = PresetRegistry()
        assert isinstance(registry.presets, dict)
        assert len(registry.presets) == 0  # Empty on initialization

    def test_register_valid_preset(self):
        """Test registering a valid preset succeeds."""
        registry = PresetRegistry()
        preset = ExperimentPreset(
            name="Test Preset",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        registry.register("test-preset", preset)
        assert "test-preset" in registry.presets
        assert registry.presets["test-preset"] == preset

    def test_register_empty_preset_id(self):
        """Test that empty preset ID raises ValueError."""
        registry = PresetRegistry()
        preset = ExperimentPreset(
            name="Test",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        with pytest.raises(ValueError, match="Preset ID cannot be empty"):
            registry.register("", preset)

    def test_register_duplicate_preset_logs_warning(self):
        """Test that registering duplicate preset logs warning."""
        registry = PresetRegistry()
        preset1 = ExperimentPreset(
            name="Original",
            description="First preset",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        preset2 = ExperimentPreset(
            name="Replacement",
            description="Second preset",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(
                type="locked", implementation="gifsicle-color"
            ),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        registry.register("test-preset", preset1)

        # This should log a warning but still succeed
        registry.register("test-preset", preset2)
        assert registry.presets["test-preset"].name == "Replacement"

    def test_get_existing_preset(self):
        """Test getting an existing preset."""
        registry = PresetRegistry()
        preset = ExperimentPreset(
            name="Test Preset",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        registry.register("test-preset", preset)
        retrieved = registry.get("test-preset")
        assert retrieved == preset

    def test_get_nonexistent_preset(self):
        """Test that getting nonexistent preset raises ValueError."""
        registry = PresetRegistry()

        with pytest.raises(ValueError, match="Unknown preset: nonexistent"):
            registry.get("nonexistent")

    def test_list_presets_empty(self):
        """Test listing presets when empty."""
        registry = PresetRegistry()
        presets = registry.list_presets()
        assert presets == {}

    def test_list_presets_with_content(self):
        """Test listing presets with content."""
        registry = PresetRegistry()
        preset1 = ExperimentPreset(
            name="First Preset",
            description="First description",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        preset2 = ExperimentPreset(
            name="Second Preset",
            description="Second description",
            frame_slot=SlotConfiguration(
                type="locked", implementation="animately-frame"
            ),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        registry.register("first", preset1)
        registry.register("second", preset2)

        presets = registry.list_presets()
        assert presets == {"first": "First description", "second": "Second description"}

    def test_find_similar_presets(self):
        """Test finding similar presets."""
        registry = PresetRegistry()

        # Create base preset
        preset1 = ExperimentPreset(
            name="Base Preset",
            description="Base",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        registry.register("base", preset1)

        # Create similar preset (same slot types, different implementation)
        preset2 = ExperimentPreset(
            name="Similar Preset",
            description="Similar",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(
                type="locked", implementation="gifsicle-color"
            ),  # Different implementation
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        similar = registry.find_similar_presets(
            preset2, threshold=0.6
        )  # 2/3 = 0.67 > 0.6
        assert "base" in similar

    def test_calculate_preset_similarity(self):
        """Test preset similarity calculation."""
        registry = PresetRegistry()

        # Identical presets
        preset1 = ExperimentPreset(
            name="Same Name",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )
        preset2 = ExperimentPreset(
            name="Same Name",
            description="Test",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        similarity = registry._calculate_preset_similarity(preset1, preset2)
        assert similarity == 1.0

        # Completely different presets
        preset3 = ExperimentPreset(
            name="Different",
            description="Test",
            frame_slot=SlotConfiguration(
                type="locked", implementation="animately-frame"
            ),
            color_slot=SlotConfiguration(type="variable", scope=["gifsicle-color"]),
            lossy_slot=SlotConfiguration(
                type="variable", scope=["animately-advanced-lossy"]
            ),
        )

        similarity = registry._calculate_preset_similarity(preset1, preset3)
        assert similarity == 0.0  # No slots match


class TestTargetedPipelineGenerator:
    """Test TargetedPipelineGenerator functionality."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = TargetedPipelineGenerator()
        assert generator._tool_cache == {}

    def test_validate_preset_feasibility_valid(self):
        """Test validation of valid preset."""
        generator = TargetedPipelineGenerator()
        preset = ExperimentPreset(
            name="Valid Test",
            description="Valid preset for testing",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        validation = generator.validate_preset_feasibility(preset)

        assert validation["valid"] is True
        assert validation["errors"] == []
        assert validation["estimated_pipelines"] > 0
        assert 0.0 <= validation["efficiency_gain"] <= 1.0
        assert isinstance(validation["tool_availability"], dict)

    def test_validate_preset_feasibility_invalid_tool(self):
        """Test validation with invalid tool name."""
        generator = TargetedPipelineGenerator()
        preset = ExperimentPreset(
            name="Invalid Tool Test",
            description="Preset with invalid tool",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(
                type="locked", implementation="nonexistent-tool"
            ),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        validation = generator.validate_preset_feasibility(preset)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert "nonexistent-tool" in str(validation["errors"])

    def test_resolve_locked_implementation_valid(self):
        """Test resolving valid locked implementation."""
        generator = TargetedPipelineGenerator()

        tools = generator._resolve_locked_implementation(
            "color_reduction", "ffmpeg-color"
        )

        assert len(tools) == 1
        assert tools[0].NAME == "ffmpeg-color"

    def test_resolve_locked_implementation_invalid(self):
        """Test resolving invalid locked implementation raises error."""
        generator = TargetedPipelineGenerator()

        with pytest.raises(ValueError, match="Implementation 'nonexistent' not found"):
            generator._resolve_locked_implementation("color_reduction", "nonexistent")

    def test_expand_variable_scope_wildcard(self):
        """Test expanding wildcard variable scope."""
        generator = TargetedPipelineGenerator()

        tools = generator._expand_variable_scope("frame_reduction", ["*"])

        assert len(tools) > 0  # Should find frame reduction tools
        tool_names = [tool.NAME for tool in tools]
        assert "animately-frame" in tool_names  # Should include common tools

    def test_expand_variable_scope_specific(self):
        """Test expanding specific tool list scope."""
        generator = TargetedPipelineGenerator()

        tools = generator._expand_variable_scope(
            "color_reduction", ["ffmpeg-color", "gifsicle-color"]
        )

        assert len(tools) == 2
        tool_names = [tool.NAME for tool in tools]
        assert set(tool_names) == {"ffmpeg-color", "gifsicle-color"}

    def test_expand_variable_scope_empty(self):
        """Test that empty scope raises error."""
        generator = TargetedPipelineGenerator()

        with pytest.raises(ValueError, match="Variable slot must specify a scope"):
            generator._expand_variable_scope("frame_reduction", [])

    def test_tool_caching(self):
        """Test that tool resolution uses caching."""
        generator = TargetedPipelineGenerator()

        # First call should populate cache
        tools1 = generator._get_tools_for_variable("frame_reduction")
        assert "frame_reduction" in generator._tool_cache

        # Second call should use cache
        tools2 = generator._get_tools_for_variable("frame_reduction")
        assert tools1 == tools2  # Same result from cache


class TestBuiltinPresets:
    """Test built-in preset definitions."""

    def test_builtin_presets_loaded(self):
        """Test that built-in presets are loaded."""
        # Import should trigger auto-registration

        presets = PRESET_REGISTRY.list_presets()
        assert len(presets) > 0
        assert "frame-focus" in presets
        assert "color-optimization" in presets

    def test_all_builtin_presets_valid(self):
        """Test that all built-in presets are valid."""

        generator = TargetedPipelineGenerator()

        for preset_id in PRESET_REGISTRY.list_presets().keys():
            preset = PRESET_REGISTRY.get(preset_id)
            validation = generator.validate_preset_feasibility(preset)

            assert validation[
                "valid"
            ], f"Preset {preset_id} is invalid: {validation['errors']}"
            assert validation["estimated_pipelines"] > 0

    def test_builtin_presets_generate_pipelines(self):
        """Test that built-in presets can generate pipelines."""

        generator = TargetedPipelineGenerator()

        # Test a few key presets
        test_presets = ["frame-focus", "color-optimization", "quick-test"]

        for preset_id in test_presets:
            if preset_id in PRESET_REGISTRY.list_presets():
                preset = PRESET_REGISTRY.get(preset_id)
                pipelines = generator.generate_targeted_pipelines(preset)

                assert len(pipelines) > 0, f"Preset {preset_id} generated no pipelines"
                assert all(
                    hasattr(p, "identifier") for p in pipelines
                ), f"Invalid pipeline objects from {preset_id}"

    def test_frame_focus_preset_configuration(self):
        """Test frame-focus preset has correct configuration."""

        if "frame-focus" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("frame-focus")

            assert preset.frame_slot.type == "variable"
            assert preset.color_slot.type == "locked"
            assert preset.lossy_slot.type == "locked"
            assert "frame" in preset.get_variable_slots()
            assert len(preset.get_variable_slots()) == 1

    def test_color_optimization_preset_configuration(self):
        """Test color-optimization preset has correct configuration."""

        if "color-optimization" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("color-optimization")

            assert preset.frame_slot.type == "locked"
            assert preset.color_slot.type == "variable"
            assert preset.lossy_slot.type == "locked"
            assert "color" in preset.get_variable_slots()
            assert len(preset.get_variable_slots()) == 1

    def test_tool_comparison_baseline_preset(self):
        """Test tool-comparison-baseline preset has multiple variables."""

        if "tool-comparison-baseline" in PRESET_REGISTRY.list_presets():
            preset = PRESET_REGISTRY.get("tool-comparison-baseline")

            variable_slots = preset.get_variable_slots()
            assert (
                len(variable_slots) >= 2
            ), "Tool comparison should vary multiple dimensions"

    def test_preset_efficiency_gains(self):
        """Test that presets achieve expected efficiency gains."""

        generator = TargetedPipelineGenerator()
        baseline_pipelines = 935  # Typical generate_all_pipelines count

        for preset_id in PRESET_REGISTRY.list_presets().keys():
            preset = PRESET_REGISTRY.get(preset_id)
            validation = generator.validate_preset_feasibility(preset)

            if validation["valid"]:
                efficiency_gain = validation["efficiency_gain"]
                estimated_count = validation["estimated_pipelines"]

                # Most presets should achieve significant efficiency gains
                if (
                    preset_id != "tool-comparison-baseline"
                ):  # Exception for comprehensive preset
                    assert (
                        efficiency_gain > 0.5
                    ), f"Preset {preset_id} has low efficiency gain: {efficiency_gain:.1%}"

                assert (
                    estimated_count < baseline_pipelines
                ), f"Preset {preset_id} estimates more pipelines than baseline"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_slot_configuration_in_preset(self):
        """Test that invalid slot configurations are caught."""
        with pytest.raises(ValueError):
            ExperimentPreset(
                name="Invalid Slot",
                description="Test",
                frame_slot=SlotConfiguration(
                    type="invalid_type", scope=["*"]
                ),  # Invalid type
                color_slot=SlotConfiguration(
                    type="locked", implementation="ffmpeg-color"
                ),
                lossy_slot=SlotConfiguration(
                    type="locked", implementation="none-lossy"
                ),
            )

    def test_generator_with_no_available_tools(self):
        """Test generator behavior when no tools are available."""
        generator = TargetedPipelineGenerator()
        preset = ExperimentPreset(
            name="No Tools Test",
            description="Test with nonexistent tools",
            frame_slot=SlotConfiguration(type="variable", scope=["nonexistent-tool"]),
            color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        with pytest.raises(RuntimeError, match="No tools found in scope"):
            generator.generate_targeted_pipelines(preset)

    def test_registry_validation_catches_invalid_presets(self):
        """Test that preset validation catches invalid presets during construction."""
        PresetRegistry()

        # Should raise during preset construction due to validation (all slots locked)
        with pytest.raises(ValueError, match="At least one slot must be variable"):
            ExperimentPreset(
                name="Invalid",
                description="Should fail",
                frame_slot=SlotConfiguration(
                    type="locked", implementation="animately-frame"
                ),
                color_slot=SlotConfiguration(
                    type="locked", implementation="ffmpeg-color"
                ),
                lossy_slot=SlotConfiguration(
                    type="locked", implementation="none-lossy"
                ),
            )

    def test_empty_pipeline_generation_handling(self):
        """Test handling when pipeline generation results in empty list."""
        generator = TargetedPipelineGenerator()

        # Create preset that would result in no valid combinations
        preset = ExperimentPreset(
            name="Empty Result",
            description="Test empty results",
            frame_slot=SlotConfiguration(type="variable", scope=["nonexistent1"]),
            color_slot=SlotConfiguration(type="variable", scope=["nonexistent2"]),
            lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
        )

        with pytest.raises(RuntimeError):
            generator.generate_targeted_pipelines(preset)

    def test_max_combinations_limit_applied(self):
        """Test that max_combinations limit is properly applied."""

        generator = TargetedPipelineGenerator()

        # Create a preset that would generate many pipelines but limit it
        preset = ExperimentPreset(
            name="Limited Test",
            description="Test max combinations limit",
            frame_slot=SlotConfiguration(type="variable", scope=["*"]),
            color_slot=SlotConfiguration(type="variable", scope=["*"]),
            lossy_slot=SlotConfiguration(type="variable", scope=["*"]),
            max_combinations=5,  # Limit to 5 pipelines
        )

        pipelines = generator.generate_targeted_pipelines(preset)
        assert len(pipelines) == 5  # Should be limited


class TestIntegrationWithExistingSystem:
    """Test integration with existing GifLab systems."""

    def test_pipeline_objects_have_correct_structure(self):
        """Test that generated pipelines have expected structure."""

        generator = TargetedPipelineGenerator()
        preset = PRESET_REGISTRY.get("frame-focus")
        pipelines = generator.generate_targeted_pipelines(preset)

        for pipeline in pipelines[:3]:  # Test first few
            assert hasattr(pipeline, "steps")
            assert hasattr(pipeline, "identifier")
            assert len(pipeline.steps) == 3  # frame, color, lossy

            # Check each step has required attributes
            for step in pipeline.steps:
                assert hasattr(step, "variable")
                assert hasattr(step, "tool_cls")
                assert step.variable in [
                    "frame_reduction",
                    "color_reduction",
                    "lossy_compression",
                ]
                assert hasattr(step.tool_cls, "NAME")

    def test_pipeline_identifiers_unique(self):
        """Test that pipeline identifiers are unique."""

        generator = TargetedPipelineGenerator()
        preset = PRESET_REGISTRY.get("color-optimization")
        pipelines = generator.generate_targeted_pipelines(preset)

        identifiers = [p.identifier() for p in pipelines]
        assert len(identifiers) == len(
            set(identifiers)
        ), "Pipeline identifiers are not unique"

    def test_tool_name_validation_against_capability_registry(self):
        """Test that tool names used in presets exist in capability registry."""
        from giflab.capability_registry import tools_for

        # Get all available tools
        all_tools = {}
        for variable in ["frame_reduction", "color_reduction", "lossy_compression"]:
            all_tools[variable] = {tool.NAME for tool in tools_for(variable)}

        # Check all built-in presets use valid tool names
        for preset_id in PRESET_REGISTRY.list_presets().keys():
            preset = PRESET_REGISTRY.get(preset_id)

            # Check locked implementations
            for slot_name in ["frame", "color", "lossy"]:
                slot = getattr(preset, f"{slot_name}_slot")
                if slot.type == "locked":
                    variable = (
                        f"{slot_name}_reduction"
                        if slot_name != "lossy"
                        else "lossy_compression"
                    )
                    assert (
                        slot.implementation in all_tools[variable]
                    ), f"Preset {preset_id} uses invalid {slot_name} tool: {slot.implementation}"

            # Check variable scopes (for specific tool names, not wildcards)
            for slot_name in ["frame", "color", "lossy"]:
                slot = getattr(preset, f"{slot_name}_slot")
                if slot.type == "variable" and slot.scope and "*" not in slot.scope:
                    variable = (
                        f"{slot_name}_reduction"
                        if slot_name != "lossy"
                        else "lossy_compression"
                    )
                    for tool_name in slot.scope:
                        assert (
                            tool_name in all_tools[variable]
                        ), f"Preset {preset_id} scope includes invalid {slot_name} tool: {tool_name}"


if __name__ == "__main__":
    pytest.main([__file__])
