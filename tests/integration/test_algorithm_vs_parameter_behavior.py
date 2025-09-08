"""Integration tests that verify actual experimental behavior for algorithm vs parameter distinction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.giflab.core.runner import GifLabRunner
from src.giflab.core.targeted_presets import PRESET_REGISTRY
from src.giflab.dynamic_pipeline import Pipeline


class TestAlgorithmVsParameterBehavior:
    """Integration tests that verify the actual experimental behavior."""

    def _create_mock_pipeline(
        self, tool_name: str, frame: float, lossy: int, color: int
    ) -> Pipeline:
        """Create a mock pipeline for testing."""
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.frame_ratio = frame
        mock_pipeline.lossy_level = lossy
        mock_pipeline.color_count = color

        # Mock the steps
        mock_step = MagicMock()
        mock_step.tool_cls.NAME = tool_name
        mock_pipeline.steps = [mock_step]

        return mock_pipeline


class TestPresetExperimentOutcomes:
    """Test that presets actually produce the expected experimental outcomes."""

    @pytest.mark.parametrize(
        "preset_name,expected_type",
        [
            ("frame-focus", "algorithm_comparison"),
            ("custom-gif-frame-study", "algorithm_comparison"),
            ("color-optimization", "algorithm_comparison"),
            ("lossy-quality-sweep", "algorithm_comparison"),
            ("frame-parameter-sweep", "parameter_sweep"),
            ("color-parameter-sweep", "parameter_sweep"),
            ("lossy-parameter-sweep", "parameter_sweep"),
        ],
    )
    def test_preset_type_consistency(self, preset_name: str, expected_type: str):
        """Test that each preset consistently implements its intended type."""
        preset = PRESET_REGISTRY.get(preset_name)

        if expected_type == "algorithm_comparison":
            # Algorithm comparison: multiple variable slots (comparing different algorithms)
            variable_slot_names = preset.get_variable_slots()
            locked_slots = preset.get_locked_slots()

            # Should have at least one variable slot for algorithm comparison
            assert (
                len(variable_slot_names) >= 1
            ), f"Algorithm comparison should vary at least one dimension"

            # Variable slots should have multiple tools in scope (comparing algorithms)
            for slot_name in variable_slot_names:
                slot = getattr(preset, f"{slot_name}_slot")
                if slot.scope != [
                    "*"
                ]:  # If not wildcard, should have multiple specific tools
                    assert len(slot.scope) >= 2 or slot.scope == [
                        "*"
                    ], f"Algorithm comparison slot '{slot_name}' should compare multiple algorithms"

        elif expected_type == "parameter_sweep":
            # Parameter sweep: single algorithm with parameter variation
            variable_slot_names = preset.get_variable_slots()

            # Should have exactly one variable slot for parameter sweep
            assert (
                len(variable_slot_names) == 1
            ), f"Parameter sweep should vary exactly one dimension"

            # Variable slot should be restricted to one algorithm (parameter sweep, not algorithm comparison)
            for slot_name in variable_slot_names:
                slot = getattr(preset, f"{slot_name}_slot")
                assert (
                    len(slot.scope) == 1
                ), f"Parameter sweep slot '{slot_name}' should be limited to one algorithm, got: {slot.scope}"
