"""Integration tests for preset CLI functionality."""


import pytest
from click.testing import CliRunner

from src.giflab.cli.run_cmd import run
from src.giflab.core.builtin_presets import PRESET_REGISTRY


class TestPresetCLIIntegration:
    """Test CLI integration for experiment presets."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    def test_preset_option_validation(self, runner):
        """Test that preset option validates correctly."""
        # Valid preset should be accepted (but may fail later due to missing dependencies)
        result = runner.invoke(run, ["--preset", "frame-focus", "--estimate-time"])
        # Should not fail due to invalid preset name (may fail for other reasons)
        assert "Invalid preset" not in result.output

        # Invalid preset should be rejected with error message
        with runner.isolated_filesystem():
            # Create a directory with a test GIF for validation
            from pathlib import Path

            from PIL import Image

            test_dir = Path("test_gifs")
            test_dir.mkdir()
            test_gif = test_dir / "test.gif"

            # Create a minimal test GIF
            img = Image.new("RGB", (10, 10), (255, 0, 0))
            img.save(test_gif)

            result = runner.invoke(run, [str(test_dir), "--preset", "invalid-preset"])
            assert result.exit_code == 0  # Current CLI returns 0 but shows error
            assert (
                "invalid-preset" in result.output and "Unknown preset" in result.output
            )

    def test_preset_help_includes_choices(self, runner):
        """Test that help text includes preset option and --list-presets command."""
        result = runner.invoke(run, ["--help"])
        assert result.exit_code == 0

        # Should include preset option
        assert "--preset" in result.output

        # Should mention --list-presets for seeing available presets
        assert "--list-presets" in result.output

        # Test that --list-presets actually shows the presets
        result = runner.invoke(run, ["--list-presets"])
        assert result.exit_code == 0
        for preset_name in PRESET_REGISTRY.list_presets().keys():
            assert preset_name in result.output
