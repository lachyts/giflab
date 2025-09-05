"""Tests for the view-failures CLI command."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from giflab.cli import main


class TestViewFailuresCommand:
    """Tests for the view-failures CLI command."""

    @pytest.fixture
    def sample_failed_pipelines(self, tmp_path):
        """Create sample failed pipelines data for testing."""
        failed_pipelines = [
            {
                "pipeline_id": "test_pipeline_1",
                "gif_name": "test_image",
                "content_type": "geometric_patterns",
                "error_message": "gifski frame size mismatch",
                "tools_used": ["imagemagick", "gifski"],
                "error_timestamp": "2024-01-01T12:00:00",
                "test_parameters": {"colors": 64, "lossy": 30, "frame_ratio": 0.8},
            },
            {
                "pipeline_id": "test_pipeline_2",
                "gif_name": "another_test",
                "content_type": "animation_heavy",
                "error_message": "animately command failed",
                "tools_used": ["animately"],
                "error_timestamp": "2024-01-01T12:01:00",
                "test_parameters": {"colors": 128, "lossy": 50, "frame_ratio": 1.0},
            },
            {
                "pipeline_id": "test_pipeline_3",
                "gif_name": "timeout_test",
                "content_type": "large_animation",
                "error_message": "Pipeline timeout after 300 seconds",
                "tools_used": ["ffmpeg"],
                "error_timestamp": "2024-01-01T12:02:00",
                "test_parameters": {"colors": 256, "lossy": 0, "frame_ratio": 0.5},
            },
        ]

        failed_pipelines_file = tmp_path / "failed_pipelines.json"
        with open(failed_pipelines_file, "w") as f:
            json.dump(failed_pipelines, f, indent=2)

        return tmp_path, failed_pipelines

    def test_view_failures_basic(self, sample_failed_pipelines):
        """Test basic view-failures command functionality."""
        results_dir, expected_failures = sample_failed_pipelines

        runner = CliRunner()
        result = runner.invoke(main, ["view-failures", str(results_dir)])

        assert result.exit_code == 0
        assert "Total failures: 3" in result.output
        assert "test_pipeline_1" in result.output
        assert "gifski frame size mismatch" in result.output

    def test_view_failures_filter_by_error_type(self, sample_failed_pipelines):
        """Test filtering failures by error type."""
        results_dir, expected_failures = sample_failed_pipelines

        runner = CliRunner()
        result = runner.invoke(
            main, ["view-failures", str(results_dir), "--error-type", "gifski"]
        )

        assert result.exit_code == 0
        assert "gifski: 1" in result.output
        assert "test_pipeline_1" in result.output
        assert "test_pipeline_2" not in result.output  # Should be filtered out

    def test_view_failures_limit(self, sample_failed_pipelines):
        """Test limiting the number of failures shown."""
        results_dir, expected_failures = sample_failed_pipelines

        runner = CliRunner()
        result = runner.invoke(
            main, ["view-failures", str(results_dir), "--limit", "2"]
        )

        assert result.exit_code == 0
        assert "Total failures: 3" in result.output  # Total count
        # Should only show first 2 failures in detail

    def test_view_failures_detailed(self, sample_failed_pipelines):
        """Test detailed output mode."""
        results_dir, expected_failures = sample_failed_pipelines

        runner = CliRunner()
        result = runner.invoke(main, ["view-failures", str(results_dir), "--detailed"])

        assert result.exit_code == 0
        assert "colors=64" in result.output  # Should show test parameters
        assert "2024-01-01T12:00:00" in result.output  # Should show timestamp

    def test_view_failures_no_file(self, tmp_path):
        """Test handling when failed_pipelines.json doesn't exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, ["view-failures", str(empty_dir)])

        assert result.exit_code == 0
        assert "No failed pipelines file found at:" in result.output

    def test_view_failures_invalid_directory(self):
        """Test handling of non-existent directory."""
        runner = CliRunner()
        result = runner.invoke(main, ["view-failures", "/non/existent/directory"])

        assert result.exit_code != 0  # Should fail with invalid directory

    def test_view_failures_error_type_breakdown(self, sample_failed_pipelines):
        """Test error type breakdown display."""
        results_dir, expected_failures = sample_failed_pipelines

        runner = CliRunner()
        result = runner.invoke(main, ["view-failures", str(results_dir)])

        assert result.exit_code == 0
        assert "Error type breakdown:" in result.output
        assert "gifski: 1" in result.output
        assert "animately: 1" in result.output
        assert "timeout: 1" in result.output
