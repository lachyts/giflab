"""Tests for CLI commands using click.testing.CliRunner."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from giflab.cli import (
    debug_failures,
    main,
    organize_directories,
    run,
    select_pipelines,
    tag,
)


class TestMainCLI:
    """Tests for main CLI group."""

    def test_main_help(self):
        """Test main CLI help command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "🎞️ GifLab — GIF compression and analysis laboratory." in result.output
        assert "Commands:" in result.output

    def test_main_version(self):
        """Test main CLI version command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "giflab, version 0.1.0" in result.output

    def test_main_invalid_command(self):
        """Test main CLI with invalid command."""
        runner = CliRunner()
        result = runner.invoke(main, ["invalid-command"])

        assert result.exit_code == 2
        assert "No such command" in result.output


class TestRunCommand:
    """Tests for run CLI command."""

    def test_run_help(self):
        """Test run command help."""
        runner = CliRunner()
        result = runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "Run comprehensive GIF compression analysis and optimization" in result.output
        assert "--workers" in result.output
        assert "--resume" in result.output

    def test_run_missing_directory(self):
        """Test run command with missing directory."""
        runner = CliRunner()
        result = runner.invoke(run, ["/nonexistent/directory"])

        assert result.exit_code != 0
        # Should fail because directory doesn't exist

    @patch("giflab.core.runner.GifLabRunner")
    @patch("giflab.cli.utils.validate_and_get_raw_dir")
    def test_run_estimate_time_mode(self, mock_validate, mock_pipeline_class):
        """Test run command in estimate-time mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Mock validation to pass
            mock_validate.return_value = test_dir

            # Mock pipeline runner
            mock_runner = MagicMock()
            mock_pipeline_class.return_value = mock_runner
            mock_runner.generate_synthetic_gifs.return_value = []
            mock_runner._estimate_execution_time.return_value = "1 minute"

            runner = CliRunner()
            result = runner.invoke(run, [str(test_dir), "--estimate-time"])

            # Should succeed in estimate-time mode
            assert result.exit_code == 0
            assert "Estimated time:" in result.output
            # Analysis should not be called in estimate-time mode
            mock_runner.run_analysis.assert_not_called()

    @patch("giflab.cli.run_cmd.validate_and_get_worker_count")
    @patch("giflab.core.runner.GifLabRunner")
    @patch("giflab.cli.utils.validate_and_get_raw_dir")
    def test_run_with_workers_option(self, mock_validate, mock_runner_class, mock_worker_validate):
        """Test run command with workers option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            mock_validate.return_value = test_dir
            mock_worker_validate.return_value = 4

            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            mock_runner.generate_synthetic_gifs.return_value = []
            mock_runner._estimate_execution_time.return_value = "1 minute"

            runner = CliRunner()
            result = runner.invoke(run, [str(test_dir), "--workers", "4", "--estimate-time"])

            # Check that worker validation was called with correct value
            mock_worker_validate.assert_called_once_with(4)
            assert result.exit_code == 0


class TestTagCommand:
    """Tests for tag CLI command."""

    def test_tag_help(self):
        """Test tag command help."""
        runner = CliRunner()
        result = runner.invoke(tag, ["--help"])

        assert result.exit_code == 0
        assert (
            "Add comprehensive tagging scores to existing compression results"
            in result.output
        )
        assert "--workers" in result.output
        assert "--validate-only" in result.output

    def test_tag_missing_csv(self):
        """Test tag command with missing CSV file."""
        runner = CliRunner()
        result = runner.invoke(tag, ["/nonexistent/file.csv", "/some/raw/dir"])

        assert result.exit_code != 0
        # Should fail because CSV file doesn't exist

    @patch("giflab.tag_pipeline.TaggingPipeline")
    @patch("giflab.io.read_csv_as_dicts")
    def test_tag_validate_only(self, mock_read_csv, mock_tagging_pipeline):
        """Test tag command in validate-only mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test.csv"
            csv_file.write_text("gif_sha,orig_filename\nabc123,test.gif\n")

            # Create a dummy GIF file so the directory validation passes
            dummy_gif = Path(tmpdir) / "test.gif"
            dummy_gif.write_bytes(b"GIF89a")  # Minimal GIF header

            # Mock CSV reading
            mock_read_csv.return_value = [
                {"gif_sha": "abc123", "orig_filename": "test.gif"}
            ]

            runner = CliRunner()
            # Use the temporary directory as the raw directory so it exists
            result = runner.invoke(tag, [str(csv_file), str(tmpdir), "--validate-only"])

            # Should run validation without creating tagging pipeline
            assert "🔍 Validation mode" in result.output
            mock_tagging_pipeline.assert_not_called()


class TestRunCommandAdvanced:
    """Tests for advanced run CLI command features."""

    def test_run_help(self):
        """Test run command help."""
        runner = CliRunner()
        result = runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert (
            "Run comprehensive GIF compression analysis and optimization"
            in result.output
        )
        assert "--sampling" in result.output
        assert "--threshold" in result.output


class TestOrganizeDirectoriesCommand:
    """Tests for organize-directories CLI command."""

    def test_organize_help(self):
        """Test organize-directories command help."""
        runner = CliRunner()
        result = runner.invoke(organize_directories, ["--help"])

        assert result.exit_code == 0
        assert (
            "Create organized directory structure for source-based GIF collection"
            in result.output
        )

    def test_organize_missing_directory(self):
        """Test organize command with missing directory."""
        runner = CliRunner()
        result = runner.invoke(organize_directories, ["/nonexistent/directory"])

        assert result.exit_code != 0
        # Should fail because directory doesn't exist


class TestSelectPipelinesCommand:
    """Tests for select-pipelines CLI command."""

    def test_select_pipelines_help(self):
        """Test select-pipelines command help."""
        runner = CliRunner()
        result = runner.invoke(select_pipelines, ["--help"])

        assert result.exit_code == 0
        assert (
            "Pick the best pipelines from an experiment CSV and write a YAML list"
            in result.output
        )
        assert "--metric" in result.output
        assert "--top" in result.output


class TestDebugFailuresCommand:
    """Tests for debug-failures CLI command."""

    def test_debug_failures_help(self):
        """Test debug-failures command help."""
        runner = CliRunner()
        result = runner.invoke(debug_failures, ["--help"])

        assert result.exit_code == 0
        assert (
            "Debug pipeline elimination failures using the cached failure database"
            in result.output
        )
        assert "--error-type" in result.output

    def test_debug_failures_missing_cache(self):
        """Test debug command with missing cache directory."""
        runner = CliRunner()
        result = runner.invoke(debug_failures, ["--cache-dir", "/nonexistent/cache"])

        # Should handle missing cache gracefully
        assert "No failures found" in result.output or result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_all_commands_have_help(self):
        """Test that all CLI commands provide help."""
        runner = CliRunner()
        commands = [
            "run",
            "tag",
            "organize-directories",
            "select-pipelines",
            "view-failures",
            "debug-failures",
        ]

        for cmd in commands:
            result = runner.invoke(main, [cmd, "--help"])
            assert result.exit_code == 0, f"Command {cmd} --help failed"
            assert "Usage:" in result.output, f"Command {cmd} help missing usage"

    def test_command_discovery(self):
        """Test that all expected commands are registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0

        expected_commands = [
            "run",
            "tag",
            "organize-directories",
            "select-pipelines",
            "view-failures",
            "debug-failures",
        ]

        for cmd in expected_commands:
            assert cmd in result.output, f"Command {cmd} not found in help"

    @patch("giflab.core.runner.GifLabRunner")
    @patch("giflab.cli.utils.validate_and_get_raw_dir")
    def test_keyboard_interrupt_handling(self, mock_validate, mock_pipeline_class):
        """Test that CLI handles KeyboardInterrupt gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            mock_validate.return_value = test_dir

            # Mock pipeline to raise KeyboardInterrupt
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.run.side_effect = KeyboardInterrupt()

            runner = CliRunner()
            result = runner.invoke(run, [str(test_dir)])

            # Should handle KeyboardInterrupt gracefully
            assert "interrupted" in result.output.lower() or result.exit_code != 0


@pytest.mark.fast
class TestCLIFast:
    """Fast CLI tests without external dependencies."""

    def test_main_cli_structure(self):
        """Test main CLI structure and imports."""
        # Test imports work
        from giflab.cli import main

        assert main is not None

        # Test main is a click group
        assert hasattr(main, "commands")
        assert len(main.commands) > 0

    def test_command_registration(self):
        """Test that commands are properly registered."""
        from giflab.cli import main

        expected_commands = {
            "run",
            "tag",
            "organize-directories",
            "select-pipelines",
            "view-failures",
            "debug-failures",
        }

        registered_commands = set(main.commands.keys())
        assert expected_commands.issubset(registered_commands)

    def test_click_context_handling(self):
        """Test Click context handling in commands."""
        runner = CliRunner()

        # Test invalid arguments are caught
        result = runner.invoke(main, ["run", "--invalid-flag"])
        assert result.exit_code != 0
        assert "no such option" in result.output.lower()

    def test_help_messages_contain_emojis(self):
        """Test that help messages contain expected emojis."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "🎞️" in result.output  # Main CLI emoji

    def test_version_option(self):
        """Test version option format."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        # Should contain program name and version
        assert "giflab" in result.output
        assert "0.1.0" in result.output
