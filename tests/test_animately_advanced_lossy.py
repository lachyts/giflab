"""Tests for AnimatelyAdvancedLossyCompressor and PNG sequence optimization."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from giflab.lossy import (
    _execute_animately_advanced,
    _extract_frame_timing,
    _extract_gif_metadata,
    _generate_frame_list,
    _generate_json_config,
    _managed_temp_directory,
    _process_advanced_lossy,
    _setup_png_sequence_directory,
    _validate_animately_availability,
    compress_with_animately_advanced_lossy,
)
from giflab.tool_wrappers import AnimatelyAdvancedLossyCompressor


class TestAnimatelyAdvancedLossyCompressor:
    """Test the new AnimatelyAdvancedLossyCompressor wrapper class."""

    def test_tool_registration(self):
        """Test that the tool is properly registered with correct attributes."""
        tool = AnimatelyAdvancedLossyCompressor()

        assert tool.NAME == "animately-advanced-lossy"
        assert tool.COMBINE_GROUP == "animately"
        assert tool.VARIABLE == "lossy_compression"
        assert hasattr(tool, "available")
        assert hasattr(tool, "version")
        assert hasattr(tool, "apply")

    def test_available_method(self):
        """Test the availability check."""
        with patch("giflab.tool_wrappers._is_executable") as mock_is_executable:
            mock_is_executable.return_value = True
            tool = AnimatelyAdvancedLossyCompressor()
            assert tool.available() is True

            mock_is_executable.return_value = False
            assert tool.available() is False

    def test_version_method(self):
        """Test version retrieval."""
        with patch("giflab.tool_wrappers.get_animately_version") as mock_version:
            mock_version.return_value = "1.1.20.0"
            tool = AnimatelyAdvancedLossyCompressor()
            assert tool.version() == "1.1.20.0"

            mock_version.side_effect = Exception("Failed")
            assert tool.version() == "unknown"

    def test_apply_missing_lossy_level(self):
        """Test that apply raises error when lossy_level is missing."""
        tool = AnimatelyAdvancedLossyCompressor()

        with pytest.raises(ValueError, match="params must include 'lossy_level'"):
            tool.apply(Path("input.gif"), Path("output.gif"), params={})

        with pytest.raises(ValueError, match="params must include 'lossy_level'"):
            tool.apply(Path("input.gif"), Path("output.gif"), params=None)

    @patch("giflab.lossy.compress_with_animately_advanced_lossy")
    def test_apply_basic_params(self, mock_compress):
        """Test apply with basic parameters."""
        mock_compress.return_value = {"render_ms": 100, "engine": "animately-advanced"}

        tool = AnimatelyAdvancedLossyCompressor()
        result = tool.apply(
            Path("input.gif"), Path("output.gif"), params={"lossy_level": 60}
        )

        mock_compress.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            lossy_level=60,
            color_keep_count=None,
            png_sequence_dir=None,
        )
        assert result["engine"] == "animately-advanced"

    @patch("giflab.lossy.compress_with_animately_advanced_lossy")
    def test_apply_with_colors(self, mock_compress):
        """Test apply with color reduction parameter."""
        mock_compress.return_value = {"render_ms": 100, "engine": "animately-advanced"}

        tool = AnimatelyAdvancedLossyCompressor()
        tool.apply(
            Path("input.gif"),
            Path("output.gif"),
            params={"lossy_level": 60, "colors": 32},
        )

        mock_compress.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            lossy_level=60,
            color_keep_count=32,
            png_sequence_dir=None,
        )

    @patch("giflab.lossy.compress_with_animately_advanced_lossy")
    def test_apply_with_png_sequence_dir(self, mock_compress):
        """Test apply with PNG sequence directory provided by pipeline."""
        mock_compress.return_value = {"render_ms": 100, "engine": "animately-advanced"}

        tool = AnimatelyAdvancedLossyCompressor()
        tool.apply(
            Path("input.gif"),
            Path("output.gif"),
            params={
                "lossy_level": 60,
                "colors": 32,
                "png_sequence_dir": "/tmp/png_sequence",
            },
        )

        mock_compress.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            lossy_level=60,
            color_keep_count=32,
            png_sequence_dir=Path("/tmp/png_sequence"),
        )


class TestCompressWithAnimatelyAdvancedLossy:
    """Test the core compress_with_animately_advanced_lossy function."""

    @patch("giflab.lossy._is_executable")
    def test_animately_not_available(self, mock_is_executable):
        """Test error when Animately is not available."""
        mock_is_executable.return_value = False

        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            compress_with_animately_advanced_lossy(
                Path("input.gif"), Path("output.gif"), lossy_level=60
            )

    @patch("giflab.lossy.extract_gif_metadata")
    @patch("giflab.lossy._is_executable")
    def test_metadata_extraction_failure(self, mock_is_executable, mock_extract):
        """Test error when metadata extraction fails."""
        mock_is_executable.return_value = True
        mock_extract.side_effect = Exception("Failed to extract metadata")

        with pytest.raises(RuntimeError, match="Failed to extract metadata"):
            compress_with_animately_advanced_lossy(
                Path("input.gif"), Path("output.gif"), lossy_level=60
            )

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.export_png_sequence")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.validate_path_security")
    def test_successful_compression_without_provided_sequence(
        self, mock_validate, mock_is_executable, mock_extract, mock_export, mock_run
    ):
        """Test successful compression when creating new PNG sequence."""
        # Setup mocks
        mock_is_executable.return_value = True
        mock_validate.side_effect = lambda x: x

        mock_metadata = Mock()
        mock_metadata.orig_frames = 3
        mock_metadata.orig_n_colors = 128
        mock_extract.return_value = mock_metadata

        mock_export.return_value = {
            "frame_count": 3,
            "render_ms": 10,
            "engine": "imagemagick",
        }

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            output_path.touch()  # Create the output file

            # Create some mock PNG files
            png_dir = Path(tmpdir) / "animately_png_test"
            png_dir.mkdir()
            for i in range(3):
                (png_dir / f"frame_{i:04d}.png").touch()

            with patch("tempfile.mkdtemp", return_value=str(png_dir)):
                with patch("PIL.Image.open") as mock_pil:
                    mock_img = Mock()
                    mock_img.seek = Mock()
                    mock_img.info = {"duration": 100}
                    mock_pil.return_value.__enter__.return_value = mock_img

                    result = compress_with_animately_advanced_lossy(
                        Path("input.gif"),
                        output_path,
                        lossy_level=60,
                        color_keep_count=32,
                    )

        # Verify the result
        assert result["engine"] == "animately-advanced"
        assert result["lossy_level"] == 60
        assert result["color_keep_count"] == 32
        assert result["frames_processed"] == 3
        assert "png_sequence_metadata" in result
        assert "json_config_path" in result

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.extract_gif_metadata")
    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.validate_path_security")
    def test_successful_compression_with_provided_sequence(
        self, mock_validate, mock_is_executable, mock_extract, mock_run
    ):
        """Test successful compression when PNG sequence is provided by previous step."""
        # Setup mocks
        mock_is_executable.return_value = True
        mock_validate.side_effect = lambda x: x

        mock_metadata = Mock()
        mock_metadata.orig_frames = 3
        mock_metadata.orig_n_colors = 128
        mock_extract.return_value = mock_metadata

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            output_path.touch()  # Create the output file

            # Create provided PNG sequence directory
            png_dir = Path(tmpdir) / "provided_png_sequence"
            png_dir.mkdir()
            for i in range(3):
                (png_dir / f"frame_{i:04d}.png").touch()

            with patch("PIL.Image.open") as mock_pil:
                mock_img = Mock()
                mock_img.seek = Mock()
                mock_img.info = {"duration": 100}
                mock_pil.return_value.__enter__.return_value = mock_img

                result = compress_with_animately_advanced_lossy(
                    Path("input.gif"),
                    output_path,
                    lossy_level=60,
                    color_keep_count=32,
                    png_sequence_dir=png_dir,  # Provided sequence
                )

        # Verify the result uses provided sequence
        assert result["engine"] == "animately-advanced"
        assert result["png_sequence_metadata"]["engine"] == "provided_by_previous_step"
        assert result["png_sequence_metadata"]["render_ms"] == 0  # No export time


class TestPNGSequenceOptimization:
    """Test PNG sequence optimization in pipeline elimination."""

    def test_optimization_detection(self):
        """Test that PNG-capable tools are correctly identified."""
        from giflab.core import GifLabRunner

        # Create eliminator instance
        GifLabRunner()

        # Test the PNG capability detection logic
        png_capable_tools = [
            "FFmpegFrameReducer",
            "FFmpegColorReducer",
            "ImageMagickFrameReducer",
            "ImageMagickColorReducer",
            "AnimatelyFrameReducer",
            "AnimatelyColorReducer",
        ]

        for tool_name in png_capable_tools:
            supports_png_export = (
                "FFmpegFrameReducer" in tool_name
                or "FFmpegColorReducer" in tool_name
                or "ImageMagickFrameReducer" in tool_name
                or "ImageMagickColorReducer" in tool_name
                or "AnimatelyFrameReducer" in tool_name
                or "AnimatelyColorReducer" in tool_name
            )
            assert supports_png_export is True, f"{tool_name} should support PNG export"

    def test_pipeline_generation_includes_advanced_lossy(self):
        """Test that pipeline generation includes the new advanced lossy compressor."""
        from giflab.dynamic_pipeline import generate_all_pipelines

        all_pipelines = generate_all_pipelines()
        advanced_pipelines = [
            p
            for p in all_pipelines
            if any(
                step.tool_cls.__name__ == "AnimatelyAdvancedLossyCompressor"
                for step in p.steps
            )
        ]

        assert (
            len(advanced_pipelines) > 0
        ), "Should have pipelines with AnimatelyAdvancedLossyCompressor"
        assert (
            len(advanced_pipelines) >= 5
        ), f"Should have multiple advanced pipelines, found {len(advanced_pipelines)}"


class TestManagedTempDirectory:
    """Test the context manager for temporary directory management."""

    def test_creates_and_cleans_up_directory(self):
        """Test that directory is created and cleaned up properly."""
        created_path = None

        with _managed_temp_directory("test_prefix_") as temp_dir:
            created_path = temp_dir
            assert temp_dir.exists()
            assert "test_prefix_" in temp_dir.name

            # Create a test file to verify cleanup
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()

        # Directory should be cleaned up after exiting context
        assert not created_path.exists()

    def test_cleanup_on_exception(self):
        """Test that directory is cleaned up even when exception occurs."""
        created_path = None

        with pytest.raises(ValueError, match="test error"):
            with _managed_temp_directory("test_prefix_") as temp_dir:
                created_path = temp_dir
                assert temp_dir.exists()
                raise ValueError("test error")

        # Directory should still be cleaned up
        assert not created_path.exists()


class TestValidateAnimatelyAvailability:
    """Test Animately availability validation."""

    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.DEFAULT_ENGINE_CONFIG")
    def test_animately_available(self, mock_config, mock_is_executable):
        """Test successful validation when Animately is available."""
        mock_config.ANIMATELY_PATH = "/usr/bin/animately"
        mock_is_executable.return_value = True

        result = _validate_animately_availability()

        assert result == "/usr/bin/animately"
        mock_is_executable.assert_called_once_with("/usr/bin/animately")

    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.DEFAULT_ENGINE_CONFIG")
    def test_animately_not_executable(self, mock_config, mock_is_executable):
        """Test error when Animately path is not executable."""
        mock_config.ANIMATELY_PATH = "/usr/bin/animately"
        mock_is_executable.return_value = False

        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            _validate_animately_availability()

    @patch("giflab.lossy.DEFAULT_ENGINE_CONFIG")
    def test_animately_path_none(self, mock_config):
        """Test error when Animately path is None."""
        mock_config.ANIMATELY_PATH = None

        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            _validate_animately_availability()


class TestExtractGifMetadata:
    """Test GIF metadata extraction."""

    @patch("giflab.lossy.extract_gif_metadata")
    def test_successful_extraction(self, mock_extract):
        """Test successful metadata extraction."""
        mock_metadata = Mock()
        mock_metadata.orig_frames = 10
        mock_metadata.orig_n_colors = 256
        mock_extract.return_value = mock_metadata

        frames, colors = _extract_gif_metadata(Path("test.gif"))

        assert frames == 10
        assert colors == 256
        mock_extract.assert_called_once_with(Path("test.gif"))

    @patch("giflab.lossy.extract_gif_metadata")
    def test_extraction_failure(self, mock_extract):
        """Test error handling when metadata extraction fails."""
        mock_extract.side_effect = Exception("Extraction failed")

        with pytest.raises(RuntimeError, match="Failed to extract metadata"):
            _extract_gif_metadata(Path("test.gif"))


class TestSetupPngSequenceDirectory:
    """Test PNG sequence directory setup."""

    def test_provided_directory_with_files(self):
        """Test using a provided directory with PNG files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir) / "png_sequence"
            png_dir.mkdir()

            # Create test PNG files
            (png_dir / "frame_0001.png").touch()
            (png_dir / "frame_0002.png").touch()

            result_dir, result_metadata, was_provided = _setup_png_sequence_directory(
                png_dir, Path("input.gif"), 2
            )

            assert result_dir == png_dir
            assert was_provided is True
            assert result_metadata["frame_count"] == 2
            assert result_metadata["engine"] == "provided_by_previous_step"
            assert result_metadata["render_ms"] == 0

    def test_provided_directory_no_files(self):
        """Test error when provided directory has no PNG files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir) / "png_sequence"
            png_dir.mkdir()

            with pytest.raises(RuntimeError, match="contains no frames"):
                _setup_png_sequence_directory(png_dir, Path("input.gif"), 2)

    @patch("giflab.lossy.export_png_sequence")
    def test_create_new_directory(self, mock_export):
        """Test creating new PNG sequence directory."""
        mock_export.return_value = {
            "frame_count": 5,
            "render_ms": 1500,
            "engine": "imagemagick",
        }

        result_dir, result_metadata, was_provided = _setup_png_sequence_directory(
            None, Path("input.gif"), 5
        )

        assert result_dir.name.startswith("animately_png_")
        assert was_provided is False
        assert result_metadata["frame_count"] == 5
        assert result_metadata["render_ms"] == 1500
        mock_export.assert_called_once()

    @patch("giflab.lossy.export_png_sequence")
    def test_export_failure(self, mock_export):
        """Test error when PNG export fails."""
        mock_export.return_value = {"frame_count": 0}

        with pytest.raises(RuntimeError, match="Failed to export PNG sequence"):
            _setup_png_sequence_directory(None, Path("input.gif"), 5)


class TestExtractFrameTiming:
    """Test frame timing extraction from GIF."""

    @patch("PIL.Image.open")
    def test_successful_timing_extraction(self, mock_open):
        """Test successful frame timing extraction."""
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_open.return_value = mock_img

        # Set initial info for logging
        mock_img.info = {"duration": 100}

        # Mock seeking to different frames with different durations
        def mock_seek(frame_idx):
            if frame_idx == 0:
                mock_img.info = {"duration": 100}
            elif frame_idx == 1:
                mock_img.info = {"duration": 50}
            else:
                mock_img.info = {"duration": 200}

        mock_img.seek = mock_seek

        delays = _extract_frame_timing(Path("test.gif"), 3)

        assert len(delays) == 3
        assert delays[0] == 100
        assert delays[1] == 50  # Above minimum
        assert delays[2] == 200

    @patch("PIL.Image.open")
    def test_minimum_delay_enforcement(self, mock_open):
        """Test that minimum delay is enforced."""
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_open.return_value = mock_img

        # Mock frame with very short duration
        mock_img.seek = lambda x: None
        mock_img.info = {"duration": 10}  # Below minimum of 20ms

        delays = _extract_frame_timing(Path("test.gif"), 1)

        assert delays[0] == 20  # Should be clamped to minimum

    @patch("PIL.Image.open")
    def test_fallback_on_exception(self, mock_open):
        """Test fallback to default timing on exception."""
        mock_open.side_effect = Exception("PIL error")

        delays = _extract_frame_timing(Path("test.gif"), 3)

        assert len(delays) == 3
        assert all(delay == 100 for delay in delays)  # Default timing


class TestGenerateFrameList:
    """Test frame list generation for JSON config."""

    def test_frame_list_generation(self):
        """Test generating frame list with timing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir)

            # Create test PNG files
            (png_dir / "frame_0001.png").touch()
            (png_dir / "frame_0002.png").touch()
            (png_dir / "frame_0003.png").touch()

            frame_delays = [100, 150, 200]
            frame_list = _generate_frame_list(png_dir, frame_delays)

            assert len(frame_list) == 3
            assert frame_list[0]["delay"] == 100
            assert frame_list[1]["delay"] == 150
            assert frame_list[2]["delay"] == 200

            # Check that PNG paths are absolute
            for frame in frame_list:
                assert Path(frame["png"]).is_absolute()
                assert frame["png"].endswith(".png")

    def test_mismatched_delays_length(self):
        """Test handling when delays list is shorter than PNG files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir)

            # Create more PNG files than delays
            (png_dir / "frame_0001.png").touch()
            (png_dir / "frame_0002.png").touch()
            (png_dir / "frame_0003.png").touch()

            frame_delays = [100, 150]  # Only 2 delays for 3 frames
            frame_list = _generate_frame_list(png_dir, frame_delays)

            assert len(frame_list) == 3
            assert frame_list[0]["delay"] == 100
            assert frame_list[1]["delay"] == 150
            assert frame_list[2]["delay"] == 100  # Default fallback


class TestGenerateJsonConfig:
    """Test JSON configuration generation."""

    @patch("giflab.lossy.validate_color_keep_count")
    def test_json_config_with_colors(self, mock_validate):
        """Test JSON config generation with color count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir)

            frame_files = [
                {"png": "/tmp/frame_001.png", "delay": 100},
                {"png": "/tmp/frame_002.png", "delay": 150},
            ]

            config_path = _generate_json_config(
                png_dir, lossy_level=75, color_keep_count=64, frame_files=frame_files
            )

            assert config_path.exists()
            assert config_path.name == "animately_config.json"

            # Verify JSON content
            with open(config_path) as f:
                config = json.load(f)

            assert config["lossy"] == 75
            assert config["colors"] == 64
            assert len(config["frames"]) == 2
            assert config["frames"][0]["delay"] == 100
            mock_validate.assert_called_once_with(64)

    def test_json_config_without_colors(self):
        """Test JSON config generation without color count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir)

            frame_files = [{"png": "/tmp/frame_001.png", "delay": 100}]

            config_path = _generate_json_config(
                png_dir, lossy_level=50, color_keep_count=None, frame_files=frame_files
            )

            with open(config_path) as f:
                config = json.load(f)

            assert config["lossy"] == 50
            assert "colors" not in config
            assert len(config["frames"]) == 1

    def test_lossy_level_clamping(self):
        """Test that lossy level is clamped to valid range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir)
            frame_files = [{"png": "/tmp/frame_001.png", "delay": 100}]

            # Test clamping high value
            config_path = _generate_json_config(
                png_dir, lossy_level=150, color_keep_count=None, frame_files=frame_files
            )

            with open(config_path) as f:
                config = json.load(f)

            assert config["lossy"] == 100  # Clamped to max

            # Test clamping negative value
            config_path2 = _generate_json_config(
                png_dir, lossy_level=-10, color_keep_count=None, frame_files=frame_files
            )

            with open(config_path2) as f:
                config2 = json.load(f)

            assert config2["lossy"] == 0  # Clamped to min


class TestExecuteAnimatelyAdvanced:
    """Test Animately advanced execution."""

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.validate_path_security")
    def test_successful_execution(self, mock_validate, mock_run):
        """Test successful Animately execution."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.stderr = None
        mock_run.return_value = mock_result

        mock_validate.return_value = Path("/safe/output.gif")

        # Mock output file creation
        output_path = Path("output.gif")
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat:
            mock_stat.return_value.st_size = 1024

            render_ms, stderr = _execute_animately_advanced(
                "/usr/bin/animately", Path("/tmp/config.json"), output_path
            )

            assert render_ms >= 0
            assert stderr is None
            mock_run.assert_called_once()

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.validate_path_security")
    def test_execution_failure(self, mock_validate, mock_run):
        """Test Animately execution failure."""
        from subprocess import CalledProcessError

        mock_validate.return_value = Path("/safe/output.gif")
        mock_run.side_effect = CalledProcessError(
            1, "animately", stderr="Error message"
        )

        with pytest.raises(RuntimeError, match="failed with exit code 1"):
            _execute_animately_advanced(
                "/usr/bin/animately", Path("/tmp/config.json"), Path("output.gif")
            )

    @patch("giflab.lossy.subprocess.run")
    @patch("giflab.lossy.validate_path_security")
    def test_execution_timeout(self, mock_validate, mock_run):
        """Test Animately execution timeout."""
        from subprocess import TimeoutExpired

        mock_validate.return_value = Path("/safe/output.gif")
        mock_run.side_effect = TimeoutExpired("animately", 30)

        with pytest.raises(RuntimeError, match="timed out after"):
            _execute_animately_advanced(
                "/usr/bin/animately", Path("/tmp/config.json"), Path("output.gif")
            )


class TestProcessAdvancedLossy:
    """Test the core processing function."""

    @patch("giflab.lossy._extract_frame_timing")
    @patch("giflab.lossy._generate_frame_list")
    @patch("giflab.lossy._generate_json_config")
    @patch("giflab.lossy._execute_animately_advanced")
    @patch("giflab.lossy.get_animately_version")
    def test_successful_processing(
        self, mock_version, mock_execute, mock_json, mock_frame_list, mock_timing
    ):
        """Test successful processing pipeline."""
        # Setup mocks
        mock_timing.return_value = [100, 150, 200]
        mock_frame_list.return_value = [
            {"png": "/tmp/frame_001.png", "delay": 100},
            {"png": "/tmp/frame_002.png", "delay": 150},
            {"png": "/tmp/frame_003.png", "delay": 200},
        ]
        mock_json.return_value = Path("/tmp/config.json")
        mock_execute.return_value = (2500, None)
        mock_version.return_value = "1.1.20.0"

        png_export_result = {
            "frame_count": 3,
            "render_ms": 1000,
            "engine": "imagemagick",
        }

        result = _process_advanced_lossy(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            lossy_level=75,
            color_keep_count=64,
            png_sequence_dir=Path("/tmp/png_seq"),
            png_export_result=png_export_result,
            total_frames=3,
            original_colors=256,
            animately_path="/usr/bin/animately",
        )

        # Verify result structure
        assert result["render_ms"] == 2500
        assert result["engine"] == "animately-advanced"
        assert result["engine_version"] == "1.1.20.0"
        assert result["lossy_level"] == 75
        assert result["color_keep_count"] == 64
        assert result["original_frames"] == 3
        assert result["original_colors"] == 256
        assert result["frames_processed"] == 3
        assert result["png_sequence_metadata"] == png_export_result

        # Verify function calls
        mock_timing.assert_called_once_with(Path("input.gif"), 3)
        mock_frame_list.assert_called_once_with(Path("/tmp/png_seq"), [100, 150, 200])
        mock_json.assert_called_once()
        mock_execute.assert_called_once()

    @patch("giflab.lossy._extract_frame_timing")
    @patch("giflab.lossy._generate_frame_list")
    @patch("giflab.lossy._generate_json_config")
    @patch("giflab.lossy._execute_animately_advanced")
    @patch("giflab.lossy.get_animately_version")
    def test_version_fallback(
        self, mock_version, mock_execute, mock_json, mock_frame_list, mock_timing
    ):
        """Test fallback when version detection fails."""
        # Setup mocks
        mock_timing.return_value = [100]
        mock_frame_list.return_value = [{"png": "/tmp/frame_001.png", "delay": 100}]
        mock_json.return_value = Path("/tmp/config.json")
        mock_execute.return_value = (1000, None)
        mock_version.side_effect = RuntimeError("Version detection failed")

        result = _process_advanced_lossy(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            lossy_level=50,
            color_keep_count=None,
            png_sequence_dir=Path("/tmp/png_seq"),
            png_export_result={},
            total_frames=1,
            original_colors=128,
            animately_path="/usr/bin/animately",
        )

        assert result["engine_version"] == "unknown"


class TestEndToEndIntegration:
    """Test the main compress_with_animately_advanced_lossy function end-to-end."""

    @patch("giflab.lossy._process_advanced_lossy")
    @patch("giflab.lossy._setup_png_sequence_directory")
    @patch("giflab.lossy._extract_gif_metadata")
    @patch("giflab.lossy._validate_animately_availability")
    def test_successful_compression_with_provided_png(
        self, mock_validate, mock_metadata, mock_setup, mock_process
    ):
        """Test successful compression using provided PNG sequence."""
        # Setup mocks
        mock_validate.return_value = "/usr/bin/animately"
        mock_metadata.return_value = (5, 256)
        mock_setup.return_value = (
            Path("/provided/png_seq"),
            {"frame_count": 5, "engine": "provided_by_previous_step"},
            True,  # was_provided = True
        )
        mock_process.return_value = {
            "render_ms": 3000,
            "engine": "animately-advanced",
            "lossy_level": 80,
        }

        result = compress_with_animately_advanced_lossy(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            lossy_level=80,
            color_keep_count=32,
            png_sequence_dir=Path("/provided/png_seq"),
        )

        assert result["render_ms"] == 3000
        assert result["engine"] == "animately-advanced"
        mock_process.assert_called_once()

    @patch("giflab.lossy.rmtree")
    @patch("giflab.lossy._process_advanced_lossy")
    @patch("giflab.lossy._setup_png_sequence_directory")
    @patch("giflab.lossy._extract_gif_metadata")
    @patch("giflab.lossy._validate_animately_availability")
    def test_successful_compression_with_cleanup(
        self, mock_validate, mock_metadata, mock_setup, mock_process, mock_rmtree
    ):
        """Test successful compression with automatic cleanup."""
        # Setup mocks
        mock_validate.return_value = "/usr/bin/animately"
        mock_metadata.return_value = (3, 128)

        temp_dir = Path("/tmp/animately_png_12345")
        mock_setup.return_value = (
            temp_dir,
            {"frame_count": 3, "engine": "imagemagick"},
            False,  # was_provided = False
        )
        mock_process.return_value = {"render_ms": 2000, "engine": "animately-advanced"}

        result = compress_with_animately_advanced_lossy(
            input_path=Path("input.gif"), output_path=Path("output.gif"), lossy_level=60
        )

        assert result["render_ms"] == 2000
        # Verify cleanup was called
        mock_rmtree.assert_called_once_with(temp_dir)

    @patch("giflab.lossy.rmtree")
    @patch("giflab.lossy._process_advanced_lossy")
    @patch("giflab.lossy._setup_png_sequence_directory")
    @patch("giflab.lossy._extract_gif_metadata")
    @patch("giflab.lossy._validate_animately_availability")
    def test_cleanup_on_processing_failure(
        self, mock_validate, mock_metadata, mock_setup, mock_process, mock_rmtree
    ):
        """Test that cleanup happens even when processing fails."""
        # Setup mocks
        mock_validate.return_value = "/usr/bin/animately"
        mock_metadata.return_value = (3, 128)

        temp_dir = Path("/tmp/animately_png_67890")
        mock_setup.return_value = (
            temp_dir,
            {"frame_count": 3, "engine": "imagemagick"},
            False,  # was_provided = False
        )
        mock_process.side_effect = RuntimeError("Processing failed")

        with pytest.raises(RuntimeError, match="Processing failed"):
            compress_with_animately_advanced_lossy(
                input_path=Path("input.gif"),
                output_path=Path("output.gif"),
                lossy_level=60,
            )

        # Verify cleanup was still called
        mock_rmtree.assert_called_once_with(temp_dir)


@pytest.mark.slow
class TestIntegrationWithRealFiles:
    """Integration tests with real GIF files (marked as slow)."""

    @patch("giflab.external_engines.imagemagick._magick_binary")
    @patch("giflab.lossy._is_executable")
    @patch("giflab.lossy.DEFAULT_ENGINE_CONFIG")
    def test_end_to_end_png_sequence_optimization(
        self, mock_config, mock_is_executable, mock_magick_binary
    ):
        """Test the complete PNG sequence optimization flow with real files."""
        # This test uses actual sample files to verify the complete pipeline
        import shutil
        import tempfile
        from pathlib import Path

        # Mock ImageMagick binary discovery
        mock_magick_binary.return_value = "/usr/bin/convert"

        # Mock Animately availability
        mock_config.ANIMATELY_PATH = "/usr/bin/animately"
        mock_is_executable.return_value = True

        # Use a sample GIF from the test workspace
        sample_gif = Path("test-workspace/samples/simple_4frame.gif")
        if not sample_gif.exists():
            # Use fixture as fallback
            sample_gif = Path("tests/fixtures/simple_4frame.gif")

        if sample_gif.exists():
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_gif = temp_path / "input.gif"
                output_gif = temp_path / "output.gif"
                png_sequence_dir = (
                    temp_path / "png_frames"
                )  # Create PNG directory for test
                png_sequence_dir.mkdir(parents=True, exist_ok=True)

                # Copy sample to temp location
                shutil.copy(sample_gif, input_gif)

                # Mock the actual Animately execution and ImageMagick calls
                with patch("giflab.lossy.subprocess.run") as mock_run, patch(
                    "giflab.lossy.validate_path_security"
                ) as mock_validate, patch(
                    "giflab.external_engines.imagemagick.run_command"
                ) as mock_run_command, patch(
                    "giflab.lossy._extract_frame_timing"
                ) as mock_frame_timing:
                    # Mock successful execution
                    mock_result = Mock()
                    mock_result.stderr = None
                    mock_run.return_value = mock_result
                    mock_validate.side_effect = lambda x: x

                    # Mock frame timing extraction to return reasonable delays
                    mock_frame_timing.return_value = [
                        100,
                        100,
                        100,
                        100,
                    ]  # 100ms per frame

                    # Create mock PNG files in the provided directory since we're providing it
                    for i in range(4):
                        (png_sequence_dir / f"frame_{i:04d}.png").touch()

                    # Mock ImageMagick PNG export (though it won't be called since we provide the directory)
                    def mock_png_export(*args, **kwargs):
                        return {
                            "render_ms": 100,
                            "engine": "imagemagick",
                            "command": ["convert", "input.gif", "output.png"],
                            "kilobytes": 5.0,
                            "frame_count": 4,
                            "frame_pattern": "frame_%04d.png",
                        }

                    mock_run_command.side_effect = mock_png_export

                    # Mock output file creation
                    def create_output_file(*args, **kwargs):
                        output_gif.touch()
                        return mock_result

                    mock_run.side_effect = create_output_file

                    try:
                        # This should work through the complete pipeline:
                        # 1. Validate Animately availability ✓ (mocked)
                        # 2. Extract GIF metadata ✓ (real)
                        # 3. Export PNG sequence ✓ (real ImageMagick call)
                        # 4. Extract frame timing ✓ (real PIL call)
                        # 5. Generate JSON config ✓ (real)
                        # 6. Execute Animately ✓ (mocked)
                        result = compress_with_animately_advanced_lossy(
                            input_path=input_gif,
                            output_path=output_gif,
                            lossy_level=60,
                            color_keep_count=32,
                            png_sequence_dir=png_sequence_dir,
                        )

                        # Verify the pipeline completed successfully
                        assert result["engine"] == "animately-advanced"
                        assert result["lossy_level"] == 60
                        assert result["color_keep_count"] == 32
                        assert "render_ms" in result
                        assert "png_sequence_metadata" in result
                        assert "json_config_path" in result

                        # Verify PNG sequence was processed
                        png_metadata = result["png_sequence_metadata"]
                        assert png_metadata["frame_count"] > 0
                        assert "render_ms" in png_metadata

                        # Verify JSON config path was provided (file may be cleaned up)
                        json_path = result["json_config_path"]
                        assert json_path  # Should have a path even if cleaned up
                        assert "animately_config.json" in str(json_path)

                        with open(json_path) as f:
                            config = json.load(f)

                        assert config["lossy"] == 60
                        assert config["colors"] == 32
                        assert "frames" in config
                        assert len(config["frames"]) > 0

                        # Verify frame timing was preserved
                        for frame in config["frames"]:
                            assert "png" in frame
                            assert "delay" in frame
                            assert frame["delay"] >= 20  # Minimum delay enforced
                            assert Path(frame["png"]).suffix == ".png"

                    except Exception as e:
                        # If ImageMagick is not available, skip this test
                        if "ImageMagick" in str(e) or "convert" in str(e):
                            pytest.skip(
                                "ImageMagick not available for integration test"
                            )
                        else:
                            raise
        else:
            pytest.skip("No sample GIF files available for integration test")

    def test_png_sequence_directory_provided_integration(self):
        """Test integration when PNG sequence directory is provided by previous step."""
        import tempfile
        from pathlib import Path

        # Create a mock PNG sequence directory
        with tempfile.TemporaryDirectory() as temp_dir:
            png_dir = Path(temp_dir) / "png_sequence"
            png_dir.mkdir()

            # Create mock PNG files
            for i in range(3):
                png_file = png_dir / f"frame_{i+1:04d}.png"
                png_file.touch()

            input_gif = Path(temp_dir) / "input.gif"
            output_gif = Path(temp_dir) / "output.gif"

            # Create a minimal mock GIF for metadata extraction
            input_gif.touch()

            with patch("giflab.lossy._is_executable", return_value=True), patch(
                "giflab.lossy.DEFAULT_ENGINE_CONFIG"
            ) as mock_config, patch(
                "giflab.lossy.extract_gif_metadata"
            ) as mock_metadata, patch(
                "giflab.lossy.subprocess.run"
            ) as mock_run, patch(
                "giflab.lossy.validate_path_security"
            ) as mock_validate:
                # Setup mocks
                mock_config.ANIMATELY_PATH = "/usr/bin/animately"
                mock_metadata_obj = Mock()
                mock_metadata_obj.orig_frames = 3
                mock_metadata_obj.orig_n_colors = 128
                mock_metadata.return_value = mock_metadata_obj

                mock_result = Mock()
                mock_result.stderr = None
                mock_run.return_value = mock_result
                mock_validate.side_effect = lambda x: x

                # Mock output file creation
                def create_output_file(*args, **kwargs):
                    output_gif.touch()
                    return mock_result

                mock_run.side_effect = create_output_file

                # Test with provided PNG sequence directory
                result = compress_with_animately_advanced_lossy(
                    input_path=input_gif,
                    output_path=output_gif,
                    lossy_level=75,
                    png_sequence_dir=png_dir,
                )

                # Verify PNG sequence was used from provided directory
                assert result["engine"] == "animately-advanced"
                png_metadata = result["png_sequence_metadata"]
                assert png_metadata["engine"] == "provided_by_previous_step"
                assert png_metadata["frame_count"] == 3
                assert png_metadata["render_ms"] == 0  # No export time

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            input_gif = Path(temp_dir) / "input.gif"
            output_gif = Path(temp_dir) / "output.gif"
            input_gif.touch()

            # Test 1: Animately not available
            with patch("giflab.lossy._is_executable", return_value=False):
                with pytest.raises(RuntimeError, match="Animately launcher not found"):
                    compress_with_animately_advanced_lossy(
                        input_path=input_gif, output_path=output_gif, lossy_level=60
                    )

            # Test 2: Metadata extraction failure
            with patch("giflab.lossy._is_executable", return_value=True), patch(
                "giflab.lossy.DEFAULT_ENGINE_CONFIG"
            ) as mock_config, patch(
                "giflab.lossy.extract_gif_metadata"
            ) as mock_metadata:
                mock_config.ANIMATELY_PATH = "/usr/bin/animately"
                mock_metadata.side_effect = Exception("Metadata extraction failed")

                with pytest.raises(RuntimeError, match="Failed to extract metadata"):
                    compress_with_animately_advanced_lossy(
                        input_path=input_gif, output_path=output_gif, lossy_level=60
                    )

            # Test 3: PNG sequence export failure
            with patch("giflab.lossy._is_executable", return_value=True), patch(
                "giflab.lossy.DEFAULT_ENGINE_CONFIG"
            ) as mock_config, patch(
                "giflab.lossy.extract_gif_metadata"
            ) as mock_metadata, patch(
                "giflab.lossy.export_png_sequence"
            ) as mock_export:
                mock_config.ANIMATELY_PATH = "/usr/bin/animately"
                mock_metadata_obj = Mock()
                mock_metadata_obj.orig_frames = 5
                mock_metadata_obj.orig_n_colors = 256
                mock_metadata.return_value = mock_metadata_obj

                # Mock failed PNG export
                mock_export.return_value = {"frame_count": 0}

                with pytest.raises(RuntimeError, match="Failed to export PNG sequence"):
                    compress_with_animately_advanced_lossy(
                        input_path=input_gif, output_path=output_gif, lossy_level=60
                    )


if __name__ == "__main__":
    pytest.main([__file__])
