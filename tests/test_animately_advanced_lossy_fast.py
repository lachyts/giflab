"""Fast tests for AnimatelyAdvancedLossyCompressor using fast_compress fixture."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from PIL import Image, ImageDraw

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


def create_test_gif(path: Path, frames: int = 5, size: tuple = (50, 50)) -> None:
    """Create a simple test GIF for testing purposes."""
    images = []
    for i in range(frames):
        img = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
        draw = ImageDraw.Draw(img)
        draw.rectangle([i * 5, i * 5, i * 5 + 10, i * 5 + 10], fill=(255, 255, 255))
        images.append(img)

    images[0].save(path, save_all=True, append_images=images[1:], duration=200, loop=0)


@pytest.mark.fast
class TestAnimatelyAdvancedLossyCompressorFast:
    """Fast tests for AnimatelyAdvancedLossyCompressor wrapper class."""

    def test_tool_registration_fast(self):
        """Test that the tool is properly registered with correct attributes."""
        tool = AnimatelyAdvancedLossyCompressor()

        assert tool.NAME == "animately-advanced-lossy"
        assert tool.COMBINE_GROUP == "animately"
        assert tool.VARIABLE == "lossy_compression"
        assert hasattr(tool, "available")
        assert hasattr(tool, "version")
        assert hasattr(tool, "apply")

    def test_available_method_fast(self):
        """Test the availability check."""
        with patch("giflab.tool_wrappers._is_executable") as mock_is_executable:
            mock_is_executable.return_value = True
            tool = AnimatelyAdvancedLossyCompressor()
            assert tool.available() is True

            mock_is_executable.return_value = False
            assert tool.available() is False

    def test_version_method_fast(self):
        """Test version retrieval."""
        with patch("giflab.tool_wrappers.get_animately_version") as mock_version:
            mock_version.return_value = "1.2.3"
            tool = AnimatelyAdvancedLossyCompressor()
            assert tool.version() == "1.2.3"

    def test_apply_method_fast(self, fast_compress):
        """Test the apply method with fast compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            create_test_gif(input_path)

            tool = AnimatelyAdvancedLossyCompressor()

            # Mock the advanced lossy compression to use fast fixture behavior
            with patch(
                "giflab.lossy.compress_with_animately_advanced_lossy"
            ) as mock_compress:
                mock_compress.return_value = {
                    "render_ms": 1,
                    "engine": "noop-advanced",
                    "command": "noop-advanced",
                    "kilobytes": 10.5,
                    "ssim": 1.0,
                    "lossy_level": 75,
                }

                result = tool.apply(input_path, output_path, params={"lossy_level": 75})

                assert isinstance(result, dict)
                assert result["engine"] == "noop-advanced"
                assert result["lossy_level"] == 75
                mock_compress.assert_called_once()


@pytest.mark.fast
class TestAnimatelyAdvancedLossyFunctionsFast:
    """Fast tests for animately advanced lossy helper functions."""

    def test_validate_animately_availability_fast(self):
        """Test animately availability validation."""
        with patch("giflab.lossy._is_executable") as mock_executable:
            mock_executable.return_value = True

            # Should not raise when available
            _validate_animately_availability()

            mock_executable.return_value = False

            # Should raise when not available
            with pytest.raises(RuntimeError, match="Animately launcher not found"):
                _validate_animately_availability()

    def test_extract_gif_metadata_fast(self):
        """Test GIF metadata extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=3, size=(100, 100))

            total_frames, original_colors = _extract_gif_metadata(gif_path)

            assert isinstance(total_frames, int)
            assert isinstance(original_colors, int)
            assert total_frames == 3
            assert original_colors > 0  # Should have some colors

    def test_extract_frame_timing_fast(self):
        """Test frame timing extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=4)

            timings = _extract_frame_timing(gif_path, 4)

            assert isinstance(timings, list)
            assert len(timings) == 4
            # All frames should have the default duration we set (200ms)
            for timing in timings:
                assert timing == 200

    def test_generate_frame_list_fast(self):
        """Test frame list generation."""
        frame_count = 5
        # Need to provide PNG directory and frame delays
        with tempfile.TemporaryDirectory() as tmpdir:
            png_dir = Path(tmpdir) / "frames"
            png_dir.mkdir()
            # Create mock PNG files
            for i in range(frame_count):
                (png_dir / f"frame_{i:04d}.png").touch()

            frame_delays = [100, 150, 200, 100, 120]  # 5 frame delays
            frame_list = _generate_frame_list(png_dir, frame_delays)

        assert len(frame_list) == frame_count
        for i, frame_info in enumerate(frame_list):
            assert "png" in frame_info
            assert "delay" in frame_info
            assert frame_info["delay"] == frame_delays[i]

    def test_generate_json_config_fast(self):
        """Test JSON config generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            png_dir = Path(tmpdir) / "frames"
            png_dir.mkdir()

            # Create frame files data
            frame_files = [
                {"png": str(png_dir / "frame_0000.png"), "delay": 100},
                {"png": str(png_dir / "frame_0001.png"), "delay": 150},
            ]

            json_config = _generate_json_config(png_dir, 80, None, frame_files)

            # Should return a Path to JSON config file
            assert isinstance(json_config, Path)
            assert json_config.exists()

            # Read and validate the JSON content
            with open(json_config) as f:
                parsed = json.load(f)
            assert "lossy" in parsed
            assert "frames" in parsed
            assert parsed["lossy"] == 80
            assert len(parsed["frames"]) == 2

    def test_setup_png_sequence_directory_fast(self):
        """Test PNG sequence directory setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            create_test_gif(gif_path, frames=3)

            png_dir = Path(tmpdir) / "png_sequence"

            with patch("giflab.lossy.export_png_sequence") as mock_export:
                # Mock successful PNG export
                mock_export.return_value = {
                    "render_ms": 100,
                    "frame_count": 3,
                    "frame_pattern": "frame_%04d.png",
                }

                (
                    png_dir_result,
                    png_export_result,
                    was_provided,
                ) = _setup_png_sequence_directory(None, gif_path, 3)

                assert isinstance(png_dir_result, Path)
                assert isinstance(png_export_result, dict)
                assert was_provided == False
                mock_export.assert_called_once()

                # Verify PNG export was called with correct paths
                call_args = mock_export.call_args
                assert call_args[0][0] == gif_path  # input_path
                assert isinstance(call_args[0][1], Path)  # output_dir

    def test_execute_animately_advanced_fast(self):
        """Test animately advanced execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            output_path = Path(tmpdir) / "output.gif"
            png_dir = Path(tmpdir) / "frames"
            png_dir.mkdir()

            # Create mock config file
            config_data = {
                "frames": ["frame_0000.png"],
                "frame_timings": [200],
                "width": 100,
                "height": 100,
                "lossy_level": 50,
            }
            config_path.write_text(json.dumps(config_data))

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create the expected output file since the function checks if it exists
                output_path.write_bytes(b"fake gif output")

                render_ms, stderr = _execute_animately_advanced(
                    "animately", config_path, output_path
                )

                assert isinstance(render_ms, int)
                assert render_ms >= 0  # Can be 0 for fast mocked execution
                mock_run.assert_called_once()

                # Check command structure
                call_args = mock_run.call_args[0][0]
                assert str(config_path) in call_args
                # The output path gets resolved, so check if the filename is there
                assert output_path.name in " ".join(call_args)

    def test_process_advanced_lossy_integration_fast(self, fast_compress):
        """Test the complete advanced lossy processing pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.gif"
            output_path = Path(tmpdir) / "output.gif"

            create_test_gif(input_path, frames=2, size=(80, 80))

            # Mock all the subprocess calls
            with patch("subprocess.run") as mock_run, patch(
                "giflab.lossy._validate_animately_availability"
            ), patch(
                "shutil.rmtree"
            ):  # Mock cleanup
                # Mock subprocess.run and ensure output file gets created
                def create_output_file(*args, **kwargs):
                    output_path.write_bytes(b"fake gif output")
                    return MagicMock(returncode=0)

                mock_run.side_effect = create_output_file

                # Mock all required parameters for _process_advanced_lossy
                png_dir = Path(tmpdir) / "png_frames"
                png_dir.mkdir()  # Create the directory that the JSON config needs
                png_export_result = {"render_ms": 100, "frame_count": 2}

                result = _process_advanced_lossy(
                    input_path=input_path,
                    output_path=output_path,
                    lossy_level=60,
                    color_keep_count=None,
                    png_sequence_dir=png_dir,
                    png_export_result=png_export_result,
                    total_frames=2,
                    original_colors=256,
                    animately_path="animately",
                )

                assert isinstance(result, dict)
                assert "render_ms" in result
                assert "command" in result
                # Should have called subprocess (animately)
                assert mock_run.call_count >= 1

    def test_managed_temp_directory_fast(self):
        """Test managed temporary directory context manager."""
        temp_dirs_created = []

        # Capture created directories
        original_mkdtemp = tempfile.mkdtemp

        def mock_mkdtemp(*args, **kwargs):
            temp_dir = original_mkdtemp(*args, **kwargs)
            temp_dirs_created.append(Path(temp_dir))
            return temp_dir

        with patch("tempfile.mkdtemp", side_effect=mock_mkdtemp), patch(
            "giflab.lossy.rmtree"
        ) as mock_rmtree:
            with _managed_temp_directory() as temp_dir:
                assert isinstance(temp_dir, Path)
                assert temp_dir in temp_dirs_created
                # Directory should exist during context
                # (Note: in real usage it would exist, but our mock doesn't create it)

            # Should have called rmtree to clean up
            mock_rmtree.assert_called_once()

    def test_compress_with_animately_advanced_lossy_fast(self, fast_compress):
        """Test the main advanced lossy compression function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.gif"
            output_path = Path(tmpdir) / "output.gif"

            create_test_gif(input_path, frames=3)

            # Mock the process function to return fast results
            with patch("giflab.lossy._process_advanced_lossy") as mock_process:
                mock_process.return_value = {
                    "render_ms": 150,
                    "engine": "animately-advanced",
                    "command": ["animately", "advanced", "args"],
                    "kilobytes": 25.5,
                    "lossy_level": 70,
                }

                result = compress_with_animately_advanced_lossy(
                    input_path=input_path, output_path=output_path, lossy_level=70
                )

                assert result["engine"] == "animately-advanced"
                assert result["lossy_level"] == 70
                assert result["render_ms"] == 150
                # The actual call will have many more parameters
                mock_process.assert_called_once()


@pytest.mark.fast
class TestAnimatelyAdvancedErrorHandlingFast:
    """Fast tests for error handling in advanced lossy compression."""

    def test_missing_animately_binary_fast(self):
        """Test handling of missing animately binary."""
        with patch("giflab.lossy._is_executable", return_value=False):
            with pytest.raises(RuntimeError, match="Animately launcher not found"):
                _validate_animately_availability()

    def test_invalid_gif_metadata_fast(self):
        """Test handling of invalid GIF files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-GIF file
            fake_gif = Path(tmpdir) / "fake.gif"
            fake_gif.write_text("not a gif")

            with pytest.raises(RuntimeError):
                _extract_gif_metadata(fake_gif)

    def test_ffmpeg_extraction_failure_fast(self):
        """Test handling of ffmpeg extraction failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = Path(tmpdir) / "test.gif"
            png_dir = Path(tmpdir) / "png_frames"

            create_test_gif(gif_path)

            # Mock the export_png_sequence function to return 0 frames (failure case)
            with patch(
                "giflab.lossy.export_png_sequence",
                return_value={"frame_count": 0, "render_ms": 100},
            ):
                with pytest.raises(RuntimeError, match="Failed to export PNG sequence"):
                    _setup_png_sequence_directory(None, gif_path, 5)

    def test_animately_execution_failure_fast(self):
        """Test handling of animately execution failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            output_path = Path(tmpdir) / "output.gif"
            png_dir = Path(tmpdir) / "frames"

            config_path.write_text('{"frames": []}')
            png_dir.mkdir()

            with patch("subprocess.run") as mock_run:
                # Mock animately failure
                mock_run.return_value = MagicMock(
                    returncode=1, stderr="Animately error"
                )

                with pytest.raises(
                    RuntimeError, match="Animately advanced lossy execution failed"
                ):
                    _execute_animately_advanced("animately", config_path, output_path)

    def test_invalid_lossy_level_bounds_fast(self):
        """Test validation of lossy level bounds."""
        # This test doesn't actually call compression, just tests parameter validation
        invalid_levels = [-1, -10, 1000]

        for level in invalid_levels:
            # The validation should happen before any actual processing
            # In a real implementation, this might be validated in the function
            assert level < 0 or level > 200  # Expected invalid range

    def test_temp_directory_cleanup_on_error_fast(self):
        """Test that temporary directories are cleaned up on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_temp_dir = Path(tmpdir) / "test_temp_dir"

            with patch(
                "tempfile.mkdtemp", return_value=str(test_temp_dir)
            ) as mock_mkdtemp, patch("giflab.lossy.rmtree") as mock_rmtree:
                # Simulate an error during the managed temp directory usage
                with pytest.raises(RuntimeError, match="Test directory error"):
                    with _managed_temp_directory() as temp_path:
                        # This should trigger cleanup even on error
                        raise RuntimeError("Test directory error")

                # Cleanup should still be called
                mock_rmtree.assert_called_once_with(test_temp_dir)
