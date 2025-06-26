"""Tests for giflab.lossy module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess
import time

from giflab.lossy import (
    LossyEngine,
    apply_lossy_compression,
    compress_with_gifsicle,
    compress_with_animately,
    validate_lossy_level,
    apply_compression_with_all_params
)


class TestLossyEngine:
    """Tests for LossyEngine enum."""

    def test_engine_values(self):
        """Test that enum values are correct."""
        assert LossyEngine.GIFSICLE.value == "gifsicle"
        assert LossyEngine.ANIMATELY.value == "animately"


class TestValidateLossyLevel:
    """Tests for validate_lossy_level function."""

    def test_valid_levels(self):
        """Test that valid lossy levels pass validation."""
        for level in [0, 40, 120]:
            # Should not raise any exception
            validate_lossy_level(level, LossyEngine.GIFSICLE)
            validate_lossy_level(level, LossyEngine.ANIMATELY)

    def test_negative_level(self):
        """Test that negative lossy levels raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_lossy_level(-1, LossyEngine.GIFSICLE)

    def test_invalid_level(self):
        """Test that invalid lossy levels raise ValueError."""
        with pytest.raises(ValueError, match="not in supported levels"):
            validate_lossy_level(99, LossyEngine.GIFSICLE)


class TestApplyLossyCompression:
    """Tests for apply_lossy_compression function."""

    def test_negative_lossy_level(self):
        """Test that negative lossy level raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            apply_lossy_compression(
                Path("input.gif"),
                Path("output.gif"),
                -1,
                1.0,  # frame_keep_ratio
                LossyEngine.GIFSICLE
            )

    def test_missing_input_file(self):
        """Test that missing input file raises IOError."""
        with pytest.raises(IOError, match="Input file not found"):
            apply_lossy_compression(
                Path("nonexistent.gif"),
                Path("output.gif"),
                0,
                1.0,  # frame_keep_ratio
                LossyEngine.GIFSICLE
            )

    @patch('giflab.lossy.compress_with_gifsicle')
    @patch('pathlib.Path.exists')
    def test_gifsicle_engine_dispatch(self, mock_exists, mock_compress):
        """Test that gifsicle engine is called correctly."""
        mock_exists.return_value = True
        mock_compress.return_value = {"render_ms": 100}
        
        input_path = Path("input.gif")
        output_path = Path("output.gif")
        
        result = apply_lossy_compression(
            input_path,
            output_path,
            40,
            1.0,  # frame_keep_ratio
            LossyEngine.GIFSICLE
        )
        
        mock_compress.assert_called_once_with(input_path, output_path, 40, 1.0)
        assert result == {"render_ms": 100}

    @patch('giflab.lossy.compress_with_animately')
    @patch('pathlib.Path.exists')
    def test_animately_engine_dispatch(self, mock_exists, mock_compress):
        """Test that animately engine is called correctly."""
        mock_exists.return_value = True
        mock_compress.return_value = {"render_ms": 200}
        
        input_path = Path("input.gif")
        output_path = Path("output.gif")
        
        result = apply_lossy_compression(
            input_path,
            output_path,
            120,
            0.8,  # frame_keep_ratio
            LossyEngine.ANIMATELY
        )
        
        mock_compress.assert_called_once_with(input_path, output_path, 120, 0.8)
        assert result == {"render_ms": 200}

    @patch('pathlib.Path.exists')
    def test_unsupported_engine(self, mock_exists):
        """Test that unsupported engine raises ValueError."""
        mock_exists.return_value = True
        
        # Create a mock engine that's not in the enum
        with pytest.raises(ValueError, match="Unsupported engine"):
            apply_lossy_compression(
                Path("input.gif"),
                Path("output.gif"),
                0,
                1.0,  # frame_keep_ratio
                "invalid_engine"  # This will cause the error
            )


class TestCompressWithGifsicle:
    """Tests for compress_with_gifsicle function."""

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossless(self, mock_time, mock_exists, mock_run, mock_metadata):
        """Test successful gifsicle compression with lossless setting."""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms execution
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 10
        mock_metadata.return_value = mock_meta
        
        # Mock successful subprocess
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock output file exists
        mock_exists.return_value = True
        
        input_path = Path("input.gif")
        output_path = Path("output.gif")
        
        result = compress_with_gifsicle(input_path, output_path, 0, 1.0)
        
        # Verify command construction (no frame reduction for ratio 1.0)
        expected_cmd = ["gifsicle", "--optimize", str(input_path), "--output", str(output_path)]
        mock_run.assert_called_once_with(
            expected_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        
        # Verify result
        assert result["render_ms"] == 500
        assert result["engine"] == "gifsicle"
        assert result["lossy_level"] == 0
        assert result["frame_keep_ratio"] == 1.0
        assert result["original_frames"] == 10
        assert result["command"] == " ".join(expected_cmd)

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossy(self, mock_time, mock_exists, mock_run, mock_metadata):
        """Test successful gifsicle compression with lossy setting."""
        mock_time.side_effect = [1000.0, 1001.0]  # 1000ms execution
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 20
        mock_metadata.return_value = mock_meta
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True
        
        result = compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 40, 1.0)
        
        # Verify lossy flag is included
        expected_cmd = ["gifsicle", "--optimize", "--lossy=40", "input.gif", "--output", "output.gif"]
        mock_run.assert_called_once_with(
            expected_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        
        assert result["render_ms"] == 1000
        assert result["lossy_level"] == 40
        assert result["frame_keep_ratio"] == 1.0
        assert result["original_frames"] == 20

    @patch('giflab.lossy.build_gifsicle_frame_args')
    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_compression_with_frame_reduction(self, mock_time, mock_exists, mock_run, mock_metadata, mock_frame_args):
        """Test gifsicle compression with frame reduction."""
        mock_time.side_effect = [1000.0, 1000.8]  # 800ms execution
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 10
        mock_metadata.return_value = mock_meta
        
        # Mock frame reduction arguments
        mock_frame_args.return_value = ["--delete", "#1", "--delete", "#3"]
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True
        
        result = compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 0.8)
        
        # Verify frame reduction arguments are included
        expected_cmd = ["gifsicle", "--optimize", "--delete", "#1", "--delete", "#3", "input.gif", "--output", "output.gif"]
        mock_run.assert_called_once_with(
            expected_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        
        # Verify frame args function was called
        mock_frame_args.assert_called_once_with(0.8, 10)
        
        assert result["frame_keep_ratio"] == 0.8
        assert result["original_frames"] == 10

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    def test_subprocess_error(self, mock_run, mock_metadata):
        """Test handling of subprocess errors."""
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_metadata.return_value = mock_meta
        
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "gifsicle", stderr="Error message"
        )
        
        with pytest.raises(RuntimeError, match="Gifsicle failed with exit code 1"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 1.0)

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    def test_timeout_error(self, mock_run, mock_metadata):
        """Test handling of subprocess timeout."""
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_metadata.return_value = mock_meta
        
        mock_run.side_effect = subprocess.TimeoutExpired("gifsicle", 300)
        
        with pytest.raises(RuntimeError, match="Gifsicle timed out"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 1.0)

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_missing_output_file(self, mock_time, mock_exists, mock_run, mock_metadata):
        """Test error when output file is not created."""
        mock_time.side_effect = [1000.0, 1000.1]
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_metadata.return_value = mock_meta
        
        mock_run.return_value = MagicMock()
        mock_exists.return_value = False  # Output file doesn't exist
        
        with pytest.raises(RuntimeError, match="failed to create output file"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0, 1.0)


class TestCompressWithAnimately:
    """Tests for compress_with_animately function."""

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossless(self, mock_time, mock_exists, mock_run, mock_metadata):
        """Test successful animately compression with lossless setting."""
        mock_time.side_effect = [1000.0, 1000.3]  # 300ms execution
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 15
        mock_metadata.return_value = mock_meta
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock animately launcher and output file both exist
        mock_exists.return_value = True
        
        result = compress_with_animately(Path("input.gif"), Path("output.gif"), 0, 1.0)
        
        # Verify command construction
        expected_cmd = ["/Users/lachlants/bin/launcher", "gif", "optimize", "input.gif", "output.gif"]
        mock_run.assert_called_once_with(
            expected_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        
        # Timing should be approximately 300ms (allow for floating point precision)
        assert 298 <= result["render_ms"] <= 301
        assert result["engine"] == "animately"
        assert result["lossy_level"] == 0
        assert result["frame_keep_ratio"] == 1.0
        assert result["original_frames"] == 15

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossy(self, mock_time, mock_exists, mock_run, mock_metadata):
        """Test successful animately compression with lossy setting."""
        mock_time.side_effect = [1000.0, 1002.0]  # 2000ms execution
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 8
        mock_metadata.return_value = mock_meta
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock both launcher and output file exist
        mock_exists.return_value = True
        
        result = compress_with_animately(Path("input.gif"), Path("output.gif"), 120, 0.7)
        
        # Should include frame reduction args
        # Note: The exact command will depend on mock_frame_args, but we can't easily test the exact command here
        # without more complex mocking, so we'll just verify the basic structure
        
        assert result["render_ms"] == 2000
        assert result["lossy_level"] == 120
        assert result["frame_keep_ratio"] == 0.7
        assert result["original_frames"] == 8

    @patch('pathlib.Path.exists')
    def test_missing_launcher(self, mock_exists):
        """Test error when animately launcher is missing."""
        mock_exists.return_value = False
        
        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            compress_with_animately(Path("input.gif"), Path("output.gif"), 0)

    @patch('giflab.lossy.extract_gif_metadata')
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_subprocess_error(self, mock_exists, mock_run, mock_metadata):
        """Test handling of subprocess errors."""
        mock_exists.return_value = True  # Launcher exists
        
        # Mock metadata extraction
        mock_meta = MagicMock()
        mock_meta.orig_frames = 5
        mock_metadata.return_value = mock_meta
        
        mock_run.side_effect = subprocess.CalledProcessError(
            2, "animately", stderr="Animately error"
        )
        
        with pytest.raises(RuntimeError, match="Animately failed with exit code 2"):
            compress_with_animately(Path("input.gif"), Path("output.gif"), 0, 1.0)


class TestApplyCompressionWithAllParams:
    """Tests for apply_compression_with_all_params function."""

    @patch('giflab.lossy.apply_lossy_compression')
    def test_delegates_to_lossy_compression(self, mock_apply_lossy):
        """Test that function delegates to apply_lossy_compression."""
        mock_apply_lossy.return_value = {"render_ms": 300, "engine": "gifsicle"}
        
        result = apply_compression_with_all_params(
            Path("input.gif"),
            Path("output.gif"),
            40,
            0.8,
            64,  # color_keep_count (not yet implemented)
            LossyEngine.GIFSICLE
        )
        
        # Should call apply_lossy_compression with lossy and frame params
        mock_apply_lossy.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            40,
            0.8,
            LossyEngine.GIFSICLE
        )
        
        assert result == {"render_ms": 300, "engine": "gifsicle"}

    @patch('giflab.lossy.apply_lossy_compression')
    def test_defaults_work(self, mock_apply_lossy):
        """Test that default parameters work correctly."""
        mock_apply_lossy.return_value = {"render_ms": 150}
        
        result = apply_compression_with_all_params(
            Path("input.gif"),
            Path("output.gif"),
            0,
            1.0
        )
        
        mock_apply_lossy.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            0,
            1.0,
            LossyEngine.GIFSICLE
        )


class TestFrameKeepRatioValidation:
    """Tests for frame keep ratio validation in lossy functions."""

    def test_invalid_frame_keep_ratio(self):
        """Test that invalid frame keep ratios are rejected."""
        with pytest.raises(ValueError, match="not in supported ratios"):
            apply_lossy_compression(
                Path("input.gif"),
                Path("output.gif"),
                0,
                0.33,  # Not in supported ratios
                LossyEngine.GIFSICLE
            )

    def test_valid_frame_keep_ratios(self):
        """Test that all configured frame keep ratios are accepted."""
        valid_ratios = [1.0, 0.9, 0.8, 0.7, 0.5]
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False  # Will fail with file not found, but validation should pass
            
            for ratio in valid_ratios:
                with pytest.raises(IOError, match="Input file not found"):
                    # Should pass validation but fail on file not found
                    apply_lossy_compression(
                        Path("nonexistent.gif"),
                        Path("output.gif"),
                        0,
                        ratio,
                        LossyEngine.GIFSICLE
                    ) 