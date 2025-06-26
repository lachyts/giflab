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
    validate_lossy_level
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
                LossyEngine.GIFSICLE
            )

    def test_missing_input_file(self):
        """Test that missing input file raises IOError."""
        with pytest.raises(IOError, match="Input file not found"):
            apply_lossy_compression(
                Path("nonexistent.gif"),
                Path("output.gif"),
                0,
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
            LossyEngine.GIFSICLE
        )
        
        mock_compress.assert_called_once_with(input_path, output_path, 40)
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
            LossyEngine.ANIMATELY
        )
        
        mock_compress.assert_called_once_with(input_path, output_path, 120)
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
                "invalid_engine"  # This will cause the error
            )


class TestCompressWithGifsicle:
    """Tests for compress_with_gifsicle function."""

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossless(self, mock_time, mock_exists, mock_run):
        """Test successful gifsicle compression with lossless setting."""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms execution
        
        # Mock successful subprocess
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock output file exists
        mock_exists.return_value = True
        
        input_path = Path("input.gif")
        output_path = Path("output.gif")
        
        result = compress_with_gifsicle(input_path, output_path, 0)
        
        # Verify command construction
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
        assert result["command"] == " ".join(expected_cmd)

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossy(self, mock_time, mock_exists, mock_run):
        """Test successful gifsicle compression with lossy setting."""
        mock_time.side_effect = [1000.0, 1001.0]  # 1000ms execution
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        mock_exists.return_value = True
        
        result = compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 40)
        
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

    @patch('subprocess.run')
    def test_subprocess_error(self, mock_run):
        """Test handling of subprocess errors."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "gifsicle", stderr="Error message"
        )
        
        with pytest.raises(RuntimeError, match="Gifsicle failed with exit code 1"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0)

    @patch('subprocess.run')
    def test_timeout_error(self, mock_run):
        """Test handling of subprocess timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("gifsicle", 300)
        
        with pytest.raises(RuntimeError, match="Gifsicle timed out"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0)

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_missing_output_file(self, mock_time, mock_exists, mock_run):
        """Test error when output file is not created."""
        mock_time.side_effect = [1000.0, 1000.1]
        mock_run.return_value = MagicMock()
        mock_exists.return_value = False  # Output file doesn't exist
        
        with pytest.raises(RuntimeError, match="failed to create output file"):
            compress_with_gifsicle(Path("input.gif"), Path("output.gif"), 0)


class TestCompressWithAnimately:
    """Tests for compress_with_animately function."""

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossless(self, mock_time, mock_exists, mock_run):
        """Test successful animately compression with lossless setting."""
        mock_time.side_effect = [1000.0, 1000.3]  # 300ms execution
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock animately launcher and output file both exist
        mock_exists.return_value = True
        
        result = compress_with_animately(Path("input.gif"), Path("output.gif"), 0)
        
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

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('time.time')
    def test_successful_compression_lossy(self, mock_time, mock_exists, mock_run):
        """Test successful animately compression with lossy setting."""
        mock_time.side_effect = [1000.0, 1002.0]  # 2000ms execution
        
        mock_result = MagicMock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock both launcher and output file exist
        mock_exists.return_value = True
        
        result = compress_with_animately(Path("input.gif"), Path("output.gif"), 120)
        
        # Verify lossy flag is included
        expected_cmd = ["/Users/lachlants/bin/launcher", "gif", "optimize", "--lossy", "120", "input.gif", "output.gif"]
        mock_run.assert_called_once_with(
            expected_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        
        assert result["render_ms"] == 2000
        assert result["lossy_level"] == 120

    @patch('pathlib.Path.exists')
    def test_missing_launcher(self, mock_exists):
        """Test error when animately launcher is missing."""
        mock_exists.return_value = False
        
        with pytest.raises(RuntimeError, match="Animately launcher not found"):
            compress_with_animately(Path("input.gif"), Path("output.gif"), 0)

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_subprocess_error(self, mock_exists, mock_run):
        """Test handling of subprocess errors."""
        mock_exists.return_value = True  # Launcher exists
        mock_run.side_effect = subprocess.CalledProcessError(
            2, "animately", stderr="Animately error"
        )
        
        with pytest.raises(RuntimeError, match="Animately failed with exit code 2"):
            compress_with_animately(Path("input.gif"), Path("output.gif"), 0) 