"""Unit tests for SSIMULACRA2 perceptual quality metrics integration."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from giflab.config import MetricsConfig
from giflab.ssimulacra2_metrics import (
    DEFAULT_SSIMULACRA2_PATH,
    SSIMULACRA2_EXCELLENT_SCORE,
    SSIMULACRA2_POOR_SCORE,
    Ssimulacra2Validator,
    calculate_ssimulacra2_quality_metrics,
    should_use_ssimulacra2,
)


class TestSsimulacra2Validator:
    """Test the SSIMULACRA2 validator class."""

    def test_init_default_path(self):
        """Test validator initialization with default path."""
        validator = Ssimulacra2Validator()
        assert validator.binary_path == Path(DEFAULT_SSIMULACRA2_PATH)

    def test_init_custom_path(self):
        """Test validator initialization with custom path."""
        custom_path = "/custom/path/to/ssimulacra2"
        validator = Ssimulacra2Validator(custom_path)
        assert validator.binary_path == Path(custom_path)

    @patch("subprocess.run")
    @patch.object(Path, "exists")
    @patch.object(Path, "is_file")
    def test_is_available_true(self, mock_is_file, mock_exists, mock_subprocess):
        """Test is_available returns True when binary exists."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)

        validator = Ssimulacra2Validator()
        assert validator.is_available() is True

    @patch("subprocess.run")
    @patch.object(Path, "exists")
    def test_is_available_false_missing_binary(self, mock_exists, mock_subprocess):
        """Test is_available returns False when binary doesn't exist."""
        mock_exists.return_value = False

        validator = Ssimulacra2Validator()
        assert validator.is_available() is False

    @patch("subprocess.run", side_effect=FileNotFoundError())
    def test_is_available_false_subprocess_error(self, mock_subprocess):
        """Test is_available returns False when subprocess fails."""
        validator = Ssimulacra2Validator()
        assert validator.is_available() is False

    def test_should_use_ssimulacra2_none_quality(self):
        """Test conditional triggering with None quality (first pass)."""
        validator = Ssimulacra2Validator()
        assert validator.should_use_ssimulacra2(None) is True

    def test_should_use_ssimulacra2_borderline_quality(self):
        """Test conditional triggering with borderline quality."""
        validator = Ssimulacra2Validator()
        assert validator.should_use_ssimulacra2(0.5) is True  # < 0.7 threshold
        assert validator.should_use_ssimulacra2(0.69) is True  # Just below threshold

    def test_should_use_ssimulacra2_high_quality(self):
        """Test conditional triggering with high quality."""
        validator = Ssimulacra2Validator()
        assert validator.should_use_ssimulacra2(0.8) is False  # > 0.7 threshold
        assert validator.should_use_ssimulacra2(0.95) is False  # High quality

    def test_normalize_score_excellent(self):
        """Test score normalization for excellent scores."""
        validator = Ssimulacra2Validator()
        assert validator.normalize_score(90.0) == 1.0
        assert validator.normalize_score(95.0) == 1.0  # Capped at 1.0

    def test_normalize_score_poor(self):
        """Test score normalization for poor scores."""
        validator = Ssimulacra2Validator()
        assert validator.normalize_score(10.0) == 0.0
        assert validator.normalize_score(5.0) == 0.0  # Capped at 0.0
        assert validator.normalize_score(-10.0) == 0.0  # Negative scores

    def test_normalize_score_medium(self):
        """Test score normalization for medium scores."""
        validator = Ssimulacra2Validator()
        # Linear interpolation between 10 (poor) and 90 (excellent)
        expected = (50.0 - 10.0) / (90.0 - 10.0)  # Should be 0.5
        assert abs(validator.normalize_score(50.0) - expected) < 0.001

    @patch("PIL.Image.fromarray")
    def test_export_frame_to_png(self, mock_fromarray):
        """Test PNG export functionality."""
        validator = Ssimulacra2Validator()

        # Create mock frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Create mock image
        mock_image = Mock()
        mock_fromarray.return_value = mock_image

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.png"
            validator._export_frame_to_png(frame, output_path)

            mock_fromarray.assert_called_once()
            mock_image.save.assert_called_once_with(output_path, "PNG")

    @patch("PIL.Image.fromarray")
    def test_export_frame_to_png_float_input(self, mock_fromarray):
        """Test PNG export with float input (needs conversion)."""
        validator = Ssimulacra2Validator()

        # Create mock frame with float values [0, 1]
        frame = np.random.random((100, 100, 3)).astype(np.float32)

        mock_image = Mock()
        mock_fromarray.return_value = mock_image

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.png"
            validator._export_frame_to_png(frame, output_path)

            # Verify the frame was converted to uint8
            call_args = mock_fromarray.call_args[0][0]
            assert call_args.dtype == np.uint8
            assert np.all(call_args >= 0) and np.all(call_args <= 255)

    @patch("subprocess.run")
    def test_run_ssimulacra2_on_pair_success(self, mock_subprocess):
        """Test successful SSIMULACRA2 execution."""
        validator = Ssimulacra2Validator()

        # Mock successful subprocess result
        mock_subprocess.return_value = Mock(returncode=0, stdout="75.42\n", stderr="")

        result = validator._run_ssimulacra2_on_pair(
            Path("/fake/orig.png"), Path("/fake/comp.png")
        )

        assert result == 75.42
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_run_ssimulacra2_on_pair_failure(self, mock_subprocess):
        """Test SSIMULACRA2 execution failure."""
        validator = Ssimulacra2Validator()

        # Mock failed subprocess result
        mock_subprocess.return_value = Mock(
            returncode=1, stdout="", stderr="Error: invalid input"
        )

        with pytest.raises(subprocess.CalledProcessError):
            validator._run_ssimulacra2_on_pair(
                Path("/fake/orig.png"), Path("/fake/comp.png")
            )

    @patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="test", timeout=30)
    )
    def test_run_ssimulacra2_timeout(self, mock_subprocess):
        """Test SSIMULACRA2 execution timeout."""
        validator = Ssimulacra2Validator()

        with pytest.raises(subprocess.TimeoutExpired):
            validator._run_ssimulacra2_on_pair(
                Path("/fake/orig.png"), Path("/fake/comp.png")
            )

    def test_sample_frame_indices_all_frames(self):
        """Test frame sampling when total <= max."""
        validator = Ssimulacra2Validator()

        indices = validator._sample_frame_indices(total_frames=5, max_frames=10)
        assert indices == [0, 1, 2, 3, 4]

        indices = validator._sample_frame_indices(total_frames=3, max_frames=3)
        assert indices == [0, 1, 2]

    def test_sample_frame_indices_uniform_sampling(self):
        """Test uniform frame sampling when total > max."""
        validator = Ssimulacra2Validator()

        indices = validator._sample_frame_indices(total_frames=100, max_frames=10)

        # Should return exactly 10 indices
        assert len(indices) == 10

        # Should be sorted
        assert indices == sorted(indices)

        # Should include first and last frames (approximately)
        assert indices[0] == 0
        assert indices[-1] == 99

    @patch.object(Ssimulacra2Validator, "is_available")
    def test_calculate_ssimulacra2_metrics_unavailable(self, mock_is_available):
        """Test metrics calculation when SSIMULACRA2 is unavailable."""
        mock_is_available.return_value = False

        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Create mock frames
        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        result = validator.calculate_ssimulacra2_metrics(
            orig_frames, comp_frames, config
        )

        expected_keys = {
            "ssimulacra2_mean",
            "ssimulacra2_p95",
            "ssimulacra2_min",
            "ssimulacra2_frame_count",
            "ssimulacra2_triggered",
        }
        assert set(result.keys()) == expected_keys
        assert result["ssimulacra2_triggered"] == 0.0

    def test_calculate_ssimulacra2_metrics_frame_mismatch(self):
        """Test metrics calculation with mismatched frame counts."""
        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        # Create mismatched frames
        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)
        ]

        with pytest.raises(ValueError, match="Frame count mismatch"):
            validator.calculate_ssimulacra2_metrics(orig_frames, comp_frames, config)

    @patch.object(Ssimulacra2Validator, "is_available")
    @patch.object(Ssimulacra2Validator, "_export_frame_to_png")
    @patch.object(Ssimulacra2Validator, "_run_ssimulacra2_on_pair")
    @patch.object(Ssimulacra2Validator, "normalize_score")
    def test_calculate_ssimulacra2_metrics_success(
        self, mock_normalize, mock_run_ssim, mock_export, mock_is_available
    ):
        """Test successful metrics calculation."""
        mock_is_available.return_value = True
        mock_run_ssim.side_effect = [75.0, 80.0, 70.0]  # Raw SSIMULACRA2 scores
        mock_normalize.side_effect = [0.75, 0.80, 0.70]  # Normalized scores

        validator = Ssimulacra2Validator()
        config = MetricsConfig()
        config.SSIMULACRA2_MAX_FRAMES = 5

        # Create mock frames
        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        result = validator.calculate_ssimulacra2_metrics(
            orig_frames, comp_frames, config
        )

        # Check that all frames were processed (3 <= 5 max frames)
        assert mock_run_ssim.call_count == 3
        assert mock_export.call_count == 6  # 3 orig + 3 comp
        assert mock_normalize.call_count == 3

        # Check results
        assert result["ssimulacra2_mean"] == 0.75  # Mean of [0.75, 0.80, 0.70]
        assert result["ssimulacra2_p95"] == 0.795  # 95th percentile
        assert result["ssimulacra2_min"] == 0.70  # Minimum
        assert result["ssimulacra2_frame_count"] == 3.0
        assert result["ssimulacra2_triggered"] == 1.0

    @patch.object(Ssimulacra2Validator, "is_available")
    @patch.object(Ssimulacra2Validator, "_export_frame_to_png")
    @patch.object(Ssimulacra2Validator, "_run_ssimulacra2_on_pair")
    def test_calculate_ssimulacra2_metrics_with_failures(
        self, mock_run_ssim, mock_export, mock_is_available
    ):
        """Test metrics calculation with some frame processing failures."""
        mock_is_available.return_value = True

        # First frame succeeds, second fails, third succeeds
        mock_run_ssim.side_effect = [75.0, Exception("Processing failed"), 70.0]

        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        orig_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]
        comp_frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)
        ]

        with patch.object(
            validator, "normalize_score", side_effect=[0.75, 0.70]
        ):
            result = validator.calculate_ssimulacra2_metrics(
                orig_frames, comp_frames, config
            )

        # Should have processed 3 frames but only 2 successful scores + 1 fallback (0.5)
        expected_scores = [0.75, 0.5, 0.70]  # Success, fallback, success
        assert result["ssimulacra2_mean"] == np.mean(expected_scores)
        assert result["ssimulacra2_frame_count"] == 3.0
        assert result["ssimulacra2_triggered"] == 1.0


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @patch.object(Ssimulacra2Validator, "calculate_ssimulacra2_metrics")
    def test_calculate_ssimulacra2_quality_metrics(self, mock_calculate):
        """Test the main module function."""
        mock_result = {"ssimulacra2_mean": 0.75}
        mock_calculate.return_value = mock_result

        config = MetricsConfig()
        orig_frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
        comp_frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]

        result = calculate_ssimulacra2_quality_metrics(orig_frames, comp_frames, config)

        assert result == mock_result
        mock_calculate.assert_called_once_with(orig_frames, comp_frames, config)

    @patch.object(Ssimulacra2Validator, "should_use_ssimulacra2")
    def test_should_use_ssimulacra2_function(self, mock_should_use):
        """Test the convenience function for conditional triggering."""
        mock_should_use.return_value = True

        result = should_use_ssimulacra2(0.5)

        assert result is True
        mock_should_use.assert_called_once_with(0.5)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch.object(Ssimulacra2Validator, "is_available")
    def test_high_quality_skip_scenario(self, mock_is_available):
        """Test that high quality GIFs skip SSIMULACRA2 calculation."""
        mock_is_available.return_value = True

        validator = Ssimulacra2Validator()

        # High composite quality should not trigger SSIMULACRA2
        assert validator.should_use_ssimulacra2(0.85) is False

    @patch.object(Ssimulacra2Validator, "is_available")
    def test_borderline_quality_trigger_scenario(self, mock_is_available):
        """Test that borderline quality GIFs trigger SSIMULACRA2."""
        mock_is_available.return_value = True

        validator = Ssimulacra2Validator()

        # Borderline quality should trigger SSIMULACRA2
        assert validator.should_use_ssimulacra2(0.45) is True
        assert validator.should_use_ssimulacra2(0.65) is True

    def test_score_normalization_edge_cases(self):
        """Test score normalization with edge cases."""
        validator = Ssimulacra2Validator()

        # Test extreme values
        assert validator.normalize_score(float("inf")) == 1.0
        assert validator.normalize_score(float("-inf")) == 0.0
        assert validator.normalize_score(0.0) == 0.0

        # Test boundary values
        assert validator.normalize_score(SSIMULACRA2_POOR_SCORE) == 0.0
        assert validator.normalize_score(SSIMULACRA2_EXCELLENT_SCORE) == 1.0

    @patch.object(Ssimulacra2Validator, "is_available")
    @patch("tempfile.TemporaryDirectory")
    def test_temporary_file_cleanup(self, mock_tempdir, mock_is_available):
        """Test that temporary files are properly cleaned up."""
        mock_is_available.return_value = True

        # Mock the context manager
        MagicMock()
        mock_tempdir.return_value.__enter__ = Mock(return_value="/fake/temp/dir")
        mock_tempdir.return_value.__exit__ = Mock(return_value=None)

        validator = Ssimulacra2Validator()
        config = MetricsConfig()

        orig_frames = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)]
        comp_frames = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)]

        with patch.object(validator, "_export_frame_to_png"), patch.object(
            validator, "_run_ssimulacra2_on_pair", return_value=50.0
        ), patch.object(validator, "normalize_score", return_value=0.5):
            result = validator.calculate_ssimulacra2_metrics(
                orig_frames, comp_frames, config
            )

        # Verify temporary directory context manager was used
        mock_tempdir.assert_called_once_with(prefix="ssimulacra2_")
        assert result["ssimulacra2_triggered"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
