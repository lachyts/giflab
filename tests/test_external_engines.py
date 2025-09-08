"""Tests for external engine helper functions."""

import tempfile
from pathlib import Path

import pytest

from giflab.external_engines import (
    ffmpeg_color_reduce,
    ffmpeg_frame_reduce,
    ffmpeg_lossy_compress,
    gifski_lossy_compress,
    imagemagick_color_reduce,
    imagemagick_frame_reduce,
    imagemagick_lossy_compress,
)
from giflab.external_engines.common import run_command

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_gif():
    """Path to a simple test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "simple_4frame.gif"


@pytest.fixture
def single_frame_gif():
    """Path to a single-frame test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "single_frame.gif"


@pytest.fixture
def many_colors_gif():
    """Path to a many-color test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "many_colors.gif"


# ---------------------------------------------------------------------------
# Common utility tests
# ---------------------------------------------------------------------------


def test_run_command_success():
    """Test run_command with a successful command."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        output_path = Path(tmp.name)

        # Simple command that should succeed
        result = run_command(["echo", "hello"], engine="test", output_path=output_path)

        assert result["engine"] == "test"
        assert result["render_ms"] >= 0
        assert "echo hello" in result["command"]
        assert "kilobytes" in result


def test_run_command_failure():
    """Test run_command with a failing command."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        output_path = Path(tmp.name)

        with pytest.raises(RuntimeError, match="test command failed"):
            run_command(
                ["false"],  # Command that always fails
                engine="test",
                output_path=output_path,
            )


# ---------------------------------------------------------------------------
# ImageMagick tests (mocked - require ImageMagick installation)
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestImageMagickHelpers:
    """Tests for ImageMagick helper functions (require ImageMagick)."""

    def test_color_reduce_basic(self, test_gif):
        """Test basic color reduction."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = imagemagick_color_reduce(test_gif, output_path, colors=16)

            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert "colors" in result["command"]
            assert output_path.exists()

    def test_frame_reduce_basic(self, test_gif):
        """Test basic frame reduction."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = imagemagick_frame_reduce(test_gif, output_path, keep_ratio=0.5)

            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert output_path.exists()

    def test_lossy_compress_basic(self, test_gif):
        """Test basic lossy compression."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = imagemagick_lossy_compress(test_gif, output_path, quality=75)

            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert "quality" in result["command"]
            assert output_path.exists()


# ---------------------------------------------------------------------------
# FFmpeg tests (mocked - require FFmpeg installation)
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestFFmpegHelpers:
    """Tests for FFmpeg helper functions (require FFmpeg)."""

    def test_color_reduce_basic(self, test_gif):
        """Test FFmpeg color reduction via palette."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = ffmpeg_color_reduce(test_gif, output_path)

            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert "palettegen" in result["command"]
            assert "paletteuse" in result["command"]
            assert output_path.exists()

    def test_frame_reduce_basic(self, test_gif):
        """Test FFmpeg frame reduction."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = ffmpeg_frame_reduce(test_gif, output_path, fps=5.0)

            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert "fps=5.0" in result["command"]
            assert output_path.exists()

    def test_lossy_compress_basic(self, test_gif):
        """Test FFmpeg lossy compression."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = ffmpeg_lossy_compress(test_gif, output_path, q_scale=25)

            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert "q:v" in result["command"]
            assert output_path.exists()


# ---------------------------------------------------------------------------
# gifski tests (mocked - require gifski + ImageMagick)
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestGifskiHelpers:
    """Tests for gifski helper functions (require gifski + ImageMagick)."""

    def test_lossy_compress_basic(self, test_gif):
        """Test gifski lossy compression."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = gifski_lossy_compress(test_gif, output_path, quality=80)

            assert result["engine"] == "gifski"
            assert result["render_ms"] > 0
            assert "quality" in result["command"]
            assert output_path.exists()


# ---------------------------------------------------------------------------
# Parameter validation tests (no external tools required)
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """Test parameter validation without requiring external tools."""

    def test_imagemagick_color_reduce_invalid_colors(self, test_gif):
        """Test color reduction with invalid color count."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            with pytest.raises(ValueError, match="colors must be between 1 and 256"):
                imagemagick_color_reduce(test_gif, output_path, colors=300)

    def test_imagemagick_frame_reduce_invalid_ratio(self, test_gif):
        """Test frame reduction with invalid ratio."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            with pytest.raises(ValueError, match="keep_ratio must be in"):
                imagemagick_frame_reduce(test_gif, output_path, keep_ratio=1.5)

    def test_imagemagick_lossy_compress_invalid_quality(self, test_gif):
        """Test lossy compression with invalid quality."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            with pytest.raises(ValueError, match="quality must be in 0–100 range"):
                imagemagick_lossy_compress(test_gif, output_path, quality=150)

    def test_gifski_lossy_compress_invalid_quality(self, test_gif):
        """Test gifski lossy compression with invalid quality."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            with pytest.raises(ValueError, match="quality must be in 0–100"):
                gifski_lossy_compress(test_gif, output_path, quality=150)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.external_tools
class TestEdgeCases:
    """Test edge cases that might cause issues."""

    def test_single_frame_gif_frame_reduction(self, single_frame_gif):
        """Test frame reduction on single-frame GIF."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            # This might fail or succeed depending on engine - test graceful handling
            try:
                result = imagemagick_frame_reduce(
                    single_frame_gif, output_path, keep_ratio=0.5
                )
                # If it succeeds, validate the result
                assert result["engine"] == "imagemagick"
                assert output_path.exists()
            except RuntimeError:
                # If it fails, that's also acceptable for edge cases
                pass

    def test_no_op_frame_reduction(self, test_gif):
        """Test frame reduction with ratio=1.0 (should be no-op)."""
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)

            result = imagemagick_frame_reduce(test_gif, output_path, keep_ratio=1.0)

            assert result["engine"] == "imagemagick"
            assert result["command"] == "cp"  # Should be a copy operation
            assert output_path.exists()
