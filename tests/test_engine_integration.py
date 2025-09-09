"""Integration tests for compression engines (gifsicle and animately)."""

import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
from giflab.config import DEFAULT_ENGINE_CONFIG
from giflab.lossy import (
    LossyEngine,
    _is_executable,
    apply_lossy_compression,
    compress_with_animately,
    compress_with_gifsicle,
)
from PIL import Image, ImageDraw


def create_test_gif(path: Path, frames: int = 5, size: tuple = (50, 50)) -> None:
    """Create a simple test GIF for testing purposes.

    Args:
        path: Path where to save the GIF
        frames: Number of frames in the GIF
        size: Size of each frame (width, height)
    """
    images = []
    for i in range(frames):
        # Create a simple colored square that changes color
        img = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
        draw = ImageDraw.Draw(img)
        # Add a simple shape that moves
        draw.rectangle([i * 5, i * 5, i * 5 + 10, i * 5 + 10], fill=(255, 255, 255))
        images.append(img)

    # Save as GIF
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 200ms per frame
        loop=0,
    )


class TestEngineAvailability:
    """Test that both engines are available and properly configured."""

    def test_gifsicle_available(self):
        """Test that gifsicle is available and executable."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        if not _is_executable(gifsicle_path):
            pytest.skip(f"gifsicle not found at {gifsicle_path}")

        try:
            result = subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, text=True, timeout=9
            )
            assert result.returncode == 0, f"gifsicle not working: {result.stderr}"
            assert (
                "gifsicle" in result.stdout.lower()
            ), f"Unexpected version output: {result.stdout}"
        except FileNotFoundError:
            pytest.skip(f"gifsicle not found at {gifsicle_path}")
        except subprocess.TimeoutExpired:
            pytest.fail("gifsicle --version timed out")

    def test_animately_available(self):
        """Test that animately is available and executable."""
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        if not _is_executable(animately_path):
            pytest.skip(f"Animately not found at {animately_path}")

        try:
            result = subprocess.run(
                [animately_path, "--help"], capture_output=True, text=True, timeout=9
            )
            # Animately might return non-zero for --help, so check output instead
            assert (
                "--input" in result.stdout or "--input" in result.stderr
            ), f"Unexpected help output: {result.stdout}"
        except FileNotFoundError:
            pytest.skip(f"Animately not found at {animately_path}")
        except subprocess.TimeoutExpired:
            pytest.fail("animately --help timed out")

    def test_engine_config_paths_exist(self):
        """Test that configured engine paths exist."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        gifsicle_exists = _is_executable(gifsicle_path)
        animately_exists = _is_executable(animately_path)

        print(f"Gifsicle path: {gifsicle_path} (exists: {gifsicle_exists})")
        print(f"Animately path: {animately_path} (exists: {animately_exists})")

        # At least one should exist for the tests to be meaningful
        assert gifsicle_exists or animately_exists, "Neither engine is available"


class TestGifsicleIntegration:
    """Integration tests for gifsicle engine."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path)
            yield gif_path

    @pytest.fixture
    def output_path(self):
        """Create a temporary output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "output.gif"

    def test_gifsicle_lossless_compression(self, test_gif, output_path):
        """Test gifsicle lossless compression."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        # Check if gifsicle is available
        try:
            subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, check=True, timeout=5
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            pytest.skip(f"gifsicle not available at {gifsicle_path}")

        # Test compression
        result = compress_with_gifsicle(test_gif, output_path, lossy_level=0)

        # Verify results
        assert output_path.exists(), "Output file was not created"
        assert result["engine"] == "gifsicle"
        assert result["lossy_level"] == 0
        assert result["render_ms"] > 0
        assert "command" in result

        # Verify output is a valid GIF
        assert output_path.stat().st_size > 0, "Output file is empty"

    def test_gifsicle_lossy_compression(self, test_gif, output_path):
        """Test gifsicle lossy compression."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        # Check if gifsicle is available
        try:
            subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, check=True, timeout=5
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            pytest.skip(f"gifsicle not available at {gifsicle_path}")

        # Test compression
        result = compress_with_gifsicle(test_gif, output_path, lossy_level=40)

        # Verify results
        assert output_path.exists(), "Output file was not created"
        assert result["engine"] == "gifsicle"
        assert result["lossy_level"] == 40
        assert result["render_ms"] > 0
        assert "--lossy=40" in result["command"]


class TestAnimatelyIntegration:
    """Integration tests for animately engine."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path)
            yield gif_path

    @pytest.fixture
    def output_path(self):
        """Create a temporary output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "output.gif"

    def test_animately_lossless_compression(self, test_gif, output_path):
        """Test animately lossless compression."""
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        # Check if animately is available
        if not _is_executable(animately_path):
            pytest.skip(f"Animately not found at {animately_path}")

        # Test compression
        result = compress_with_animately(test_gif, output_path, lossy_level=0)

        # Verify results
        assert output_path.exists(), "Output file was not created"
        assert result["engine"] == "animately"
        assert result["lossy_level"] == 0
        assert result["render_ms"] > 0
        assert "command" in result

        # Verify output is a valid GIF
        assert output_path.stat().st_size > 0, "Output file is empty"

    def test_animately_lossy_compression(self, test_gif, output_path):
        """Test animately lossy compression."""
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        # Check if animately is available
        if not _is_executable(animately_path):
            pytest.skip(f"Animately not found at {animately_path}")

        # Test compression
        result = compress_with_animately(test_gif, output_path, lossy_level=40)

        # Verify results
        assert output_path.exists(), "Output file was not created"
        assert result["engine"] == "animately"
        assert result["lossy_level"] == 40
        assert result["render_ms"] > 0
        assert "--lossy" in result["command"] and "40" in result["command"]


class TestEngineComparison:
    """Compare both engines on the same input."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path, frames=10)  # Larger GIF for better comparison
            yield gif_path

    def test_engine_comparison(self, test_gif):
        """Compare both engines on the same input."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        # Check availability
        gifsicle_available = _is_executable(gifsicle_path)
        animately_available = _is_executable(animately_path)

        # Double-check gifsicle with version command
        if gifsicle_available:
            try:
                subprocess.run(
                    [gifsicle_path, "--version"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
            except Exception:
                gifsicle_available = False

        if not gifsicle_available and not animately_available:
            pytest.skip("Neither engine is available")

        results = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test gifsicle if available
            if gifsicle_available:
                gifsicle_output = Path(temp_dir) / "gifsicle_output.gif"
                try:
                    results["gifsicle"] = compress_with_gifsicle(
                        test_gif, gifsicle_output, lossy_level=0
                    )
                    results["gifsicle"]["output_size"] = gifsicle_output.stat().st_size
                except Exception as e:
                    results["gifsicle"] = {"error": str(e)}

            # Test animately if available
            if animately_available:
                animately_output = Path(temp_dir) / "animately_output.gif"
                try:
                    results["animately"] = compress_with_animately(
                        test_gif, animately_output, lossy_level=0
                    )
                    results["animately"][
                        "output_size"
                    ] = animately_output.stat().st_size
                except Exception as e:
                    results["animately"] = {"error": str(e)}

        # Print comparison results
        print("\nEngine Comparison Results:")
        for engine, result in results.items():
            if "error" in result:
                print(f"{engine}: ERROR - {result['error']}")
            else:
                print(
                    f"{engine}: {result['render_ms']}ms, {result['output_size']} bytes"
                )

        # At least one should succeed
        successful_engines = [k for k, v in results.items() if "error" not in v]
        assert len(successful_engines) > 0, f"All engines failed: {results}"


class TestHighLevelAPI:
    """Test the high-level API with both engines."""

    @pytest.fixture
    def test_gif(self):
        """Create a temporary test GIF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "test.gif"
            create_test_gif(gif_path)
            yield gif_path

    def test_apply_lossy_compression_gifsicle(self, test_gif):
        """Test high-level API with gifsicle."""
        gifsicle_path = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH

        try:
            subprocess.run(
                [gifsicle_path, "--version"], capture_output=True, check=True, timeout=5
            )
        except Exception:
            pytest.skip(f"gifsicle not available at {gifsicle_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.gif"

            result = apply_lossy_compression(
                test_gif,
                output_path,
                lossy_level=0,
                frame_keep_ratio=1.0,
                engine=LossyEngine.GIFSICLE,
            )

            assert output_path.exists()
            assert result["engine"] == "gifsicle"

    def test_apply_lossy_compression_animately(self, test_gif):
        """Test high-level API with animately."""
        animately_path = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH

        if not _is_executable(animately_path):
            pytest.skip(f"Animately not found at {animately_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.gif"

            result = apply_lossy_compression(
                test_gif,
                output_path,
                lossy_level=0,
                frame_keep_ratio=1.0,
                engine=LossyEngine.ANIMATELY,
            )

            assert output_path.exists()
            assert result["engine"] == "animately"
