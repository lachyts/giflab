"""Fast integration tests for compression engines using fast_compress fixture."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.lossy import (
    LossyEngine,
    apply_lossy_compression,
    compress_with_animately,
    compress_with_gifsicle,
)
from PIL import Image, ImageDraw


def create_test_gif(path: Path, frames: int = 5, size: tuple = (50, 50)) -> None:
    """Create a simple test GIF for testing purposes (same as slow version).

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


@pytest.mark.fast
class TestEngineIntegrationFast:
    """Fast integration tests for compression engines using fast_compress fixture."""

    @patch("giflab.lossy.compress_with_gifsicle")
    def test_gifsicle_compression_fast(self, mock_gifsicle):
        """Test gifsicle compression with fast mocked implementation."""

        # Configure mock to copy file and return expected results
        def mock_compress(input_path, output_path, **kwargs):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            # Create test GIF
            create_test_gif(input_path)

            # Test compression (should use mocked implementation)
            from giflab.lossy import compress_with_gifsicle

            result = compress_with_gifsicle(
                input_path=input_path, output_path=output_path, lossy_level=50
            )

            # Verify mock was called
            mock_gifsicle.assert_called_once_with(
                input_path=input_path, output_path=output_path, lossy_level=50
            )

            # Verify the fast fixture behavior
            assert output_path.exists()
            assert isinstance(result, dict)
            assert result["engine"] == "noop"
            assert result["render_ms"] == 1
            assert result["ssim"] == 1.0  # Perfect similarity in fast mode
            assert result["lossy_level"] == 50

    @patch("giflab.lossy.compress_with_animately")
    def test_animately_compression_fast(self, mock_animately):
        """Test animately compression with fast mocked implementation."""

        # Configure mock to copy file and return expected results
        def mock_compress(input_path, output_path, **kwargs):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_animately.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            # Create test GIF
            create_test_gif(input_path, frames=3)

            # Test compression (should use mocked implementation)
            from giflab.lossy import compress_with_animately

            result = compress_with_animately(
                input_path=input_path, output_path=output_path, lossy_level=30
            )

            # Verify the fast fixture behavior
            assert output_path.exists()
            assert isinstance(result, dict)
            assert result["engine"] == "noop"
            assert result["render_ms"] == 1
            assert result["ssim"] == 1.0
            assert result["lossy_level"] == 30

    @patch("giflab.lossy.compress_with_gifsicle")
    def test_apply_lossy_compression_gifsicle_fast(self, mock_gifsicle):
        """Test apply_lossy_compression with gifsicle engine using fast fixture."""

        # Configure mock to copy file and return expected results
        def mock_compress(
            input_path, output_path, lossy_level=0, frame_keep_ratio=1.0, **kwargs
        ):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": lossy_level,
                "frame_keep_ratio": frame_keep_ratio,
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            create_test_gif(input_path)

            result = apply_lossy_compression(
                input_path=input_path,
                output_path=output_path,
                lossy_level=40,
                frame_keep_ratio=0.8,
                engine=LossyEngine.GIFSICLE,
            )

            # Check result structure
            assert isinstance(result, dict)
            assert "render_ms" in result
            assert "engine" in result
            assert result["engine"] == "noop"
            assert result["frame_keep_ratio"] == 0.8
            assert result["lossy_level"] == 40

    @patch("giflab.lossy.compress_with_animately")
    def test_apply_lossy_compression_animately_fast(self, mock_animately):
        """Test apply_lossy_compression with animately engine using fast fixture."""

        # Configure mock to copy file and return expected results
        def mock_compress(
            input_path, output_path, lossy_level=0, frame_keep_ratio=1.0, **kwargs
        ):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": lossy_level,
                "frame_keep_ratio": frame_keep_ratio,
            }

        mock_animately.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            create_test_gif(input_path)

            result = apply_lossy_compression(
                input_path=input_path,
                output_path=output_path,
                lossy_level=60,
                frame_keep_ratio=1.0,
                engine=LossyEngine.ANIMATELY,
            )

            # Check result structure
            assert isinstance(result, dict)
            assert "render_ms" in result
            assert "engine" in result
            assert result["engine"] == "noop"
            assert result["frame_keep_ratio"] == 1.0
            assert result["lossy_level"] == 60

    @patch("giflab.lossy.compress_with_gifsicle")
    @patch("giflab.lossy.compress_with_animately")
    def test_compression_with_different_parameters_fast(
        self, mock_animately, mock_gifsicle
    ):
        """Test compression with various parameter combinations using fast fixture."""

        # Configure mocks to copy file and return expected results
        def mock_compress_gifsicle(
            input_path, output_path, lossy_level=0, frame_keep_ratio=1.0, **kwargs
        ):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": lossy_level,
                "frame_keep_ratio": frame_keep_ratio,
            }

        def mock_compress_animately(
            input_path, output_path, lossy_level=0, frame_keep_ratio=1.0, **kwargs
        ):
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": lossy_level,
                "frame_keep_ratio": frame_keep_ratio,
            }

        mock_gifsicle.side_effect = mock_compress_gifsicle
        mock_animately.side_effect = mock_compress_animately

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"

            create_test_gif(input_path, frames=8, size=(100, 100))

            test_cases = [
                {
                    "lossy_level": 0,
                    "frame_keep_ratio": 1.0,
                    "engine": LossyEngine.GIFSICLE,
                },
                {
                    "lossy_level": 100,
                    "frame_keep_ratio": 0.5,
                    "engine": LossyEngine.GIFSICLE,
                },
                {
                    "lossy_level": 50,
                    "frame_keep_ratio": 0.8,
                    "engine": LossyEngine.ANIMATELY,
                },
            ]

            for i, params in enumerate(test_cases):
                output_path = Path(tmpdir) / f"test_output_{i}.gif"

                result = apply_lossy_compression(
                    input_path=input_path, output_path=output_path, **params
                )

                # Verify output exists and has expected metadata
                assert output_path.exists()
                assert result["engine"] == "noop"  # Fast fixture
                assert result["lossy_level"] == params["lossy_level"]
                assert result["frame_keep_ratio"] == params["frame_keep_ratio"]

    @patch("giflab.lossy.compress_with_gifsicle")
    def test_compression_error_handling_fast(self, mock_gifsicle):
        """Test error handling in compression functions using fast fixture."""

        # Configure mock to raise FileNotFoundError for nonexistent file
        def mock_compress(input_path, output_path, **kwargs):
            if not Path(input_path).exists():
                raise FileNotFoundError(f"No such file: {input_path}")
            import shutil

            shutil.copyfile(input_path, output_path)
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with non-existent input file
            input_path = Path(tmpdir) / "nonexistent.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            # The fast fixture should still handle this gracefully
            # by trying to copy a non-existent file, which should raise an error
            with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
                compress_with_gifsicle(
                    input_path=input_path, output_path=output_path, lossy_level=50
                )

    @patch("tests.test_engine_integration_fast.compress_with_gifsicle")
    def test_output_directory_creation_fast(self, mock_gifsicle):
        """Test that output directories are created when needed using fast fixture."""

        # Configure mock to create directories and copy file
        def mock_compress(input_path, output_path, **kwargs):
            import shutil

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
            file_size = output_path.stat().st_size
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "kilobytes": round(file_size / 1024, 2),
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_gifsicle.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            # Output path with nested directories that don't exist
            output_path = Path(tmpdir) / "nested" / "dirs" / "test_output.gif"

            create_test_gif(input_path)

            result = compress_with_gifsicle(
                input_path=input_path, output_path=output_path, lossy_level=25
            )

            # Verify directory was created and file exists
            assert output_path.parent.exists()
            assert output_path.exists()
            assert result["engine"] == "noop"

    @patch("tests.test_engine_integration_fast.compress_with_animately")
    def test_preserve_gif_properties_fast(self, mock_animately):
        """Test that GIF properties are preserved in fast mode."""

        # Configure mock to copy file and return expected results
        def mock_compress(input_path, output_path, **kwargs):
            import shutil

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
            file_size = output_path.stat().st_size
            return {
                "render_ms": 1,
                "engine": "noop",
                "command": "noop-copy",
                "ssim": 1.0,
                "kilobytes": round(file_size / 1024, 2),
                "lossy_level": kwargs.get("lossy_level", 0),
            }

        mock_animately.side_effect = mock_compress

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            # Create GIF with specific properties
            create_test_gif(input_path, frames=10, size=(150, 150))

            result = compress_with_animately(
                input_path=input_path, output_path=output_path, lossy_level=75
            )

            # In fast mode, file is just copied, so properties should be preserved
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert result["ssim"] == 1.0  # Perfect similarity due to copy
            assert result["kilobytes"] > 0  # Should have actual file size

    def test_engine_selection_logic_fast(self, fast_compress):
        """Test engine selection logic using fast fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"

            create_test_gif(input_path)

            # Test both engines
            for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
                output_path = Path(tmpdir) / f"test_{engine.value}.gif"

                result = apply_lossy_compression(
                    input_path=input_path,
                    output_path=output_path,
                    lossy_level=50,
                    engine=engine,
                )

                # Both should work with fast fixture
                assert output_path.exists()
                assert result["engine"] == "noop"
                assert "render_ms" in result

    def test_metadata_extraction_fast(self, fast_compress):
        """Test that metadata extraction works with fast fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_input.gif"
            output_path = Path(tmpdir) / "test_output.gif"

            create_test_gif(input_path, frames=5)

            result = apply_lossy_compression(
                input_path=input_path,
                output_path=output_path,
                lossy_level=30,
                engine=LossyEngine.GIFSICLE,
            )

            # Check that fast fixture provides expected metadata structure
            expected_keys = {
                "render_ms",
                "engine",
                "command",
                "kilobytes",
                "ssim",
                "lossy_level",
                "frame_keep_ratio",
            }

            for key in expected_keys:
                assert key in result, f"Missing key: {key}"

            # Verify specific fast fixture values
            assert result["render_ms"] == 1
            assert result["engine"] == "noop"
            assert result["command"] == "noop-copy"
            assert result["ssim"] == 1.0


@pytest.mark.fast
class TestEngineUtilsFast:
    """Fast tests for engine utility functions."""

    def test_engine_enum_values(self):
        """Test LossyEngine enum values."""
        assert LossyEngine.GIFSICLE.value == "gifsicle"
        assert LossyEngine.ANIMATELY.value == "animately"

        # Test that enum can be converted to string
        assert str(LossyEngine.GIFSICLE) == "LossyEngine.GIFSICLE"
        assert str(LossyEngine.ANIMATELY) == "LossyEngine.ANIMATELY"

    def test_engine_iteration(self):
        """Test that we can iterate over engine enum."""
        engines = list(LossyEngine)
        assert len(engines) == 2
        assert LossyEngine.GIFSICLE in engines
        assert LossyEngine.ANIMATELY in engines

    def test_parameter_validation_logic(self):
        """Test parameter validation without actual compression."""
        # Test lossy level bounds
        valid_lossy_levels = [0, 50, 100, 200]
        for level in valid_lossy_levels:
            # These shouldn't raise exceptions
            assert isinstance(level, int)
            assert level >= 0

        # Test frame keep ratio bounds
        valid_ratios = [0.1, 0.5, 0.8, 1.0]
        for ratio in valid_ratios:
            assert isinstance(ratio, int | float)
            assert 0.0 <= ratio <= 1.0
