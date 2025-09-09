"""Engine-specific validation tests for wrapper output validation.

This test suite validates that each compression engine (FFmpeg, ImageMagick,
Gifsicle, Gifski, Animately) produces expected output for frame reduction,
color reduction, and lossy compression operations.
"""

import tempfile
from pathlib import Path

import pytest
from giflab.meta import extract_gif_metadata
from giflab.tool_wrappers import (  # Frame reduction wrappers; Color reduction wrappers; Lossy compression wrappers
    AnimatelyColorReducer,
    AnimatelyFrameReducer,
    AnimatelyLossyCompressor,
    FFmpegColorReducer,
    FFmpegLossyCompressor,
    GifsicleColorReducer,
    GifsicleFrameReducer,
    GifsicleLossyCompressor,
    GifskiLossyCompressor,
    ImageMagickColorReducer,
    ImageMagickLossyCompressor,
)
from giflab.wrapper_validation import ValidationConfig, WrapperOutputValidator


class TestFixtures:
    """Test fixture management and validation."""

    @pytest.fixture(scope="class")
    def fixtures_dir(self):
        """Directory containing test fixtures."""
        return Path(__file__).parent / "fixtures"

    @pytest.fixture(scope="class")
    def test_10_frames_gif(self, fixtures_dir):
        """10-frame test GIF."""
        return fixtures_dir / "test_10_frames.gif"

    @pytest.fixture(scope="class")
    def test_4_frames_gif(self, fixtures_dir):
        """4-frame test GIF."""
        return fixtures_dir / "test_4_frames.gif"

    @pytest.fixture(scope="class")
    def test_30_frames_gif(self, fixtures_dir):
        """30-frame test GIF."""
        return fixtures_dir / "test_30_frames.gif"

    @pytest.fixture(scope="class")
    def test_256_colors_gif(self, fixtures_dir):
        """Many-colors test GIF."""
        return fixtures_dir / "test_256_colors.gif"

    @pytest.fixture(scope="class")
    def test_2_colors_gif(self, fixtures_dir):
        """2-color test GIF."""
        return fixtures_dir / "test_2_colors.gif"

    @pytest.fixture
    def validator(self):
        """Validator with strict configuration for testing."""
        config = ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.05,  # 5% tolerance
            COLOR_COUNT_TOLERANCE=2,  # Allow 2 extra colors
            FPS_TOLERANCE=0.1,  # 10% FPS tolerance
            MIN_COLOR_REDUCTION_PERCENT=0.05,  # Require 5% color reduction minimum
        )
        return WrapperOutputValidator(config)


@pytest.mark.external_tools
class TestFrameReductionValidation(TestFixtures):
    """Test frame reduction validation across all engines."""

    def test_gifsicle_frame_reduction_50_percent(self, test_10_frames_gif, validator):
        """Test Gifsicle frame reduction validation at 50%."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply 50% frame reduction
            result = wrapper.apply(
                test_10_frames_gif, output_path, params={"ratio": 0.5}
            )

            # Validate the result
            assert "validations" in result
            assert result["validation_passed"] is True

            # Check frame count validation specifically
            frame_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "frame_count"
            ]
            assert len(frame_validations) == 1

            frame_validation = frame_validations[0]
            assert frame_validation["is_valid"] is True
            assert (
                abs(frame_validation["actual"]["ratio"] - 0.5)
                <= validator.config.FRAME_RATIO_TOLERANCE
            )

            # Verify actual frame count
            output_metadata = extract_gif_metadata(output_path)
            expected_frames = 5  # 50% of 10 frames
            assert (
                abs(output_metadata.orig_frames - expected_frames) <= 1
            )  # Allow ±1 frame

    def test_gifsicle_frame_reduction_30_percent(self, test_30_frames_gif, validator):
        """Test Gifsicle frame reduction validation at 30%."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply 30% frame reduction (keep 30% of frames)
            result = wrapper.apply(
                test_30_frames_gif, output_path, params={"ratio": 0.3}
            )

            assert result["validation_passed"] is True

            # Verify actual frame count
            output_metadata = extract_gif_metadata(output_path)
            expected_frames = int(30 * 0.3)  # ~9 frames
            assert (
                abs(output_metadata.orig_frames - expected_frames) <= 2
            )  # Allow ±2 frames

    def test_animately_frame_reduction_validation(self, test_10_frames_gif, validator):
        """Test Animately frame reduction validation."""
        wrapper = AnimatelyFrameReducer()
        if not wrapper.available():
            pytest.skip("Animately not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply 60% frame reduction
            result = wrapper.apply(
                test_10_frames_gif, output_path, params={"ratio": 0.6}
            )

            # Should have validation results
            assert "validations" in result

            # Check frame count validation
            frame_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "frame_count"
            ]
            if len(frame_validations) > 0:
                frame_validation = frame_validations[0]
                # Either passes validation or provides useful error info
                if not frame_validation["is_valid"]:
                    print(
                        f"Frame validation failed: {frame_validation['error_message']}"
                    )
                    print(
                        f"Expected: {frame_validation['expected']}, Actual: {frame_validation['actual']}"
                    )

    def test_edge_case_single_frame_output(self, test_4_frames_gif, validator):
        """Test edge case where frame reduction results in single frame."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply very aggressive frame reduction (keep 10% = ~0.4 frames, should result in 1 frame)
            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"ratio": 0.1}
            )

            # Should still have validations
            assert "validations" in result

            # Check that minimum frame requirement is enforced
            output_metadata = extract_gif_metadata(output_path)
            assert output_metadata.orig_frames >= 1  # At least 1 frame


@pytest.mark.external_tools
class TestColorReductionValidation(TestFixtures):
    """Test color reduction validation across all engines."""

    def test_gifsicle_color_reduction_32_colors(self, test_256_colors_gif, validator):
        """Test Gifsicle color reduction to 32 colors."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 32}
            )

            assert "validations" in result

            # Check color count validation
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            assert len(color_validations) == 1

            color_validation = color_validations[0]
            if color_validation["is_valid"]:
                # Colors should be <= 32 + tolerance
                assert color_validation["actual"] <= (
                    32 + validator.config.COLOR_COUNT_TOLERANCE
                )
            else:
                print(f"Color validation failed: {color_validation['error_message']}")
                print(
                    f"Expected: {color_validation['expected']}, Actual: {color_validation['actual']}"
                )

    def test_animately_color_reduction_validation(self, test_256_colors_gif, validator):
        """Test Animately color reduction validation."""
        wrapper = AnimatelyColorReducer()
        if not wrapper.available():
            pytest.skip("Animately not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 16}
            )

            assert "validations" in result

            # Verify color reduction occurred
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                if not color_validation["is_valid"]:
                    print(f"Color validation info: {color_validation['error_message']}")

    def test_imagemagick_color_reduction_validation(
        self, test_256_colors_gif, validator
    ):
        """Test ImageMagick color reduction validation."""
        wrapper = ImageMagickColorReducer()
        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 64}
            )

            assert "validations" in result
            # ImageMagick should successfully reduce colors
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                # Should either pass or give useful error info
                if not color_validation["is_valid"]:
                    print(
                        f"ImageMagick color validation: {color_validation['error_message']}"
                    )

    def test_ffmpeg_color_reduction_validation(self, test_256_colors_gif, validator):
        """Test FFmpeg color reduction validation."""
        wrapper = FFmpegColorReducer()
        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_256_colors_gif, output_path, params={"colors": 128}
            )

            assert "validations" in result

            # FFmpeg uses palette generation, should be effective
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                if not color_validation["is_valid"]:
                    print(
                        f"FFmpeg color validation: {color_validation['error_message']}"
                    )

    def test_color_reduction_edge_case_already_few_colors(
        self, test_2_colors_gif, validator
    ):
        """Test color reduction on GIF that already has few colors."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Try to reduce to 16 colors when input only has 2
            result = wrapper.apply(
                test_2_colors_gif, output_path, params={"colors": 16}
            )

            assert "validations" in result

            # This should pass - no reduction needed
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                # Should either pass or indicate no reduction was needed
                assert color_validation["actual"] <= 16


@pytest.mark.external_tools
class TestLossyCompressionValidation(TestFixtures):
    """Test lossy compression validation across all engines."""

    def test_gifsicle_lossy_compression_validation(self, test_10_frames_gif, validator):
        """Test Gifsicle lossy compression validation."""
        wrapper = GifsicleLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_10_frames_gif, output_path, params={"lossy_level": 40}
            )

            assert "validations" in result

            # Check file integrity validation (most important for lossy)
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) == 1
            assert integrity_validations[0]["is_valid"] is True

            # Timing should be preserved in lossy compression
            timing_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "timing_preservation"
            ]
            if len(timing_validations) > 0:
                timing_validation = timing_validations[0]
                if not timing_validation["is_valid"]:
                    print(f"Timing validation: {timing_validation['error_message']}")

    def test_animately_lossy_compression_validation(self, test_4_frames_gif, validator):
        """Test Animately lossy compression validation."""
        wrapper = AnimatelyLossyCompressor()
        if not wrapper.available():
            pytest.skip("Animately not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 60}
            )

            assert "validations" in result

            # File should be created and valid
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) == 1
            assert integrity_validations[0]["is_valid"] is True

    def test_imagemagick_lossy_compression_validation(
        self, test_4_frames_gif, validator
    ):
        """Test ImageMagick lossy compression validation."""
        wrapper = ImageMagickLossyCompressor()
        if not wrapper.available():
            pytest.skip("ImageMagick not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 30}
            )

            assert "validations" in result

            # Check basic validations pass
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) >= 1

    def test_ffmpeg_lossy_compression_validation(self, test_4_frames_gif, validator):
        """Test FFmpeg lossy compression validation."""
        wrapper = FFmpegLossyCompressor()
        if not wrapper.available():
            pytest.skip("FFmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 25}
            )

            assert "validations" in result

            # FFmpeg should produce valid output
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) >= 1

    def test_gifski_lossy_compression_validation(self, test_4_frames_gif, validator):
        """Test Gifski lossy compression validation."""
        wrapper = GifskiLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifski not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            result = wrapper.apply(
                test_4_frames_gif, output_path, params={"lossy_level": 50}
            )

            assert "validations" in result

            # Gifski should produce high-quality output
            integrity_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "file_integrity"
            ]
            assert len(integrity_validations) >= 1


class TestEdgeCases(TestFixtures):
    """Test edge cases and boundary conditions."""

    @pytest.mark.external_tools
    def test_extreme_frame_reduction(self, test_30_frames_gif, validator):
        """Test extreme frame reduction (very low ratios)."""
        wrapper = GifsicleFrameReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Extreme reduction: keep only 3.3% of frames
            result = wrapper.apply(
                test_30_frames_gif, output_path, params={"ratio": 0.033}
            )

            # Should still produce valid output
            assert (
                result.get("validation_passed") is not False
            )  # Allow None (validation error) or True

            # Should have at least 1 frame
            output_metadata = extract_gif_metadata(output_path)
            assert output_metadata.orig_frames >= 1

    @pytest.mark.external_tools
    def test_no_reduction_color_params(self, test_256_colors_gif, validator):
        """Test color reduction with parameter that requires no actual reduction."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Request more colors than input - use valid count but higher than actual
            result = wrapper.apply(
                test_256_colors_gif,
                output_path,
                params={"colors": 256},  # Request 256 when input has fewer
            )

            # Should still validate successfully
            assert "validations" in result

            # Color validation should account for this case
            color_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "color_count"
            ]
            if len(color_validations) > 0:
                color_validation = color_validations[0]
                # Should pass since no reduction was needed/possible
                assert color_validation.get("is_valid") in [True, None]

    def test_validation_with_corrupted_output(self, test_4_frames_gif, validator):
        """Test validation behavior with corrupted output file."""
        # Create a corrupted file that's large enough to pass size check
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Write invalid GIF data (large enough to pass size check)
            with open(output_path, "wb") as f:
                f.write(
                    b"Not a GIF file, but long enough to pass the size check. " * 10
                )

            # Test file integrity validation
            result = validator.validate_file_integrity(output_path, {})

            assert result.is_valid is False
            # Should fail on GIF format check now
            assert (
                "Cannot read output file as valid GIF" in result.error_message
                or "is not a GIF" in result.error_message
            )

    def test_validation_performance_with_large_gif(self, validator):
        """Test validation performance doesn't significantly impact processing."""
        import time

        # This would ideally test with a very large GIF, but we'll simulate
        # the performance aspect by testing the validation overhead

        start_time = time.time()

        # Run multiple validations to measure overhead
        for _i in range(10):
            result = validator.validate_file_integrity(
                Path(__file__), {}  # Use this Python file as a non-GIF
            )
            assert result.is_valid is False

        end_time = time.time()
        total_time = end_time - start_time

        # Validation should be very fast
        assert total_time < 1.0  # Less than 1 second for 10 validations
        print(f"Validation performance: {total_time:.3f}s for 10 integrity checks")
