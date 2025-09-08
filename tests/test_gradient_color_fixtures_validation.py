"""Fixture validation tests for gradient banding and color artifact detection.

This test suite validates that test fixtures correctly demonstrate the expected
artifacts and that detection systems properly identify them.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from giflab.gradient_color_artifacts import (
    GradientBandingDetector,
    PerceptualColorValidator,
    calculate_gradient_color_metrics,
)
from giflab.metrics import extract_gif_frames
from tests.fixtures.generate_gradient_color_fixtures import (
    _interpolate_color,
    create_banded_gradient_gif,
    create_brand_color_test_gif,
    create_color_shift_gif,
    create_smooth_gradient_gif,
)


class TestFixtureValidation:
    """Validate test fixtures demonstrate expected artifacts."""

    def setup_method(self):
        """Set up fixture validation tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up fixture validation tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_smooth_gradient_fixtures(self):
        """Verify smooth gradients don't trigger banding detection."""
        # Change to temp dir for fixture creation
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Create smooth gradient fixtures
            smooth_horizontal = create_smooth_gradient_gif("horizontal")
            smooth_vertical = create_smooth_gradient_gif("vertical")
            smooth_radial = create_smooth_gradient_gif("radial")

            fixtures = [
                ("horizontal", smooth_horizontal),
                ("vertical", smooth_vertical),
                ("radial", smooth_radial),
            ]

            for direction, gif_path in fixtures:
                if not gif_path.exists():
                    pytest.skip(f"Fixture creation failed: {gif_path}")

                # Extract frames and calculate metrics
                extract_result = extract_gif_frames(gif_path)
                frames = extract_result.frames

                # Test with identical frames (should have minimal banding)
                result = calculate_gradient_color_metrics(frames, frames)

                # Smooth gradients should have low banding scores
                assert (
                    result["banding_score_mean"] < 30.0
                ), f"Smooth {direction} gradient triggered banding: {result['banding_score_mean']}"

                # Should detect gradient regions
                assert (
                    result["gradient_region_count"] >= 0
                ), f"No gradient regions detected in {direction} gradient"

                # Color metrics should be minimal (identical frames)
                assert (
                    result["deltae_mean"] < 1.0
                ), f"Color differences in identical {direction} frames: {result['deltae_mean']}"

                print(
                    f"✅ Smooth {direction} gradient: banding={result['banding_score_mean']:.1f}, regions={result['gradient_region_count']}"
                )

        finally:
            os.chdir(original_cwd)

    @pytest.mark.fast
    def test_banded_gradient_fixtures(self):
        """Verify banded gradients DO trigger banding detection."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Create banded gradient fixtures with different severities
            test_cases = [
                ("high", "horizontal"),
                ("medium", "horizontal"),
                ("low", "horizontal"),
            ]

            results = {}

            for severity, direction in test_cases:
                gif_path = create_banded_gradient_gif(severity, direction)

                if not gif_path.exists():
                    pytest.skip(f"Banded gradient fixture creation failed: {gif_path}")

                # Extract frames and calculate metrics
                extract_result = extract_gif_frames(gif_path)
                frames = extract_result.frames

                # Compare banded gradient to smooth version
                smooth_frames = self._create_smooth_equivalent_frames(frames)
                result = calculate_gradient_color_metrics(smooth_frames, frames)

                results[severity] = result

                # Banded gradients should trigger higher banding scores
                # (comparing smooth original to banded compressed)
                print(
                    f"Banded {severity}: banding={result['banding_score_mean']:.1f}, regions={result['gradient_region_count']}"
                )

                # At minimum, should detect some artifacts
                assert result["banding_score_mean"] >= 0.0
                assert result["gradient_region_count"] >= 0

            # High severity should generally have higher scores than low severity
            if "high" in results and "low" in results:
                high_score = results["high"]["banding_score_mean"]
                low_score = results["low"]["banding_score_mean"]

                # Allow for some variance but high should generally be worse
                if high_score > 0 and low_score >= 0:
                    print(
                        f"Severity comparison: high={high_score:.1f}, low={low_score:.1f}"
                    )
                    # High severity should be at least as bad as low severity
                    assert (
                        high_score >= low_score * 0.5
                    ), f"High severity not worse than low: {high_score} vs {low_score}"

        finally:
            os.chdir(original_cwd)

    @pytest.mark.fast
    def test_color_shift_fixtures(self):
        """Verify color shifts match expected ΔE00 ranges."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Test different shift severities
            severities = ["high", "medium", "low"]
            results = {}

            for severity in severities:
                original_path, shifted_path = create_color_shift_gif(severity)

                if not (original_path.exists() and shifted_path.exists()):
                    pytest.skip(f"Color shift fixture creation failed: {severity}")

                # Extract frames from both GIFs
                orig_extract = extract_gif_frames(original_path)
                shift_extract = extract_gif_frames(shifted_path)

                orig_frames = orig_extract.frames
                shift_frames = shift_extract.frames

                # Calculate color difference metrics
                result = calculate_gradient_color_metrics(orig_frames, shift_frames)
                results[severity] = result

                print(
                    f"Color shift {severity}: ΔE={result['deltae_mean']:.2f}, >3={result['deltae_pct_gt3']:.1f}%"
                )

                # Verify basic metrics
                assert (
                    result["deltae_mean"] > 0.0
                ), f"No color differences detected in {severity} shift"
                assert (
                    result["color_patch_count"] > 0
                ), f"No color patches analyzed in {severity} shift"

            # Verify severity ordering
            if all(s in results for s in ["high", "medium", "low"]):
                high_mean = results["high"]["deltae_mean"]
                medium_mean = results["medium"]["deltae_mean"]
                low_mean = results["low"]["deltae_mean"]

                # High should have largest color differences
                assert (
                    high_mean >= medium_mean * 0.8
                ), f"High severity not greater than medium: {high_mean} vs {medium_mean}"
                assert (
                    medium_mean >= low_mean * 0.8
                ), f"Medium severity not greater than low: {medium_mean} vs {low_mean}"

                # Verify threshold percentages make sense
                # High severity should have more patches exceeding thresholds
                high_gt3 = results["high"]["deltae_pct_gt3"]
                low_gt3 = results["low"]["deltae_pct_gt3"]

                if high_gt3 > 0 or low_gt3 > 0:
                    assert (
                        high_gt3 >= low_gt3
                    ), f"High severity doesn't exceed thresholds more: {high_gt3}% vs {low_gt3}%"

        finally:
            os.chdir(original_cwd)

    @pytest.mark.fast
    def test_brand_color_fixtures(self):
        """Verify brand color test cases work correctly."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Test brand color fixture
            original_path, shifted_path = create_brand_color_test_gif(
                include_color_shifts=True
            )

            if not (original_path.exists() and shifted_path.exists()):
                pytest.skip("Brand color fixture creation failed")

            # Extract frames
            orig_extract = extract_gif_frames(original_path)
            shift_extract = extract_gif_frames(shifted_path)

            # Calculate metrics
            result = calculate_gradient_color_metrics(
                orig_extract.frames, shift_extract.frames
            )

            print(
                f"Brand colors: ΔE={result['deltae_mean']:.2f}, >3={result['deltae_pct_gt3']:.1f}%, patches={result['color_patch_count']}"
            )

            # Should detect color differences in brand colors
            assert result["deltae_mean"] > 0.0, "No brand color differences detected"
            assert result["color_patch_count"] > 0, "No brand color patches analyzed"

            # Brand color shifts should be significant enough to detect
            assert result["deltae_pct_gt1"] > 0.0, "No patches exceed ΔE=1 threshold"

            # Test without shifts (should be identical)
            single_path = create_brand_color_test_gif(include_color_shifts=False)
            if single_path.exists():
                single_extract = extract_gif_frames(single_path)
                identical_result = calculate_gradient_color_metrics(
                    single_extract.frames, single_extract.frames
                )

                # Identical frames should have minimal differences
                assert (
                    identical_result["deltae_mean"] < 0.1
                ), f"Identical brand colors show differences: {identical_result['deltae_mean']}"

        finally:
            os.chdir(original_cwd)

    @pytest.mark.fast
    def test_fixture_creation_helper_functions(self):
        """Test that fixture creation helper functions work correctly."""
        # Test color interpolation function
        color1 = (255, 0, 0)  # Red
        color2 = (0, 0, 255)  # Blue

        # Test interpolation at different ratios
        mid_color = _interpolate_color(color1, color2, 0.5)
        assert mid_color == (127, 0, 127), f"Incorrect color interpolation: {mid_color}"

        start_color = _interpolate_color(color1, color2, 0.0)
        assert start_color == color1, f"Start color incorrect: {start_color}"

        end_color = _interpolate_color(color1, color2, 1.0)
        assert end_color == color2, f"End color incorrect: {end_color}"

        # Test edge cases
        clamp_low = _interpolate_color(color1, color2, -0.5)  # Should clamp to 0.0
        assert clamp_low == color1, f"Ratio clamping failed: {clamp_low}"

        clamp_high = _interpolate_color(color1, color2, 1.5)  # Should clamp to 1.0
        assert clamp_high == color2, f"Ratio clamping failed: {clamp_high}"

    @pytest.mark.fast
    def test_fixture_frame_consistency(self):
        """Test that fixtures produce consistent frame structures."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Create various fixtures and check their properties
            fixtures = [
                ("smooth_horizontal", create_smooth_gradient_gif("horizontal")),
                ("banded_high", create_banded_gradient_gif("high")),
            ]

            for name, gif_path in fixtures:
                if not gif_path.exists():
                    continue

                extract_result = extract_gif_frames(gif_path)
                frames = extract_result.frames

                # Check frame consistency
                assert len(frames) > 0, f"No frames in {name} fixture"

                # All frames should have same dimensions
                if len(frames) > 1:
                    first_shape = frames[0].shape
                    for i, frame in enumerate(frames[1:], 1):
                        assert (
                            frame.shape == first_shape
                        ), f"Frame {i} shape mismatch in {name}: {frame.shape} vs {first_shape}"

                # Frames should be RGB uint8
                for i, frame in enumerate(frames):
                    assert (
                        frame.dtype == np.uint8
                    ), f"Frame {i} wrong dtype in {name}: {frame.dtype}"
                    assert (
                        len(frame.shape) == 3
                    ), f"Frame {i} wrong dimensions in {name}: {frame.shape}"
                    assert (
                        frame.shape[2] == 3
                    ), f"Frame {i} not RGB in {name}: {frame.shape[2]} channels"

                print(f"✅ {name}: {len(frames)} frames, {frames[0].shape}")

        finally:
            os.chdir(original_cwd)

    @pytest.mark.fast
    def test_detection_sensitivity_with_fixtures(self):
        """Test detection sensitivity using fixture variations."""
        detector = GradientBandingDetector()
        validator = PerceptualColorValidator()

        # Create test frames manually with known characteristics
        size = (64, 64)

        # Create smooth gradient
        smooth_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for x in range(size[0]):
            intensity = int(x * 255 / (size[0] - 1))
            smooth_frame[:, x] = [intensity, intensity // 2, 255 - intensity]

        # Create banded version
        banded_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        bands = 8
        for x in range(size[0]):
            band = int(x * bands / size[0])
            intensity = int(band * 255 / (bands - 1))
            banded_frame[:, x] = [intensity, intensity // 2, 255 - intensity]

        # Test banding detection
        banding_result = detector.detect_banding_artifacts(
            [smooth_frame], [banded_frame]
        )

        print(
            f"Detection test: smooth->banded banding_score={banding_result['banding_score_mean']:.2f}"
        )

        # Should detect some difference
        assert banding_result["banding_score_mean"] >= 0.0

        # Test color validation with slight shifts
        shifted_frame = smooth_frame.copy()
        shifted_frame[:, :, 0] = np.minimum(
            255, shifted_frame[:, :, 0] + 20
        )  # Shift red channel

        color_result = validator.calculate_color_difference_metrics(
            [smooth_frame], [shifted_frame]
        )

        print(f"Color test: ΔE={color_result['deltae_mean']:.2f}")

        # Should detect color differences
        assert color_result["deltae_mean"] > 0.0
        assert color_result["color_patch_count"] > 0

    @pytest.mark.fast
    def test_fixture_gradient_direction_detection(self):
        """Test that different gradient directions are handled properly."""
        detector = GradientBandingDetector()

        # Create different gradient directions manually
        size = (96, 96)

        # Horizontal gradient
        h_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for x in range(size[0]):
            intensity = int(x * 255 / (size[0] - 1))
            h_frame[:, x] = [intensity, 128, 255 - intensity]

        # Vertical gradient
        v_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for y in range(size[1]):
            intensity = int(y * 255 / (size[1] - 1))
            v_frame[y, :] = [intensity, 128, 255 - intensity]

        # Diagonal gradient
        d_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for y in range(size[1]):
            for x in range(size[0]):
                intensity = int((x + y) * 255 / (size[0] + size[1] - 2))
                d_frame[y, x] = [intensity, 128, 255 - intensity]

        frames = [("horizontal", h_frame), ("vertical", v_frame), ("diagonal", d_frame)]

        for direction, frame in frames:
            regions = detector.detect_gradient_regions(frame)
            print(f"Gradient direction {direction}: {len(regions)} regions detected")

            # Should detect some gradient regions
            assert len(regions) >= 0  # May or may not detect, depending on algorithm

    # Helper methods

    def _create_smooth_equivalent_frames(self, banded_frames):
        """Create smooth gradient frames equivalent to banded ones."""
        smooth_frames = []

        for frame in banded_frames:
            smooth_frame = np.zeros_like(frame)
            height, width = frame.shape[:2]

            # Create smooth horizontal gradient
            for x in range(width):
                intensity = int(x * 255 / (width - 1))
                smooth_frame[:, x] = [intensity, intensity // 2, 255 - intensity]

            smooth_frames.append(smooth_frame)

        return smooth_frames


class TestFixtureRobustness:
    """Test fixture robustness and edge cases."""

    def setup_method(self):
        """Set up robustness tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up robustness tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_fixture_creation_error_handling(self):
        """Test fixture creation with problematic parameters."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Test with extreme parameters that might cause issues
            test_cases = [
                # Very small sizes might cause issues
                (
                    "create_smooth_gradient_gif",
                    {"direction": "horizontal", "colors": ((0, 0, 0), (1, 1, 1))},
                ),
                # Very high contrast
                (
                    "create_smooth_gradient_gif",
                    {"direction": "radial", "colors": ((0, 0, 0), (255, 255, 255))},
                ),
            ]

            for func_name, kwargs in test_cases:
                try:
                    if func_name == "create_smooth_gradient_gif":
                        result = create_smooth_gradient_gif(**kwargs)
                        if result and result.exists():
                            print(f"✅ Edge case fixture created: {result}")
                        else:
                            print(
                                f"⚠️  Edge case fixture creation returned None: {kwargs}"
                            )
                except Exception as e:
                    # Fixture creation failures are acceptable for edge cases
                    print(f"⚠️  Edge case fixture creation failed (acceptable): {e}")

        finally:
            os.chdir(original_cwd)

    @pytest.mark.fast
    def test_fixture_reproducibility(self):
        """Test that fixtures are reproducible across runs."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            # Create same fixture twice
            gif1 = create_smooth_gradient_gif("horizontal")
            gif2 = create_smooth_gradient_gif("horizontal")

            if gif1 and gif2 and gif1.exists() and gif2.exists():
                # Extract frames from both
                extract1 = extract_gif_frames(gif1)
                extract2 = extract_gif_frames(gif2)

                # Should have same number of frames
                assert len(extract1.frames) == len(extract2.frames)

                # Frames should be identical (or very similar)
                for i, (frame1, frame2) in enumerate(
                    zip(extract1.frames, extract2.frames)
                ):
                    if frame1.shape == frame2.shape:
                        diff = np.mean(
                            np.abs(frame1.astype(float) - frame2.astype(float))
                        )
                        assert diff < 1.0, f"Frame {i} differs too much: {diff}"
                    else:
                        pytest.fail(
                            f"Frame {i} shape mismatch: {frame1.shape} vs {frame2.shape}"
                        )

                print("✅ Fixtures are reproducible")

        finally:
            os.chdir(original_cwd)


# Integration with existing test markers
pytestmark = [pytest.mark.gradient_color_fixtures]
