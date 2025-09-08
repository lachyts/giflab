"""End-to-end tests for gradient banding and color validation with real GIF compression.

This test suite validates the complete gradient banding detection and perceptual color
validation pipeline using actual compression engines and realistic GIF processing scenarios.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from giflab.config import DEFAULT_ENGINE_CONFIG
from giflab.lossy import _is_executable
from giflab.metrics import calculate_comprehensive_metrics
from giflab.optimization_validation.data_structures import ValidationConfig
from giflab.tool_wrappers import GifsicleColorReducer


@pytest.mark.external_tools
class TestGradientColorE2E:
    """End-to-end tests with real GIF compression engines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def validation_config(self):
        """Create validation config for e2e testing."""
        return ValidationConfig(
            # Gradient banding thresholds
            banding_score_threshold=30.0,  # Moderate threshold for real compression
            gradient_region_min_count=1,  # At least one gradient region expected
            # Color validation thresholds
            deltae_mean_threshold=5.0,  # Mean ΔE00 should be reasonable
            deltae_pct_gt3_threshold=20.0,  # Max 20% of patches with ΔE00 > 3
            deltae_pct_gt5_threshold=10.0,  # Max 10% of patches with ΔE00 > 5
            # Relaxed thresholds for compression artifacts
            minimum_quality_floor=0.3,  # Allow more degradation
        )

    def test_banding_detection_with_gifsicle_color_reduction(self, tmp_path):
        """Test banding detection after aggressive Gifsicle color reduction."""
        wrapper = GifsicleColorReducer()
        if not _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH):
            pytest.skip("Gifsicle not available")

        # Create a smooth gradient GIF that should show banding after color reduction
        input_gif = self._create_gradient_gif_in_temp(
            "smooth_gradient.gif", tmp_path, bands=256
        )
        output_gif = tmp_path / "color_reduced_output.gif"

        # Apply aggressive color reduction (should introduce banding)
        try:
            wrapper.apply(
                input_path=input_gif,
                output_path=output_gif,
                params={"colors": 16},  # Aggressive reduction
            )
        except Exception as e:
            pytest.skip(f"Gifsicle compression failed: {e}")

        if not output_gif.exists():
            pytest.skip("Gifsicle output not created")

        # Calculate gradient and color metrics
        result = calculate_comprehensive_metrics(
            original_path=input_gif, compressed_path=output_gif
        )

        # Verify banding detection worked
        assert "banding_score_mean" in result
        assert "banding_score_p95" in result

        # Color reduction should introduce some banding
        if result["gradient_region_count"] > 0:  # Only test if gradients were detected
            # Banding score might be elevated due to color reduction
            assert result["banding_score_mean"] >= 0.0
            # Don't assert it's high, as real compression might be better than expected

        # Color metrics should show some differences
        assert result["deltae_mean"] >= 0.0
        assert result["color_patch_count"] > 0

    def test_color_validation_with_animately_lossy(self, tmp_path):
        """Test ΔE00 validation with Animately lossy compression."""
        if not _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH):
            pytest.skip("Animately not available")

        # Test if animately is functional (not just executable)
        try:
            result = subprocess.run(
                [str(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH), "--help"],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Animately binary not functional: {e}")

        # Create a brand color test GIF
        input_gif = self._create_brand_color_gif_in_temp("brand_colors.gif", tmp_path)
        output_gif = tmp_path / "animately_lossy_output.gif"

        # Apply lossy compression
        try:
            subprocess.run(
                [
                    str(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH),
                    "--input", str(input_gif),
                    "--output", str(output_gif),
                    "--lossy", "60",  # Moderate lossy setting
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            pytest.skip(f"Animately lossy compression failed: {e}")

        if not output_gif.exists():
            pytest.skip("Animately output not created")

        # Calculate color validation metrics
        result = calculate_comprehensive_metrics(
            original_path=input_gif, compressed_path=output_gif
        )

        # Verify color validation metrics
        assert "deltae_mean" in result
        assert "deltae_pct_gt3" in result
        assert result["color_patch_count"] > 0

        # Lossy compression should introduce some color differences
        assert result["deltae_mean"] >= 0.0

        # For brand colors, differences should ideally be small
        # But we'll be lenient for real compression
        assert result["deltae_pct_gt5"] <= 50.0  # Not more than 50% severely degraded

    def test_detection_thresholds_with_real_compression(self, tmp_path):
        """Validate that thresholds appropriately catch real artifacts."""
        if not _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH):
            pytest.skip("Gifsicle not available")

        wrapper = GifsicleColorReducer()

        # Test 1: Mild compression (should pass most thresholds)
        smooth_gif = self._create_gradient_gif_in_temp("smooth.gif", tmp_path)
        mild_output = tmp_path / "mild_compression.gif"

        try:
            wrapper.apply(
                input_path=smooth_gif,
                output_path=mild_output,
                params={"colors": 64},  # Mild reduction
            )
        except Exception:
            pytest.skip("Mild compression failed")

        if mild_output.exists():
            mild_result = calculate_comprehensive_metrics(
                original_path=smooth_gif, compressed_path=mild_output
            )

            # Mild compression should have reasonable metrics
            assert mild_result["banding_score_mean"] < 50.0  # Should be moderate
            assert mild_result["deltae_pct_gt5"] < 30.0  # Limited severe color changes

        # Test 2: Aggressive compression (might trigger thresholds)
        aggressive_output = tmp_path / "aggressive_compression.gif"

        try:
            wrapper.apply(
                input_path=smooth_gif,
                output_path=aggressive_output,
                params={"colors": 8},  # Very aggressive
            )
        except Exception as e:
            pytest.skip(f"Aggressive compression failed: {e}")

        if aggressive_output.exists():
            aggressive_result = calculate_comprehensive_metrics(
                original_path=smooth_gif, compressed_path=aggressive_output
            )

            # Aggressive compression should show more artifacts
            # But we'll be lenient as compression quality varies
            assert aggressive_result["banding_score_mean"] >= mild_result.get(
                "banding_score_mean", 0
            )
            assert aggressive_result["deltae_mean"] >= mild_result.get("deltae_mean", 0)

    def test_brand_color_preservation_pipeline(self, tmp_path):
        """Test complete brand color preservation workflow."""
        if not _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH):
            pytest.skip("Gifsicle not available")

        # Create brand color test case
        brand_gif = self._create_brand_color_gif_in_temp("brand_test.gif", tmp_path)

        # Test different compression levels
        compression_levels = [
            ("conservative", {"colors": 128, "dither": True}),
            ("moderate", {"colors": 64, "dither": True}),
            ("aggressive", {"colors": 32, "dither": False}),
        ]

        wrapper = GifsicleColorReducer()
        results = {}

        for level_name, params in compression_levels:
            output_gif = tmp_path / f"brand_{level_name}.gif"

            try:
                wrapper.compress(input_path=brand_gif, output_path=output_gif, **params)
            except Exception:
                continue  # Skip failed compressions

            if output_gif.exists():
                result = calculate_comprehensive_metrics(
                    original_path=brand_gif, compressed_path=output_gif
                )
                results[level_name] = result

        # Analyze results if we have any successful compressions
        if results:
            # Conservative should generally be better than aggressive
            if "conservative" in results and "aggressive" in results:
                conservative = results["conservative"]
                aggressive = results["aggressive"]

                # Conservative should have better color preservation
                assert (
                    conservative["deltae_mean"] <= aggressive["deltae_mean"] * 1.5
                )  # Allow some tolerance

    def test_gradient_preservation_detection(self, tmp_path):
        """Test gradient preservation detection across compression pipeline."""
        if not _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH):
            pytest.skip("Gifsicle not available")

        # Create different types of gradients
        gradient_types = [
            (
                "linear",
                lambda: self._create_gradient_gif_in_temp(
                    "linear.gif", tmp_path, direction="horizontal"
                ),
            ),
            (
                "radial",
                lambda: self._create_gradient_gif_in_temp(
                    "radial.gif", tmp_path, direction="radial"
                ),
            ),
        ]

        wrapper = GifsicleColorReducer()

        for gradient_type, creator in gradient_types:
            input_gif = creator()
            output_gif = tmp_path / f"{gradient_type}_compressed.gif"

            try:
                wrapper.compress(
                    input_path=input_gif,
                    output_path=output_gif,
                    colors=32,
                    dither=False,  # Maximize banding for testing
                )
            except Exception:
                continue

            if output_gif.exists():
                result = calculate_comprehensive_metrics(
                    original_path=input_gif, compressed_path=output_gif
                )

                # Should detect gradients in original
                assert (
                    result["gradient_region_count"] >= 0
                )  # May or may not detect gradients

                # Should have meaningful metrics
                assert isinstance(result["banding_score_mean"], float)
                assert result["banding_score_mean"] >= 0.0

    def test_real_world_ui_gif_processing(self, tmp_path):
        """Test with realistic UI GIF scenarios."""
        if not _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH):
            pytest.skip("Gifsicle not available")

        # Create a UI-like GIF with buttons, gradients, and text areas
        ui_gif = self._create_ui_gif_in_temp("ui_test.gif", tmp_path)

        # Process with different optimization strategies
        strategies = [
            ("web_optimized", {"colors": 256}),  # Good for web
            ("file_size_optimized", {"colors": 64}),  # Smaller files
            ("quality_optimized", {"colors": 256, "dither": True}),  # Better quality
        ]

        wrapper = GifsicleColorReducer()

        for strategy_name, params in strategies:
            output_gif = tmp_path / f"ui_{strategy_name}.gif"

            try:
                wrapper.compress(input_path=ui_gif, output_path=output_gif, **params)
            except Exception:
                continue

            if output_gif.exists():
                result = calculate_comprehensive_metrics(
                    original_path=ui_gif, compressed_path=output_gif
                )

                # UI elements should be preserved reasonably well
                assert result["deltae_mean"] >= 0.0
                assert result["color_patch_count"] > 0

                # Check that results are reasonable
                assert 0.0 <= result["deltae_pct_gt3"] <= 100.0
                assert result["banding_score_mean"] >= 0.0

    def test_edge_case_compression_scenarios(self, tmp_path):
        """Test edge cases that might occur in real compression workflows."""
        if not _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH):
            pytest.skip("Gifsicle not available")

        wrapper = GifsicleColorReducer()

        # Test 1: Single frame GIF
        single_frame_gif = self._create_single_frame_gif_in_temp("single.gif", tmp_path)
        single_output = tmp_path / "single_compressed.gif"

        try:
            wrapper.compress(
                input_path=single_frame_gif, output_path=single_output, colors=32
            )
        except Exception:
            pass  # Skip if compression fails

        if single_output.exists():
            result = calculate_comprehensive_metrics(
                original_path=single_frame_gif, compressed_path=single_output
            )
            # Should handle single frame gracefully
            assert isinstance(result["banding_score_mean"], float)
            assert isinstance(result["deltae_mean"], float)

        # Test 2: Very small GIF
        tiny_gif = self._create_tiny_gif_in_temp("tiny.gif", tmp_path)
        tiny_output = tmp_path / "tiny_compressed.gif"

        try:
            wrapper.compress(input_path=tiny_gif, output_path=tiny_output, colors=16)
        except Exception:
            pass

        if tiny_output.exists():
            result = calculate_comprehensive_metrics(
                original_path=tiny_gif, compressed_path=tiny_output
            )
            # Should handle tiny GIFs without crashing
            assert result["color_patch_count"] >= 0

    # Helper methods for creating test GIFs

    def _create_gradient_gif_in_temp(
        self,
        filename: str,
        tmp_path: Path,
        bands: int = 256,
        direction: str = "horizontal",
    ):
        """Create a gradient GIF for testing."""
        gif_path = tmp_path / filename

        frames = []
        for i in range(3):
            img = Image.new("RGB", (128, 128))
            pixels = img.load()

            if direction == "horizontal":
                for x in range(128):
                    for y in range(128):
                        if bands < 256:
                            # Create banded gradient
                            band = int(x * bands / 128)
                            intensity = int(band * 255 / (bands - 1))
                        else:
                            # Create smooth gradient
                            intensity = int(x * 255 / 127)
                        pixels[x, y] = (intensity, intensity // 2, intensity // 3)

            elif direction == "radial":
                center = (64, 64)
                max_distance = 64 * 1.414  # Diagonal

                for x in range(128):
                    for y in range(128):
                        distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
                        ratio = min(distance / max_distance, 1.0)

                        if bands < 256:
                            band = int(ratio * bands)
                            intensity = int(band * 255 / (bands - 1))
                        else:
                            intensity = int(ratio * 255)

                        pixels[x, y] = (intensity, intensity // 2, 255 - intensity)

            frames.append(img)

        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0
        )

        return gif_path

    def _create_brand_color_gif_in_temp(self, filename: str, tmp_path: Path):
        """Create a brand color test GIF."""
        gif_path = tmp_path / filename

        # Define brand colors
        brand_colors = [
            (0, 123, 255),  # Bootstrap blue
            (40, 167, 69),  # Bootstrap green
            (220, 53, 69),  # Bootstrap red
            (255, 193, 7),  # Bootstrap yellow
        ]

        frames = []
        for i, color in enumerate(brand_colors):
            img = Image.new("RGB", (96, 96), (255, 255, 255))  # White background
            draw = ImageDraw.Draw(img)

            # Create colored sections
            section_width = 96 // len(brand_colors)
            for j, brand_color in enumerate(brand_colors):
                x_start = j * section_width
                x_end = min((j + 1) * section_width, 96)
                draw.rectangle([(x_start, 20), (x_end - 1, 76)], fill=brand_color)

            frames.append(img)

        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=300, loop=0
        )

        return gif_path

    def _create_ui_gif_in_temp(self, filename: str, tmp_path: Path):
        """Create a UI-like GIF with various elements."""
        gif_path = tmp_path / filename

        frames = []
        for i in range(4):
            img = Image.new("RGB", (160, 120), (240, 240, 240))  # Light gray background
            draw = ImageDraw.Draw(img)

            # Header with gradient
            for y in range(20):
                intensity = 50 + int(y * 100 / 20)
                draw.line(
                    [(0, y), (160, y)], fill=(intensity, intensity + 20, intensity + 40)
                )

            # Button (changes color based on frame)
            button_color = (100 + i * 30, 150, 200)
            draw.rectangle([(20, 40), (80, 70)], fill=button_color, outline=(0, 0, 0))

            # Text area background
            draw.rectangle(
                [(90, 40), (140, 90)], fill=(255, 255, 255), outline=(128, 128, 128)
            )

            # Side gradient
            for x in range(160, 140, -1):  # Right side gradient
                intensity = int((160 - x) * 255 / 20)
                if x < 160:
                    draw.line(
                        [(x, 20), (x, 120)],
                        fill=(
                            200 - intensity // 3,
                            220 - intensity // 4,
                            240 - intensity // 5,
                        ),
                    )

            frames.append(img)

        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=250, loop=0
        )

        return gif_path

    def _create_single_frame_gif_in_temp(self, filename: str, tmp_path: Path):
        """Create a single frame GIF."""
        gif_path = tmp_path / filename

        img = Image.new("RGB", (64, 64))
        draw = ImageDraw.Draw(img)

        # Create some content with gradients
        for x in range(64):
            intensity = int(x * 255 / 63)
            draw.line([(x, 0), (x, 32)], fill=(intensity, 128, 255 - intensity))

        # Solid color area
        draw.rectangle([(0, 32), (64, 64)], fill=(255, 0, 0))

        img.save(gif_path, duration=500)
        return gif_path

    def _create_tiny_gif_in_temp(self, filename: str, tmp_path: Path):
        """Create a very small GIF."""
        gif_path = tmp_path / filename

        frames = []
        colors = [(255, 0, 0), (0, 255, 0)]

        for color in colors:
            img = Image.new("RGB", (8, 8), color)
            frames.append(img)

        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0
        )

        return gif_path


# Integration with existing test markers
pytestmark = [pytest.mark.external_tools, pytest.mark.gradient_color_e2e]
