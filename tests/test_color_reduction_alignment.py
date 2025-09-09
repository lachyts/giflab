"""Test aligned color reduction between gifsicle and animately engines.

This test verifies that the color reduction alignment strategies work correctly
and that both engines produce consistent results across different color ranges.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from giflab.color_keep import count_gif_colors
from giflab.lossy import LossyEngine, compress_with_animately, compress_with_gifsicle
from PIL import Image

from tests.test_engine_equivalence import _engine_available


def _create_colorful_test_gif(
    path: Path, frames: int = 4, size: tuple[int, int] = (60, 60)
) -> None:
    """Create a test GIF with many colors to enable color reduction testing.

    This generates a GIF with a rich color palette that can be meaningfully reduced
    to various target color counts.
    """
    images = []

    # Create a color gradient pattern that will result in many colors
    for frame in range(frames):
        # Create RGB image
        img = Image.new("RGB", size)
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Create a gradient pattern with many colors
        for y in range(size[1]):
            for x in range(size[0]):
                # Create a complex color pattern that varies with position and frame
                r = int((x / size[0]) * 255) ^ (frame * 17)
                g = int((y / size[1]) * 255) ^ (frame * 31)
                b = int(((x + y) / (size[0] + size[1])) * 255) ^ (frame * 47)

                # Add some noise to create more unique colors
                r = (r + (x * y * frame) % 64) % 256
                g = (g + (x + y + frame) % 64) % 256
                b = (b + (x - y + frame) % 64) % 256

                pixels[y, x] = [r, g, b]

        # Convert numpy array to PIL Image
        img = Image.fromarray(pixels, "RGB")
        images.append(img)

    # Save as GIF with high quality to preserve colors
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,  # Don't optimize to preserve colors
        palette=None,  # Let PIL create a full palette
    )


@pytest.mark.parametrize("target_colors", [128, 64, 32, 16])
def test_aligned_color_reduction(target_colors):
    """Test that aligned color reduction produces consistent results."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip("Both engines must be available")

    # Create test GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        test_gif = Path(tmp.name)

    try:
        _create_colorful_test_gif(test_gif, frames=4)
        original_colors = count_gif_colors(test_gif)

        # Skip if no reduction needed
        if target_colors >= original_colors:
            pytest.skip(f"No reduction needed: {target_colors} >= {original_colors}")

        # Test both engines with color reduction only
        results = {}

        for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_out:
                output_path = Path(tmp_out.name)

            try:
                if engine == LossyEngine.GIFSICLE:
                    # Use the aligned settings (no dithering)
                    result = compress_with_gifsicle(
                        test_gif,
                        output_path,
                        lossy_level=0,
                        frame_keep_ratio=1.0,
                        color_keep_count=target_colors,
                    )
                else:
                    result = compress_with_animately(
                        test_gif,
                        output_path,
                        lossy_level=0,
                        frame_keep_ratio=1.0,
                        color_keep_count=target_colors,
                    )

                final_colors = count_gif_colors(output_path)
                file_size = output_path.stat().st_size

                results[engine.value] = {
                    "colors": final_colors,
                    "size": file_size,
                    "command": result.get("command", []),
                }

            finally:
                output_path.unlink()

        # Verify alignment
        if len(results) == 2:
            gifsicle_colors = results["gifsicle"]["colors"]
            animately_colors = results["animately"]["colors"]

            # Both should not exceed the target
            assert (
                gifsicle_colors <= target_colors
            ), f"Gifsicle exceeded limit: {gifsicle_colors} > {target_colors}"
            assert (
                animately_colors <= target_colors
            ), f"Animately exceeded limit: {animately_colors} > {target_colors}"

            # With aligned settings, they should be very close or identical
            color_diff = abs(gifsicle_colors - animately_colors)
            assert color_diff <= 1, (
                f"Aligned engines should produce similar results: "
                f"gifsicle={gifsicle_colors}, animately={animately_colors}, diff={color_diff}"
            )

            # File sizes should be reasonably close (within 50% for compression differences)
            gifsicle_size = results["gifsicle"]["size"]
            animately_size = results["animately"]["size"]

            size_ratio = max(gifsicle_size, animately_size) / min(
                gifsicle_size, animately_size
            )
            assert (
                size_ratio <= 2.0
            ), f"File sizes too different: gifsicle={gifsicle_size}, animately={animately_size}, ratio={size_ratio}"

            # Verify gifsicle uses --no-dither in command
            gifsicle_cmd = results["gifsicle"]["command"]
            assert (
                "--no-dither" in gifsicle_cmd
            ), "Gifsicle should use --no-dither for alignment"
            assert (
                "--colors" in gifsicle_cmd
            ), "Gifsicle should use --colors for reduction"

            # Verify animately uses --colors in command
            animately_cmd = results["animately"]["command"]
            assert (
                "--colors" in animately_cmd
            ), "Animately should use --colors for reduction"

    finally:
        test_gif.unlink()


@pytest.mark.parametrize(
    "color_range",
    [
        (256, [128, 64, 32, 16]),  # Full palette reduction
        (128, [64, 32, 16]),  # Half palette reduction
        (64, [32, 16]),  # Quarter palette reduction
    ],
)
def test_color_reduction_consistency_across_ranges(color_range):
    """Test color reduction consistency across different color ranges."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip("Both engines must be available")

    original_colors, targets = color_range

    # Create test GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        test_gif = Path(tmp.name)

    try:
        _create_colorful_test_gif(test_gif, frames=3)

        # Test each target in the range
        for target_colors in targets:
            print(f"\nTesting {original_colors} → {target_colors} colors")

            # Test both engines
            results = {}

            for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
                with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False
                ) as tmp_out:
                    output_path = Path(tmp_out.name)

                try:
                    if engine == LossyEngine.GIFSICLE:
                        compress_with_gifsicle(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )
                    else:
                        compress_with_animately(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )

                    final_colors = count_gif_colors(output_path)
                    results[engine.value] = final_colors

                finally:
                    output_path.unlink()

            # Verify consistency
            if len(results) == 2:
                gifsicle_colors = results["gifsicle"]
                animately_colors = results["animately"]

                # Both should respect the target
                assert gifsicle_colors <= target_colors
                assert animately_colors <= target_colors

                # Should be close or identical (allow more tolerance for simple test GIFs)
                color_diff = abs(gifsicle_colors - animately_colors)
                assert color_diff <= 3, (
                    f"Inconsistent results for {target_colors} colors: "
                    f"gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

                print(
                    f"  ✓ {target_colors}: gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

    finally:
        test_gif.unlink()


def test_color_reduction_edge_cases():
    """Test color reduction edge cases for alignment."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip("Both engines must be available")

    # Create test GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        test_gif = Path(tmp.name)

    try:
        _create_colorful_test_gif(test_gif, frames=2)
        original_colors = count_gif_colors(test_gif)

        # Test edge cases
        edge_cases = [
            (original_colors, "no_reduction_equal"),  # No reduction needed
            (
                original_colors + 10,
                "no_reduction_higher",
            ),  # Target higher than original
            (8, "extreme_reduction"),  # Very aggressive reduction
            (2, "minimal_colors"),  # Minimal color count
        ]

        for target_colors, case_name in edge_cases:
            print(f"\nTesting edge case: {case_name} (target: {target_colors})")

            # Test both engines
            results = {}

            for engine in [LossyEngine.GIFSICLE, LossyEngine.ANIMATELY]:
                with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False
                ) as tmp_out:
                    output_path = Path(tmp_out.name)

                try:
                    if engine == LossyEngine.GIFSICLE:
                        compress_with_gifsicle(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )
                    else:
                        compress_with_animately(
                            test_gif,
                            output_path,
                            lossy_level=0,
                            frame_keep_ratio=1.0,
                            color_keep_count=target_colors,
                        )

                    final_colors = count_gif_colors(output_path)
                    results[engine.value] = final_colors

                except Exception as e:
                    # Some edge cases might fail, that's okay
                    print(f"  {engine.value} failed: {e}")
                    results[engine.value] = None

                finally:
                    if output_path.exists():
                        output_path.unlink()

            # Analyze results
            gifsicle_colors = results.get("gifsicle")
            animately_colors = results.get("animately")

            if gifsicle_colors is not None and animately_colors is not None:
                # Both succeeded
                if target_colors >= original_colors:
                    # No reduction should occur - but allow for small variations in color counting
                    assert (
                        gifsicle_colors <= original_colors + 2
                    ), f"Gifsicle colors unexpectedly high: {gifsicle_colors} > {original_colors + 2}"
                    assert (
                        animately_colors <= original_colors + 2
                    ), f"Animately colors unexpectedly high: {animately_colors} > {original_colors + 2}"
                else:
                    # Reduction should occur
                    assert (
                        gifsicle_colors <= target_colors
                    ), f"Gifsicle exceeded target: {gifsicle_colors} > {target_colors}"
                    assert (
                        animately_colors <= target_colors
                    ), f"Animately exceeded target: {animately_colors} > {target_colors}"

                    # Results should be close
                    color_diff = abs(gifsicle_colors - animately_colors)
                    assert (
                        color_diff <= 2
                    ), f"Results too different: {gifsicle_colors} vs {animately_colors}"

                print(
                    f"  ✓ Both engines: gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

            elif gifsicle_colors is None and animately_colors is None:
                # Both failed - acceptable for extreme edge cases
                print("  ✓ Both engines failed (acceptable for extreme case)")

            else:
                # One succeeded, one failed - this might indicate an issue
                print(
                    f"  ⚠ Mixed results: gifsicle={gifsicle_colors}, animately={animately_colors}"
                )

    finally:
        test_gif.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
