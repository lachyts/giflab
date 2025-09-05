#!/usr/bin/env python3
"""Generate test GIF fixtures that exhibit specific gradient and color artifacts.

This module creates synthetic test GIFs that demonstrate different types of
gradient banding and color artifacts that the enhanced gradient and color
validation system should be able to identify.
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


def create_smooth_gradient_gif(direction="horizontal", colors=None):
    """Create GIF with smooth gradient that should NOT trigger banding detection.

    Args:
        direction: 'horizontal', 'vertical', or 'radial'
        colors: Tuple of (start_color, end_color) or None for default

    Returns:
        Path to created GIF file
    """
    if colors is None:
        colors = ((50, 100, 200), (200, 150, 50))  # Blue to orange

    frames = []
    size = (128, 128)

    for frame_idx in range(8):
        img = Image.new("RGB", size)
        draw = ImageDraw.Draw(img)

        # Create smooth gradient
        for i in range(size[0] if direction == "horizontal" else size[1]):
            if direction == "horizontal":
                ratio = i / (size[0] - 1)
                color = _interpolate_color(colors[0], colors[1], ratio)
                draw.line([(i, 0), (i, size[1] - 1)], fill=color)
            elif direction == "vertical":
                ratio = i / (size[1] - 1)
                color = _interpolate_color(colors[0], colors[1], ratio)
                draw.line([(0, i), (size[0] - 1, i)], fill=color)
            elif direction == "radial":
                # Create radial gradient
                center = (size[0] // 2, size[1] // 2)
                max_distance = math.sqrt(center[0] ** 2 + center[1] ** 2)

                for x in range(size[0]):
                    for y in range(size[1]):
                        distance = math.sqrt(
                            (x - center[0]) ** 2 + (y - center[1]) ** 2
                        )
                        ratio = min(distance / max_distance, 1.0)
                        color = _interpolate_color(colors[0], colors[1], ratio)
                        draw.point((x, y), fill=color)

        # Slight animation to create multiple frames
        if frame_idx > 0:
            # Add very subtle variation to test temporal consistency
            overlay = Image.new("RGBA", size, (255, 255, 255, 5))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        frames.append(img)

    output_path = Path("test_fixtures") / f"smooth_gradient_{direction}.gif"
    output_path.parent.mkdir(exist_ok=True)

    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=200, loop=0
    )

    return output_path


def create_banded_gradient_gif(severity="high", direction="horizontal"):
    """Create GIF with posterized gradient that SHOULD trigger banding detection.

    Args:
        severity: 'high' for severe banding, 'medium' for moderate, 'low' for subtle
        direction: 'horizontal', 'vertical', or 'radial'

    Returns:
        Path to created GIF file
    """
    frames = []
    size = (128, 128)

    # Define number of bands based on severity
    if severity == "high":
        num_bands = 8  # Very visible banding
    elif severity == "medium":
        num_bands = 16  # Moderate banding
    else:  # low
        num_bands = 32  # Subtle banding

    colors = ((20, 50, 150), (220, 180, 50))  # Dark blue to yellow

    for _frame_idx in range(8):
        img = Image.new("RGB", size)
        draw = ImageDraw.Draw(img)

        if direction == "horizontal":
            band_width = size[0] // num_bands
            for band in range(num_bands):
                ratio = band / (num_bands - 1)
                color = _interpolate_color(colors[0], colors[1], ratio)
                x_start = band * band_width
                x_end = min((band + 1) * band_width, size[0])
                draw.rectangle([(x_start, 0), (x_end - 1, size[1] - 1)], fill=color)

        elif direction == "vertical":
            band_height = size[1] // num_bands
            for band in range(num_bands):
                ratio = band / (num_bands - 1)
                color = _interpolate_color(colors[0], colors[1], ratio)
                y_start = band * band_height
                y_end = min((band + 1) * band_height, size[1])
                draw.rectangle([(0, y_start), (size[0] - 1, y_end - 1)], fill=color)

        elif direction == "radial":
            center = (size[0] // 2, size[1] // 2)
            max_distance = math.sqrt(center[0] ** 2 + center[1] ** 2)
            band_distance = max_distance / num_bands

            for band in range(num_bands):
                ratio = band / (num_bands - 1)
                color = _interpolate_color(colors[0], colors[1], ratio)
                inner_radius = band * band_distance
                outer_radius = (band + 1) * band_distance

                # Draw ring
                for x in range(size[0]):
                    for y in range(size[1]):
                        distance = math.sqrt(
                            (x - center[0]) ** 2 + (y - center[1]) ** 2
                        )
                        if inner_radius <= distance < outer_radius:
                            draw.point((x, y), fill=color)

        frames.append(img)

    output_path = Path("test_fixtures") / f"banded_gradient_{severity}_{direction}.gif"
    output_path.parent.mkdir(exist_ok=True)

    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=200, loop=0
    )

    return output_path


def create_color_shift_gif(shift_severity="high", preserve_original=True):
    """Create GIF pair with known color shifts for ΔE00 testing.

    Args:
        shift_severity: 'high', 'medium', or 'low' color difference
        preserve_original: If True, also save original for comparison

    Returns:
        Tuple of (original_path, shifted_path)
    """
    frames_original = []
    frames_shifted = []
    size = (96, 96)

    # Define test color patches
    test_patches = [
        ((255, 0, 0), (10, 10, 32, 32)),  # Red patch
        ((0, 255, 0), (50, 10, 32, 32)),  # Green patch
        ((0, 0, 255), (10, 50, 32, 32)),  # Blue patch
        ((128, 128, 128), (50, 50, 32, 32)),  # Gray patch
    ]

    # Define color shift amounts based on severity
    if shift_severity == "high":
        # Large perceptual differences (ΔE00 > 5)
        color_shifts = [
            (50, -30, 20),  # Shift red patch
            (-40, 30, -20),  # Shift green patch
            (20, 20, -50),  # Shift blue patch
            (40, -40, 40),  # Shift gray patch
        ]
    elif shift_severity == "medium":
        # Moderate differences (ΔE00 2-5)
        color_shifts = [
            (20, -15, 10),
            (-15, 15, -10),
            (10, 10, -20),
            (15, -15, 15),
        ]
    else:  # low
        # Subtle differences (ΔE00 1-2)
        color_shifts = [
            (8, -5, 3),
            (-6, 6, -4),
            (4, 4, -8),
            (6, -6, 6),
        ]

    for frame_idx in range(6):
        # Create original frame
        img_orig = Image.new("RGB", size, (240, 240, 240))  # Light gray background
        draw_orig = ImageDraw.Draw(img_orig)

        # Create shifted frame
        img_shift = Image.new("RGB", size, (240, 240, 240))
        draw_shift = ImageDraw.Draw(img_shift)

        # Add test color patches
        for i, ((r, g, b), (x, y, w, h)) in enumerate(test_patches):
            # Original patch
            draw_orig.rectangle([(x, y), (x + w, y + h)], fill=(r, g, b))

            # Shifted patch
            shift_r, shift_g, shift_b = color_shifts[i]
            shifted_color = (
                max(0, min(255, r + shift_r)),
                max(0, min(255, g + shift_g)),
                max(0, min(255, b + shift_b)),
            )
            draw_shift.rectangle([(x, y), (x + w, y + h)], fill=shifted_color)

        # Add some texture/noise for more realistic testing
        random.seed(42 + frame_idx)
        for _ in range(20):
            x, y = random.randint(0, size[0] - 1), random.randint(0, size[1] - 1)
            noise_color = (
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255),
            )
            draw_orig.point((x, y), fill=noise_color)
            draw_shift.point((x, y), fill=noise_color)

        frames_original.append(img_orig)
        frames_shifted.append(img_shift)

    # Save GIFs
    output_dir = Path("test_fixtures")
    output_dir.mkdir(exist_ok=True)

    original_path = output_dir / f"color_original_{shift_severity}.gif"
    shifted_path = output_dir / f"color_shifted_{shift_severity}.gif"

    frames_original[0].save(
        original_path,
        save_all=True,
        append_images=frames_original[1:],
        duration=300,
        loop=0,
    )

    frames_shifted[0].save(
        shifted_path,
        save_all=True,
        append_images=frames_shifted[1:],
        duration=300,
        loop=0,
    )

    return original_path, shifted_path


def create_brand_color_test_gif(include_color_shifts=True):
    """Create GIF with typical brand colors for UI/brand color testing.

    Args:
        include_color_shifts: If True, create shifted version for comparison

    Returns:
        Path to created GIF file (or tuple if include_color_shifts=True)
    """
    frames_original = []
    frames_shifted = [] if include_color_shifts else None
    size = (120, 80)

    # Define typical brand colors
    brand_colors = [
        (0, 123, 255),  # Bootstrap blue
        (40, 167, 69),  # Bootstrap green
        (220, 53, 69),  # Bootstrap red
        (255, 193, 7),  # Bootstrap yellow
        (108, 117, 125),  # Bootstrap gray
    ]

    for _frame_idx in range(5):
        img_orig = Image.new("RGB", size, (255, 255, 255))
        draw_orig = ImageDraw.Draw(img_orig)

        if include_color_shifts:
            img_shift = Image.new("RGB", size, (255, 255, 255))
            draw_shift = ImageDraw.Draw(img_shift)

        # Create color stripes
        stripe_height = size[1] // len(brand_colors)

        for i, color in enumerate(brand_colors):
            y_start = i * stripe_height
            y_end = min((i + 1) * stripe_height, size[1])

            # Original color
            draw_orig.rectangle([(0, y_start), (size[0] - 1, y_end - 1)], fill=color)

            if include_color_shifts:
                # Shift that would be problematic for brand colors (ΔE00 > 3)
                shifted_color = (
                    max(0, min(255, color[0] + 15)),
                    max(0, min(255, color[1] - 10)),
                    max(0, min(255, color[2] + 20)),
                )
                draw_shift.rectangle(
                    [(0, y_start), (size[0] - 1, y_end - 1)], fill=shifted_color
                )

        frames_original.append(img_orig)
        if include_color_shifts:
            frames_shifted.append(img_shift)

    # Save GIFs
    output_dir = Path("test_fixtures")
    output_dir.mkdir(exist_ok=True)

    original_path = output_dir / "brand_colors_original.gif"

    frames_original[0].save(
        original_path,
        save_all=True,
        append_images=frames_original[1:],
        duration=400,
        loop=0,
    )

    if include_color_shifts:
        shifted_path = output_dir / "brand_colors_shifted.gif"
        frames_shifted[0].save(
            shifted_path,
            save_all=True,
            append_images=frames_shifted[1:],
            duration=400,
            loop=0,
        )
        return original_path, shifted_path

    return original_path


def _interpolate_color(
    color1: tuple[int, int, int], color2: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    """Interpolate between two RGB colors.

    Args:
        color1: Start color (R, G, B)
        color2: End color (R, G, B)
        ratio: Interpolation ratio (0.0 = color1, 1.0 = color2)

    Returns:
        Interpolated color (R, G, B)
    """
    ratio = max(0.0, min(1.0, ratio))  # Clamp ratio

    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)

    return (r, g, b)


def main():
    """Generate all test fixtures for gradient and color artifact validation."""
    print("Generating gradient and color artifact test fixtures...")

    # Create output directory
    output_dir = Path("test_fixtures")
    output_dir.mkdir(exist_ok=True)

    # Generate gradient fixtures
    print("Creating smooth gradients (should NOT trigger banding detection)...")
    create_smooth_gradient_gif("horizontal")
    create_smooth_gradient_gif("vertical")
    create_smooth_gradient_gif("radial")

    print("Creating banded gradients (SHOULD trigger banding detection)...")
    create_banded_gradient_gif("high", "horizontal")
    create_banded_gradient_gif("medium", "horizontal")
    create_banded_gradient_gif("low", "horizontal")
    create_banded_gradient_gif("high", "vertical")
    create_banded_gradient_gif("high", "radial")

    # Generate color shift fixtures
    print("Creating color shift test pairs...")
    create_color_shift_gif("high")
    create_color_shift_gif("medium")
    create_color_shift_gif("low")

    # Generate brand color fixtures
    print("Creating brand color test fixtures...")
    create_brand_color_test_gif(include_color_shifts=True)

    print(f"All fixtures saved to: {output_dir.absolute()}")
    print("Fixtures created:")
    for gif_file in sorted(output_dir.glob("*.gif")):
        print(f"  - {gif_file.name}")


if __name__ == "__main__":
    main()
