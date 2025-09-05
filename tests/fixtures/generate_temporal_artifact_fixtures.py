#!/usr/bin/env python3
"""Generate test GIF fixtures that exhibit specific temporal artifacts.

This module creates synthetic test GIFs that demonstrate different types of
temporal artifacts that the enhanced temporal artifact detection system should
be able to identify.
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image


def create_flicker_excess_gif(severity="high"):
    """Create GIF with excessive flicker between frames.

    Args:
        severity: 'high' for severe flicker, 'low' for minimal flicker

    Returns:
        Path to created GIF file
    """
    frames = []
    base_color = (100, 100, 100)

    for i in range(10):
        img = Image.new("RGB", (64, 64), base_color)

        if severity == "high":
            # High flicker - dramatic changes between frames
            if i % 2 == 0:
                # Add random bright patches that flicker on/off
                random.seed(42 + i)  # Reproducible randomness
                for _ in range(5):
                    x, y = random.randint(0, 48), random.randint(0, 48)
                    for dx in range(16):
                        for dy in range(16):
                            if x + dx < 64 and y + dy < 64:
                                img.putpixel((x + dx, y + dy), (255, 255, 255))
            else:
                # Add random dark patches
                random.seed(42 + i)
                for _ in range(5):
                    x, y = random.randint(0, 48), random.randint(0, 48)
                    for dx in range(16):
                        for dy in range(16):
                            if x + dx < 64 and y + dy < 64:
                                img.putpixel((x + dx, y + dy), (0, 0, 0))
        else:  # low flicker
            # Subtle consistent changes - smooth animation
            offset = i * 5
            for x in range(64):
                for y in range(64):
                    val = min(255, max(0, base_color[0] + offset % 20))
                    img.putpixel((x, y), (val, val, val))

        frames.append(img)

    output_path = Path(__file__).parent / f"flicker_{severity}.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=2,  # Restore to background
    )
    print(f"Created: {output_path}")
    return output_path


def create_background_flicker_gif(stable=False):
    """Create GIF with flickering or stable background regions.

    Args:
        stable: If True, create stable background; if False, flickering background

    Returns:
        Path to created GIF file
    """
    frames = []

    for i in range(10):
        img = Image.new("RGB", (64, 64))

        # Create background edges that should be stable
        if stable:
            edge_color = (50, 50, 50)  # Constant background
        else:
            # Flickering background - changes unpredictably
            edge_color = (50 + (i * 20) % 100, 50, 50)

        # Fill edges with background color
        for x in range(64):
            for y in range(10):  # Top edge
                img.putpixel((x, y), edge_color)
            for y in range(54, 64):  # Bottom edge
                img.putpixel((x, y), edge_color)
        for y in range(10, 54):
            for x in range(10):  # Left edge
                img.putpixel((x, y), edge_color)
            for x in range(54, 64):  # Right edge
                img.putpixel((x, y), edge_color)

        # Animated center content (this is expected to change)
        center_color = (200, 100 + i * 20 % 100, 100)
        for x in range(20, 44):
            for y in range(20, 44):
                img.putpixel((x, y), center_color)

        frames.append(img)

    output_path = (
        Path(__file__).parent / f"background_{'stable' if stable else 'flickering'}.gif"
    )
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Created: {output_path}")
    return output_path


def create_temporal_pumping_gif(pumping=True):
    """Create GIF with quality oscillation (pumping effect).

    Args:
        pumping: If True, create temporal pumping; if False, consistent quality

    Returns:
        Path to created GIF file
    """
    frames = []

    for i in range(12):
        img = Image.new("RGB", (64, 64))

        if pumping:
            # Oscillating quality - alternating between sharp and blurry/quantized
            quality_factor = 1.0 if i % 3 == 0 else 0.3

            # Create gradient pattern with varying quality
            for x in range(64):
                for y in range(64):
                    # Base gradient pattern
                    base = int(((x + y) / 128) * 255)

                    if quality_factor < 1.0:
                        # Simulate quality loss with heavy quantization
                        base = (base // 32) * 32

                    img.putpixel((x, y), (base, base, base))

            # Add fine details that disappear/reappear (pumping effect)
            if quality_factor == 1.0:
                # High quality frame - add fine details
                for j in range(0, 64, 4):
                    if j < 64:
                        img.putpixel((j, j), (255, 0, 0))
                        if j + 1 < 64:
                            img.putpixel((j + 1, j), (0, 255, 0))
        else:
            # Consistent quality throughout
            for x in range(64):
                for y in range(64):
                    base = int(((x + y) / 128) * 255)
                    img.putpixel((x, y), (base, base, base))

            # Consistent details
            for j in range(0, 64, 8):
                if j < 64:
                    img.putpixel((j, j), (255, 0, 0))

        frames.append(img)

    output_path = Path(__file__).parent / f"pumping_{'yes' if pumping else 'no'}.gif"
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Created: {output_path}")
    return output_path


def create_disposal_artifact_gif(corrupted=True):
    """Create GIF with disposal method artifacts.

    Args:
        corrupted: If True, create disposal artifacts; if False, clean disposal

    Returns:
        Path to created GIF file
    """
    frames = []
    accumulated_artifacts = None

    for i in range(8):
        # Base background
        img = Image.new("RGB", (64, 64), (100, 100, 100))

        # Draw moving object
        obj_x = i * 8
        obj_y = i * 4
        for x in range(obj_x, min(obj_x + 10, 64)):
            for y in range(obj_y, min(obj_y + 10, 64)):
                img.putpixel((x, y), (255, 0, 0))

        if corrupted and i > 0:
            # Simulate disposal artifacts - previous frames bleeding through
            if accumulated_artifacts is None:
                accumulated_artifacts = np.array(img)
            else:
                # Accumulate artifacts from previous frames
                current = np.array(img)
                # Blend 20% of accumulated artifacts with current frame
                accumulated_artifacts = (
                    0.2 * accumulated_artifacts + 0.8 * current
                ).astype(np.uint8)
                img = Image.fromarray(accumulated_artifacts)

                # Add some random corruption patterns
                if i % 2 == 0:
                    for _ in range(3):
                        x, y = random.randint(0, 54), random.randint(0, 54)
                        for dx in range(10):
                            for dy in range(10):
                                if x + dx < 64 and y + dy < 64:
                                    # Ghost pixels from previous frames
                                    img.putpixel((x + dx, y + dy), (128, 64, 64))

        frames.append(img)

    output_path = (
        Path(__file__).parent / f"disposal_{'corrupted' if corrupted else 'clean'}.gif"
    )
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=1 if corrupted else 2,  # Improper vs proper disposal method
    )
    print(f"Created: {output_path}")
    return output_path


def create_smooth_animation_gif():
    """Create a smooth, high-quality animation for comparison baseline.

    Returns:
        Path to created GIF file
    """
    frames = []

    for i in range(16):
        img = Image.new("RGB", (64, 64), (50, 50, 50))

        # Smooth circular motion
        angle = (i / 16.0) * 2 * math.pi
        center_x, center_y = 32, 32
        radius = 15

        obj_x = int(center_x + radius * math.cos(angle))
        obj_y = int(center_y + radius * math.sin(angle))

        # Draw smooth circular object
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x, y = obj_x + dx, obj_y + dy
                if 0 <= x < 64 and 0 <= y < 64:
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance <= 5:
                        # Smooth falloff
                        intensity = max(0, 1.0 - distance / 5.0)
                        color_val = int(200 * intensity)
                        img.putpixel((x, y), (color_val, color_val, 255))

        frames.append(img)

    output_path = Path(__file__).parent / "smooth_animation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        disposal=2,  # Proper disposal
    )
    print(f"Created: {output_path}")
    return output_path


def create_static_with_noise_gif():
    """Create a mostly static GIF with random noise (should have stable background).

    Returns:
        Path to created GIF file
    """
    frames = []
    base_pattern = np.random.RandomState(42).randint(
        0, 255, (64, 64, 3), dtype=np.uint8
    )

    for i in range(8):
        # Start with base pattern
        frame_array = base_pattern.copy()

        # Add small amount of random noise in center only
        noise_area = np.random.RandomState(42 + i).randint(
            -10, 11, (20, 20, 3), dtype=np.int16
        )
        frame_array[22:42, 22:42] = np.clip(
            frame_array[22:42, 22:42].astype(np.int16) + noise_area, 0, 255
        ).astype(np.uint8)

        img = Image.fromarray(frame_array)
        frames.append(img)

    output_path = Path(__file__).parent / "static_with_noise.gif"
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=150, loop=0
    )
    print(f"Created: {output_path}")
    return output_path


def main():
    """Generate all test fixtures."""
    print("Generating temporal artifact test fixtures...")

    # Ensure fixtures directory exists
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)

    # Generate all test fixtures
    fixtures = [
        create_flicker_excess_gif("high"),
        create_flicker_excess_gif("low"),
        create_background_flicker_gif(stable=True),
        create_background_flicker_gif(stable=False),
        create_temporal_pumping_gif(pumping=True),
        create_temporal_pumping_gif(pumping=False),
        create_disposal_artifact_gif(corrupted=True),
        create_disposal_artifact_gif(corrupted=False),
        create_smooth_animation_gif(),
        create_static_with_noise_gif(),
    ]

    print(f"\nGenerated {len(fixtures)} test fixtures:")
    for fixture_path in fixtures:
        print(f"  - {fixture_path.name}")

    print("\nTest fixtures ready for temporal artifact detection tests!")


if __name__ == "__main__":
    main()
