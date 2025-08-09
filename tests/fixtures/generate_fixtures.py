#!/usr/bin/env python3
"""Generate minimal test GIF fixtures for engine testing."""

from pathlib import Path

import numpy as np
from PIL import Image


def create_simple_4frame_gif():
    """Create a 4-frame, 16-color, 64x64px GIF for basic functionality tests."""
    frames = []
    colors = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
    ]

    for i, color in enumerate(colors):
        # Create 64x64 image with solid color background
        img = Image.new("RGB", (64, 64), color)

        # Add a small moving square to show animation
        square_x = 10 + i * 10
        square_y = 10 + i * 5

        # Draw a 16x16 white square
        for x in range(square_x, min(square_x + 16, 64)):
            for y in range(square_y, min(square_y + 16, 64)):
                img.putpixel((x, y), (255, 255, 255))

        frames.append(img)

    # Save as GIF with palette reduction to ~16 colors
    output_path = Path(__file__).parent / "simple_4frame.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,  # 250ms per frame
        loop=0,
        optimize=False,  # Keep unoptimized for testing
    )
    print(f"Created: {output_path}")
    return output_path


def create_single_frame_gif():
    """Create a 1-frame, 8-color, 32x32px GIF for edge case testing."""
    # Create a simple gradient pattern
    img = Image.new("RGB", (32, 32))

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 128),
        (255, 255, 255),
    ]

    # Create a simple checkerboard pattern with 8 colors
    for x in range(32):
        for y in range(32):
            color_idx = ((x // 4) + (y // 4)) % len(colors)
            img.putpixel((x, y), colors[color_idx])

    output_path = Path(__file__).parent / "single_frame.gif"
    img.save(output_path, optimize=False)
    print(f"Created: {output_path}")
    return output_path


def create_many_colors_gif():
    """Create a 4-frame, 256-color, 64x64px GIF for palette stress testing."""
    frames = []

    for frame_idx in range(4):
        img = Image.new("RGB", (64, 64))

        # Generate a gradient with many colors
        for x in range(64):
            for y in range(64):
                # Create RGB values that use the full spectrum
                r = (x * 4 + frame_idx * 16) % 256
                g = (y * 4 + frame_idx * 32) % 256
                b = ((x + y) * 2 + frame_idx * 8) % 256
                img.putpixel((x, y), (r, g, b))

        frames.append(img)

    output_path = Path(__file__).parent / "many_colors.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=False,  # Keep all colors for testing
    )
    print(f"Created: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating test fixtures...")
    create_simple_4frame_gif()
    create_single_frame_gif()
    create_many_colors_gif()
    print("Done!")
