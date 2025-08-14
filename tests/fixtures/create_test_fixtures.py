"""Create test fixtures with known properties for validation testing.

This script creates GIFs with precisely known characteristics (frame counts,
color counts, timing) that can be used to validate wrapper behavior.
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_test_gifs():
    """Create test GIF fixtures with known properties."""
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)
    
    # Test GIF 1: 10 frames, simple animation, known timing
    create_10_frame_gif(fixtures_dir / "test_10_frames.gif")
    
    # Test GIF 2: 4 frames for quick testing
    create_4_frame_gif(fixtures_dir / "test_4_frames.gif")
    
    # Test GIF 3: Many colors for color reduction testing
    create_many_colors_gif(fixtures_dir / "test_256_colors.gif")
    
    # Test GIF 4: Simple 2-color GIF
    create_simple_2_color_gif(fixtures_dir / "test_2_colors.gif")
    
    # Test GIF 5: Large frame count for frame reduction testing
    create_large_frame_count_gif(fixtures_dir / "test_30_frames.gif")
    
    print("✅ Test fixtures created successfully")


def create_10_frame_gif(output_path: Path):
    """Create 10-frame test GIF with known properties."""
    frames = []
    
    for i in range(10):
        # Create 100x100 frame
        img = Image.new('RGB', (100, 100), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw a colored rectangle that moves across frames
        x = i * 10
        color = (255, 0, 0) if i < 5 else (0, 255, 0)
        draw.rectangle([x, 40, x+20, 60], fill=color)
        
        # Add frame number text
        draw.text((10, 10), f"Frame {i+1}", fill='black')
        
        frames.append(img)
    
    # Save with 100ms delays (10 FPS)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    
    print(f"Created 10-frame GIF: {output_path}")


def create_4_frame_gif(output_path: Path):
    """Create 4-frame test GIF for quick testing."""
    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i in range(4):
        img = Image.new('RGB', (80, 80), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw colored circle
        draw.ellipse([20, 20, 60, 60], fill=colors[i])
        draw.text((5, 5), f"F{i+1}", fill='black')
        
        frames.append(img)
    
    # Save with 150ms delays
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    
    print(f"Created 4-frame GIF: {output_path}")


def create_many_colors_gif(output_path: Path):
    """Create GIF with many colors for color reduction testing."""
    frames = []
    
    for frame_idx in range(5):
        # Create gradient image with many colors
        img = Image.new('RGB', (128, 128))
        pixels = []
        
        for y in range(128):
            for x in range(128):
                # Create gradient with frame-dependent offset
                r = int((x + frame_idx * 25) % 256)
                g = int((y + frame_idx * 25) % 256)
                b = int((x + y + frame_idx * 25) % 256)
                pixels.append((r, g, b))
        
        img.putdata(pixels)
        
        # Add text overlay
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"Colors F{frame_idx+1}", fill='white')
        
        frames.append(img)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    
    print(f"Created many-colors GIF: {output_path}")


def create_simple_2_color_gif(output_path: Path):
    """Create simple 2-color GIF."""
    frames = []
    
    for i in range(6):
        img = Image.new('P', (60, 60))  # Palette mode
        img.putpalette([
            0, 0, 0,      # Black
            255, 255, 255 # White
        ] + [0] * (256-2)*3)  # Fill rest with zeros
        
        draw = ImageDraw.Draw(img)
        
        # Alternate pattern
        if i % 2 == 0:
            draw.rectangle([10, 10, 50, 50], fill=1)  # White square
        else:
            draw.ellipse([15, 15, 45, 45], fill=1)    # White circle
        
        frames.append(img.convert('RGB'))  # Convert for saving
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0
    )
    
    print(f"Created 2-color GIF: {output_path}")


def create_large_frame_count_gif(output_path: Path):
    """Create GIF with 30 frames for frame reduction testing."""
    frames = []
    
    for i in range(30):
        img = Image.new('RGB', (64, 64), 'lightgray')
        draw = ImageDraw.Draw(img)
        
        # Animated progress bar
        progress = i / 29.0
        bar_width = int(50 * progress)
        
        # Background bar
        draw.rectangle([7, 25, 57, 35], outline='black', fill='white')
        # Progress bar
        if bar_width > 0:
            draw.rectangle([8, 26, 8+bar_width-1, 34], fill='green')
        
        # Frame counter
        draw.text((2, 2), f"{i+1}/30", fill='black')
        
        frames.append(img)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,  # Faster animation
        loop=0
    )
    
    print(f"Created 30-frame GIF: {output_path}")


def analyze_gif_properties(gif_path: Path):
    """Analyze and report GIF properties for validation."""
    try:
        with Image.open(gif_path) as img:
            print(f"\nAnalyzing {gif_path.name}:")
            print(f"  Format: {img.format}")
            print(f"  Size: {img.size}")
            print(f"  Frames: {getattr(img, 'n_frames', 1)}")
            print(f"  Mode: {img.mode}")
            
            if hasattr(img, 'info') and 'duration' in img.info:
                print(f"  Duration: {img.info['duration']}ms")
            
            # Count unique colors in first frame
            img.seek(0)
            if hasattr(img, 'getcolors'):
                colors = img.getcolors(maxcolors=256*256*256)
                if colors:
                    print(f"  Unique colors (frame 1): {len(colors)}")
                    
    except Exception as e:
        print(f"  Error analyzing {gif_path}: {e}")


if __name__ == "__main__":
    create_test_gifs()
    
    # Analyze created fixtures
    fixtures_dir = Path(__file__).parent
    for gif_file in fixtures_dir.glob("test_*.gif"):
        analyze_gif_properties(gif_file)
