"""Synthetic GIF generation for pipeline elimination testing.

This module provides comprehensive synthetic GIF generation capabilities
for testing various compression pipeline combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class SyntheticGifSpec:
    """Specification for a synthetic test GIF."""
    name: str
    frames: int
    size: Tuple[int, int]
    content_type: str
    description: str


class SyntheticGifGenerator:
    """Generator for synthetic test GIFs with diverse characteristics.
    
    Frame Count Optimization Strategy:
    The synthetic GIF specifications have been optimized to balance comprehensive
    testing coverage with reasonable pipeline elimination execution times:
    
    - **Standard GIFs (18 @ 8 frames)**: Core testing patterns optimized for speed
    - **Extended GIFs (4 @ 15-25 frames)**: Temporal processing and palette tests  
    - **Long Animation GIFs (2 @ 40-60 frames)**: Stress testing without being prohibitive
    - **Minimal GIF (1 @ 2 frames)**: Edge case testing for gifski and others
    
    Previous versions had frame counts up to 100, which caused pipeline elimination
    runs to take 6+ hours. Current optimization reduces typical runs to 2-4 hours
    while maintaining comprehensive coverage of all compression scenarios.
    
    Total estimated test load: ~25 GIFs × 8.4 avg frames × ~300 pipelines = ~63K tests
    (vs previous ~90K tests with unoptimized frame counts)
    """
    
    def __init__(self, output_dir: Path):
        """Initialize the synthetic GIF generator.
        
        Args:
            output_dir: Directory to save generated GIFs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define comprehensive synthetic test cases based on research findings
        self.synthetic_specs = self._get_default_specs()
    
    def _get_default_specs(self) -> List[SyntheticGifSpec]:
        """Get the default set of synthetic GIF specifications.
        
        Frame Count Strategy:
        - **Standard testing (8 frames)**: Most synthetic GIFs use 8 frames as a balance
          between adequate animation testing and reasonable execution time
        - **Minimal frames (2 frames)**: Tests edge case behavior with minimal animation
        - **Extended frames (20-30 frames)**: Tests temporal processing and frame reduction
        - **Long animation (40 frames)**: Stress tests frame processing efficiency without
          being prohibitively slow (reduced from 50 to balance testing vs performance)
        
        This distribution ensures comprehensive coverage while maintaining reasonable
        pipeline elimination execution times.
        """
        return [
            # ORIGINAL RESEARCH-BASED CONTENT TYPES
            # From research: gradients benefit from dithering
            SyntheticGifSpec(
                "smooth_gradient", 8, (120, 120), "gradient",
                "Smooth color transitions - should benefit from Riemersma dithering"
            ),
            SyntheticGifSpec(
                "complex_gradient", 8, (150, 150), "complex_gradient", 
                "Multi-directional gradients with multiple hues"
            ),
            
            # From research: solid colors should NOT use dithering
            SyntheticGifSpec(
                "solid_blocks", 8, (100, 100), "solid",
                "Flat color blocks - dithering should provide no benefit"
            ),
            SyntheticGifSpec(
                "high_contrast", 8, (120, 120), "contrast",
                "Sharp edges and high contrast - no dithering benefit expected"
            ),
            
            # From research: complex/noise content where Bayer scales 4-5 excel
            SyntheticGifSpec(
                "photographic_noise", 8, (140, 140), "noise",
                "Photo-realistic with noise - good for testing Bayer dithering"
            ),
            SyntheticGifSpec(
                "texture_complex", 8, (130, 130), "texture",
                "Complex textures where dithering patterns can blend naturally"
            ),
            
            # Geometric patterns from research
            SyntheticGifSpec(
                "geometric_patterns", 8, (110, 110), "geometric",
                "Structured geometric shapes - test ordered dithering methods"
            ),
            
            # Edge cases for comprehensive testing
            SyntheticGifSpec(
                "few_colors", 8, (100, 100), "minimal",
                "Very few distinct colors - test edge behavior"
            ),
            SyntheticGifSpec(
                "many_colors", 20, (160, 160), "spectrum",
                "Full color spectrum over extended animation - stress test palette reduction"
            ),
            SyntheticGifSpec(
                "animation_heavy", 25, (100, 100), "motion",
                "Rapid animation with temporal coherence requirements (optimized from 30 frames)"
            ),
            
            # SIZE VARIATIONS - Test if dimensions affect pipeline performance
            SyntheticGifSpec(
                "gradient_small", 8, (50, 50), "gradient",
                "Small gradient - test compression behavior at minimum realistic size"
            ),
            SyntheticGifSpec(
                "gradient_medium", 8, (200, 200), "gradient", 
                "Medium gradient - standard web size testing"
            ),
            SyntheticGifSpec(
                "gradient_large", 8, (500, 500), "gradient",
                "Large gradient - test performance on bigger files"
            ),
            SyntheticGifSpec(
                "gradient_xlarge", 8, (1000, 1000), "gradient",
                "Extra large gradient - maximum realistic size testing"
            ),
            SyntheticGifSpec(
                "noise_small", 8, (50, 50), "noise",
                "Small noisy content - test Bayer dithering on small dimensions"
            ),
            SyntheticGifSpec(
                "noise_large", 8, (500, 500), "noise",
                "Large noisy content - test Bayer scale performance on large files"
            ),
            
            # FRAME COUNT VARIATIONS - Test temporal processing differences
            SyntheticGifSpec(
                "minimal_frames", 2, (120, 120), "gradient",
                "Minimal animation - test behavior with very few frames (intentionally 2 frames)"
            ),
            SyntheticGifSpec(
                "long_animation", 40, (120, 120), "motion",
                "Long animation - test frame processing efficiency (optimized from 50 frames)"
            ),
            SyntheticGifSpec(
                "extended_animation", 60, (120, 120), "motion",
                "Extended animation - stress test temporal optimization (reduced from 100 frames)"
            ),
            
            # MISSING CONTENT TYPES - Real-world patterns not covered
            SyntheticGifSpec(
                "mixed_content", 8, (200, 150), "mixed",
                "Text + graphics + photo elements - common real-world combination"
            ),
            SyntheticGifSpec(
                "data_visualization", 8, (300, 200), "charts",
                "Charts and graphs - technical/scientific content"
            ),
            SyntheticGifSpec(
                "transitions", 8, (150, 150), "morph",
                "Complex transitions and morphing - advanced animation patterns"
            ),
            
            # EDGE CASES - Extreme but realistic scenarios
            SyntheticGifSpec(
                "single_pixel_anim", 8, (100, 100), "micro_detail",
                "Single pixel changes - minimal motion detection"
            ),
            SyntheticGifSpec(
                "static_minimal_change", 15, (150, 150), "static_plus",
                "Mostly static with tiny changes - frame reduction opportunities (optimized from 20 frames)"
            ),
            SyntheticGifSpec(
                "high_frequency_detail", 8, (200, 200), "detail",
                "High frequency details - test aliasing and quality preservation"
            )
        ]
    
    def get_targeted_specs(self) -> List[SyntheticGifSpec]:
        """Get a strategically reduced set of synthetic GIF specs for targeted testing."""
        # Define high-value subset: Original + Size variations + 1 frame variation + 1 content type
        targeted_names = [
            # Keep all original research-based content (10 GIFs)
            'smooth_gradient', 'complex_gradient', 'solid_blocks', 'high_contrast',
            'photographic_noise', 'texture_complex', 'geometric_patterns', 
            'few_colors', 'many_colors', 'animation_heavy',
            
            # Add high-value size variations (4 GIFs)
            'gradient_small',    # 50x50 - minimum realistic
            'gradient_large',    # 500x500 - big file performance  
            'gradient_xlarge',   # 1000x1000 - maximum realistic
            'noise_large',       # 500x500 - test Bayer on large files
            
            # Add key frame variation (2 GIFs)
            'minimal_frames',    # 2 frames - edge case
            'long_animation',    # 50 frames - extended animation
            
            # Add most valuable new content type (1 GIF)
            'mixed_content'      # Real-world mixed content
        ]
        
        return [spec for spec in self.synthetic_specs if spec.name in targeted_names]
    
    def generate_gifs(self, use_targeted_set: bool = False) -> List[Path]:
        """Generate synthetic test GIFs.
        
        Args:
            use_targeted_set: If True, generate only the targeted subset
            
        Returns:
            List of paths to generated GIF files
        """
        specs = self.get_targeted_specs() if use_targeted_set else self.synthetic_specs
        
        gif_paths = []
        for spec in specs:
            gif_path = self.output_dir / f"{spec.name}.gif"
            if not gif_path.exists():
                self._create_synthetic_gif(gif_path, spec)
            gif_paths.append(gif_path)
            
        return gif_paths
    
    def _create_synthetic_gif(self, path: Path, spec: SyntheticGifSpec):
        """Create a synthetic GIF based on specification."""
        frame_generator = SyntheticFrameGenerator()
        images = []
        
        for frame_idx in range(spec.frames):
            img = frame_generator.create_frame(spec.content_type, spec.size, frame_idx, spec.frames)
            images.append(img)
        
        # Save GIF with consistent settings
        if images:
            images[0].save(
                path,
                save_all=True,
                append_images=images[1:],
                duration=100,  # 100ms per frame
                loop=0
            )


class SyntheticFrameGenerator:
    """Generator for individual synthetic frames based on content type."""
    
    def create_frame(self, content_type: str, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Create a frame based on content type.
        
        Args:
            content_type: Type of content to generate
            size: Frame dimensions (width, height)
            frame: Current frame index
            total_frames: Total number of frames
            
        Returns:
            Generated PIL Image
        """
        # Map content types to generation methods
        generators = {
            "gradient": self._create_gradient_frame,
            "complex_gradient": self._create_complex_gradient_frame,
            "solid": self._create_solid_frame,
            "contrast": self._create_contrast_frame,
            "noise": self._create_noise_frame,
            "texture": self._create_texture_frame,
            "geometric": self._create_geometric_frame,
            "minimal": self._create_minimal_frame,
            "spectrum": self._create_spectrum_frame,
            "motion": self._create_motion_frame,
            "mixed": self._create_mixed_content_frame,
            "charts": self._create_data_visualization_frame,
            "morph": self._create_transitions_frame,
            "micro_detail": self._create_single_pixel_anim_frame,
            "static_plus": self._create_static_minimal_change_frame,
            "detail": self._create_high_frequency_detail_frame,
        }
        
        generator_func = generators.get(content_type, self._create_simple_frame)
        return generator_func(size, frame, total_frames)
    
    def _create_gradient_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Create smooth gradient frame - should benefit from Riemersma dithering."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Animated gradient shift
        offset = (frame / total_frames) * 255
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Create smooth diagonal gradient with temporal shift
                r = int((x / size[0]) * 255)
                g = int((y / size[1]) * 255) 
                b = int(((x + y + offset) / (size[0] + size[1])) * 255) % 255
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_complex_gradient_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Multi-directional gradients with multiple hues."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Time-based rotation
        phase = (frame / total_frames) * 2 * np.pi
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Distance and angle from center
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx) + phase
                
                # Complex gradient based on polar coordinates
                r = int(127 + 127 * np.sin(angle))
                g = int(127 + 127 * np.cos(angle * 1.5))
                b = int(127 + 127 * np.sin(distance / 20 + phase))
                
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_solid_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Solid color blocks - dithering should provide no benefit."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Simple color blocks that change over time
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        block_size = 20
        
        for x in range(0, size[0], block_size):
            for y in range(0, size[1], block_size):
                color_idx = ((x // block_size) + (y // block_size) + frame) % len(colors)
                draw.rectangle([x, y, x + block_size - 1, y + block_size - 1], 
                             fill=colors[color_idx])
                
        return img
    
    def _create_contrast_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """High contrast patterns - no dithering benefit expected."""
        img = Image.new('RGB', size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Moving high contrast pattern
        offset = int((frame / total_frames) * 20)
        
        for x in range(0, size[0], 10):
            for y in range(0, size[1], 10):
                if ((x + y + offset) // 10) % 2 == 0:
                    draw.rectangle([x, y, x + 9, y + 9], fill=(255, 255, 255))
                    
        return img
    
    def _create_noise_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Photo-realistic noise - good for testing Bayer dithering."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Reproducible noise with temporal coherence
        np.random.seed(frame * 42)
        
        # Base image with noise
        base_color = int(128 + 50 * np.sin(frame / total_frames * 2 * np.pi))
        
        for x in range(size[0]):
            for y in range(size[1]):
                noise_r = np.random.randint(-50, 50)
                noise_g = np.random.randint(-50, 50) 
                noise_b = np.random.randint(-50, 50)
                
                r = max(0, min(255, base_color + noise_r))
                g = max(0, min(255, base_color + noise_g))
                b = max(0, min(255, base_color + noise_b))
                
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_texture_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Complex textures where dithering patterns blend naturally."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Procedural texture with animation
        phase = (frame / total_frames) * 4 * np.pi
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Multi-frequency texture pattern
                val1 = np.sin(x * 0.1 + phase) * np.cos(y * 0.1)
                val2 = np.sin(x * 0.05 + y * 0.05 + phase * 0.5)
                val3 = np.sin((x + y) * 0.02 + phase * 0.3)
                
                r = int(127 + 60 * val1)
                g = int(127 + 60 * val2) 
                b = int(127 + 60 * val3)
                
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_geometric_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Structured geometric shapes - test ordered dithering."""
        img = Image.new('RGB', size, (50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        # Animated geometric patterns
        rotation = (frame / total_frames) * 360
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Draw rotating polygons
        for radius in range(20, min(size) // 2, 15):
            vertices = []
            for i in range(6):  # Hexagon
                angle = rotation + i * 60
                x = center_x + radius * np.cos(np.radians(angle))
                y = center_y + radius * np.sin(np.radians(angle))
                vertices.append((x, y))
            
            color_intensity = int(100 + 100 * (radius / (min(size) // 2)))
            draw.polygon(vertices, fill=(color_intensity, color_intensity // 2, 255 - color_intensity))
            
        return img
    
    def _create_minimal_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Very few colors - test edge behavior."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        current_color = colors[frame % len(colors)]
        return Image.new('RGB', size, current_color)
    
    def _create_spectrum_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Full color spectrum - stress test palette reduction."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # HSV color wheel with animation
        phase = (frame / total_frames) * 360
        
        center_x, center_y = size[0] // 2, size[1] // 2
        max_radius = min(size) // 2
        
        for x in range(size[0]):
            for y in range(size[1]):
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.degrees(np.arctan2(dy, dx)) + phase
                
                if distance <= max_radius:
                    # HSV to RGB conversion
                    hue = angle % 360
                    saturation = distance / max_radius
                    value = 1.0
                    
                    h_i = int(hue / 60) % 6
                    f = hue / 60 - h_i
                    p = value * (1 - saturation)
                    q = value * (1 - f * saturation)
                    t = value * (1 - (1 - f) * saturation)
                    
                    if h_i == 0: r, g, b = value, t, p
                    elif h_i == 1: r, g, b = q, value, p
                    elif h_i == 2: r, g, b = p, value, t
                    elif h_i == 3: r, g, b = p, q, value
                    elif h_i == 4: r, g, b = t, p, value
                    else: r, g, b = value, p, q
                    
                    draw.point((x, y), (int(r*255), int(g*255), int(b*255)))
                    
        return img
    
    def _create_motion_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Rapid animation with temporal coherence requirements."""
        img = Image.new('RGB', size, (0, 0, 50))
        draw = ImageDraw.Draw(img)
        
        # Moving objects with trails
        progress = frame / total_frames
        
        # Primary moving object
        obj_x = int(progress * (size[0] - 20))
        obj_y = int(size[1] // 2 + 30 * np.sin(progress * 6 * np.pi))
        
        draw.ellipse([obj_x, obj_y, obj_x + 20, obj_y + 20], fill=(255, 200, 100))
        
        # Secondary bouncing object
        bounce_x = int(size[0] // 2 + (size[0] // 3) * np.sin(progress * 4 * np.pi))
        bounce_y = int(abs(np.sin(progress * 8 * np.pi)) * (size[1] - 30))
        
        draw.rectangle([bounce_x, bounce_y, bounce_x + 15, bounce_y + 15], fill=(100, 255, 150))
        
        return img
    
    def _create_mixed_content_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Mixed content: text + graphics + photo elements."""
        img = Image.new('RGB', size, (240, 240, 245))
        draw = ImageDraw.Draw(img)
        
        # Photo-like gradient region (top third)
        gradient_height = size[1] // 3
        for y in range(gradient_height):
            intensity = int(180 + 50 * (y / gradient_height))
            color_shift = int(20 * np.sin(frame / total_frames * 2 * np.pi))
            color = (intensity + color_shift, intensity - color_shift//2, intensity + color_shift//3)
            draw.line([(0, y), (size[0], y)], fill=color)
        
        # Text-like blocks (middle third)  
        text_y_start = gradient_height
        text_y_end = 2 * gradient_height
        block_width = size[0] // 8
        for i in range(8):
            x = i * block_width
            block_height = 5 + (frame + i) % 8
            y = text_y_start + 20 + i * 3
            if y + block_height < text_y_end:
                draw.rectangle([x, y, x + block_width - 2, y + block_height], fill=(50, 50, 50))
        
        # Graphics elements (bottom third)
        graphics_y_start = text_y_end
        circle_x = int((frame / total_frames) * (size[0] - 40) + 20)
        circle_y = graphics_y_start + 20
        draw.ellipse([circle_x - 15, circle_y - 15, circle_x + 15, circle_y + 15], 
                    fill=(255, 100, 100))
        
        return img
    
    def _create_data_visualization_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Charts and graphs - technical/scientific content."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw axes and animated data
        margin = 30
        chart_width = size[0] - 2 * margin
        chart_height = size[1] - 2 * margin
        
        # X and Y axes
        draw.line([(margin, size[1] - margin), (size[0] - margin, size[1] - margin)], 
                 fill=(0, 0, 0), width=2)
        draw.line([(margin, margin), (margin, size[1] - margin)], 
                 fill=(0, 0, 0), width=2)
        
        # Animated data points
        num_points = 10
        for i in range(num_points):
            x = margin + (i / (num_points - 1)) * chart_width
            base_height = 0.5 + 0.3 * np.sin(i / 2 + frame / total_frames * 2 * np.pi)
            y = size[1] - margin - base_height * chart_height
            
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(200, 50, 50))
            
            if i > 0:
                prev_x = margin + ((i-1) / (num_points - 1)) * chart_width
                prev_base = 0.5 + 0.3 * np.sin((i-1) / 2 + frame / total_frames * 2 * np.pi)
                prev_y = size[1] - margin - prev_base * chart_height
                draw.line([(prev_x, prev_y), (x, y)], fill=(100, 150, 200), width=2)
        
        return img
    
    def _create_transitions_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Complex transitions and morphing - advanced animation patterns."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        progress = frame / total_frames
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = min(size) // 4
        
        # Morphing between circle and square
        if progress < 0.5:
            morph_progress = progress * 2
            points = []
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                circle_x = center_x + radius * np.cos(rad)
                circle_y = center_y + radius * np.sin(rad)
                
                # Approximate square position
                if -45 <= angle < 45 or 315 <= angle < 360:
                    square_x = center_x + radius
                    square_y = center_y + radius * np.tan(rad)
                elif 45 <= angle < 135:
                    square_y = center_y - radius
                    square_x = center_x - radius * np.tan(rad - np.pi/2)
                elif 135 <= angle < 225:
                    square_x = center_x - radius
                    square_y = center_y - radius * np.tan(rad - np.pi)
                else:
                    square_y = center_y + radius
                    square_x = center_x + radius * np.tan(rad - 3*np.pi/2)
                
                x = circle_x + morph_progress * (square_x - circle_x)
                y = circle_y + morph_progress * (square_y - circle_y)
                points.append((x, y))
            
            if len(points) > 2:
                color_r = int(255 * (1 - morph_progress))
                color_g = int(255 * morph_progress)
                draw.polygon(points, fill=(color_r, color_g, 100))
        else:
            # Square to triangle transition
            morph_progress = (progress - 0.5) * 2
            
            square_points = [
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y - radius),
                (center_x + radius, center_y + radius),
                (center_x - radius, center_y + radius)
            ]
            
            triangle_points = [
                (center_x, center_y - radius),
                (center_x + radius, center_y + radius),
                (center_x - radius, center_y + radius)
            ]
            
            final_points = []
            for i in range(3):
                sq_x, sq_y = square_points[i]
                tri_x, tri_y = triangle_points[i]
                x = sq_x + morph_progress * (tri_x - sq_x)
                y = sq_y + morph_progress * (tri_y - sq_y)
                final_points.append((x, y))
            
            color_g = int(255 * (1 - morph_progress))
            color_b = int(255 * morph_progress)
            draw.polygon(final_points, fill=(100, color_g, color_b))
        
        return img
    
    def _create_single_pixel_anim_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Single pixel changes - minimal motion detection."""
        img = Image.new('RGB', size, (128, 128, 128))
        
        # Only change a few pixels each frame
        pixels_to_change = [(
            (frame * 7 + i * 13) % size[0],
            (frame * 5 + i * 11) % size[1]
        ) for i in range(3)]
        
        for x, y in pixels_to_change:
            color_shift = (frame * 17 + x + y) % 64
            img.putpixel((x, y), (128 + color_shift, 128 - color_shift//2, 128 + color_shift//3))
        
        # Add one more obvious but tiny moving element
        moving_x = (frame * 2) % size[0]
        moving_y = (frame) % size[1]
        img.putpixel((moving_x, moving_y), (255, 255, 255))
        
        return img
    
    def _create_static_minimal_change_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Mostly static with tiny changes - frame reduction opportunities."""
        img = Image.new('RGB', size, (200, 210, 220))
        draw = ImageDraw.Draw(img)
        
        # Static background pattern
        for x in range(0, size[0], 20):
            for y in range(0, size[1], 20):
                draw.rectangle([x, y, x + 18, y + 18], outline=(180, 190, 200))
        
        # Very minimal animated element
        if frame % 5 == 0 and size[0] > 10:
            change_x = (frame // 5) % (size[0] - 10)
            change_y = size[1] // 2
            draw.ellipse([change_x, change_y, change_x + 8, change_y + 8], 
                        fill=(220, 100, 100))
        
        if frame % 8 < 2 and size[0] > 20 and size[1] > 20:
            blink_x = size[0] - 20
            blink_y = 20
            draw.ellipse([blink_x, blink_y, blink_x + 4, blink_y + 4], 
                        fill=(100, 220, 100))
        
        return img
    
    def _create_high_frequency_detail_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """High frequency details - test aliasing and quality preservation."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Fine grid pattern
        for x in range(0, size[0], 2):
            for y in range(0, size[1], 2):
                if (x + y + frame) % 4 == 0:
                    draw.point((x, y), (0, 0, 0))
        
        # Moiré patterns - high frequency interference
        center_x, center_y = size[0] // 2, size[1] // 2
        for x in range(size[0]):
            for y in range(size[1]):
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                freq = 0.5 + frame / total_frames * 0.3
                if int(distance * freq) % 2 == 0:
                    img.putpixel((x, y), (200, 200, 200))
        
        # Fine diagonal lines that create aliasing
        line_spacing = 3
        offset = frame % (line_spacing * 2)
        for i in range(-size[0], size[0], line_spacing):
            x1, y1 = i + offset, 0
            x2, y2 = i + offset + size[1], size[1]
            if 0 <= x1 < size[0] and 0 <= x2 < size[0]:
                draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=1)
        
        return img
    
    def _create_simple_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Fallback simple frame."""
        img = Image.new('RGB', size, (100, 100, 100))
        draw = ImageDraw.Draw(img)
        draw.rectangle([frame * 5, frame * 5, frame * 5 + 20, frame * 5 + 20], fill=(255, 255, 255))
        return img 