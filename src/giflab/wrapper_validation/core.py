"""Core wrapper output validation framework.

This module provides the main WrapperOutputValidator class that coordinates
validation across different dimensions (frame count, color count, timing, etc.).
"""

import logging
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..meta import extract_gif_metadata
from .quality_validation import QualityThresholdValidator
from .types import ValidationResult

logger = logging.getLogger(__name__)


class WrapperOutputValidator:
    """Engine-agnostic validation for wrapper outputs.
    
    This class provides comprehensive validation of wrapper outputs to ensure
    that compression operations achieve their expected results and don't corrupt
    the data in ways that would skew experimental results.
    """
    
    def __init__(self, config=None) -> None:
        """Initialize validator with configuration.
        
        Args:
            config: Validation configuration. Uses defaults if None.
        """
        if config is None:
            # Import here to avoid circular imports
            from ..config import DEFAULT_VALIDATION_CONFIG
            self.config = DEFAULT_VALIDATION_CONFIG
        else:
            self.config = config
            
        # Initialize quality validator
        self.quality_validator = QualityThresholdValidator()
        
    def validate_frame_reduction(
        self,
        input_path: Path,
        output_path: Path,
        expected_ratio: float,
        wrapper_metadata: dict[str, Any]
    ) -> ValidationResult:
        """Validate frame reduction achieved expected ratio.
        
        Compares the actual frame reduction ratio against the expected ratio,
        accounting for rounding and minimum frame requirements.
        
        Args:
            input_path: Original GIF file
            output_path: Compressed GIF file
            expected_ratio: Expected frame reduction ratio (0.0 to 1.0)
            wrapper_metadata: Metadata from wrapper execution
            
        Returns:
            ValidationResult with frame count validation details
        """
        try:
            # Extract frame counts from both files
            input_metadata = extract_gif_metadata(input_path)
            output_metadata = extract_gif_metadata(output_path)
            
            original_frames = input_metadata.orig_frames
            output_frames = output_metadata.orig_frames
            
            # Calculate expected vs actual
            expected_frames = max(self.config.MIN_FRAMES_REQUIRED, int(original_frames * expected_ratio))
            actual_ratio = output_frames / original_frames if original_frames > 0 else 1.0
            
            # Validation logic with tolerance
            ratio_diff = abs(actual_ratio - expected_ratio)
            is_valid = ratio_diff <= self.config.FRAME_RATIO_TOLERANCE and output_frames >= self.config.MIN_FRAMES_REQUIRED
            
            return ValidationResult(
                is_valid=is_valid,
                validation_type="frame_count",
                expected={"ratio": expected_ratio, "frames": expected_frames},
                actual={"ratio": actual_ratio, "frames": output_frames},
                error_message=None if is_valid else f"Frame ratio {actual_ratio:.3f} differs from expected {expected_ratio:.3f} by {ratio_diff:.3f} (tolerance: {self.config.FRAME_RATIO_TOLERANCE})",
                details={
                    "original_frames": original_frames,
                    "tolerance": self.config.FRAME_RATIO_TOLERANCE,
                    "ratio_difference": ratio_diff,
                    "min_frames_required": self.config.MIN_FRAMES_REQUIRED
                }
            )
            
        except Exception as e:
            logger.error(f"Frame validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="frame_count",
                expected=expected_ratio,
                actual=None,
                error_message=f"Frame validation failed: {str(e)}",
                details={"exception": str(e)}
            )
    
    def validate_color_reduction(
        self,
        input_path: Path,
        output_path: Path,
        expected_colors: int,
        wrapper_metadata: dict[str, Any]
    ) -> ValidationResult:
        """Validate color reduction achieved expected color count.
        
        Analyzes the actual color palette in the output GIF and compares it
        to the requested color count, accounting for encoding tolerances.
        
        Args:
            input_path: Original GIF file
            output_path: Compressed GIF file
            expected_colors: Expected maximum color count
            wrapper_metadata: Metadata from wrapper execution
            
        Returns:
            ValidationResult with color count validation details
        """
        try:
            # Extract actual color count from output GIF
            actual_colors = self._count_unique_colors(output_path)
            
            # Get original color count for comparison
            input_metadata = extract_gif_metadata(input_path)
            original_colors = input_metadata.orig_n_colors
            
            # Validation checks
            color_count_valid = actual_colors <= (expected_colors + self.config.COLOR_COUNT_TOLERANCE)
            
            # Check that significant color reduction occurred
            reduction_percent = (original_colors - actual_colors) / original_colors if original_colors > 0 else 0
            reduction_occurred = reduction_percent >= self.config.MIN_COLOR_REDUCTION_PERCENT
            
            is_valid = color_count_valid and (reduction_occurred or expected_colors >= original_colors)
            
            error_parts = []
            if not color_count_valid:
                error_parts.append(f"Color count {actual_colors} exceeds expected {expected_colors} + tolerance {self.config.COLOR_COUNT_TOLERANCE}")
            if not reduction_occurred and expected_colors < original_colors:
                error_parts.append(f"Insufficient color reduction: {reduction_percent:.1%} < {self.config.MIN_COLOR_REDUCTION_PERCENT:.1%}")
                
            return ValidationResult(
                is_valid=is_valid,
                validation_type="color_count",
                expected=expected_colors,
                actual=actual_colors,
                error_message=None if is_valid else "; ".join(error_parts),
                details={
                    "original_colors": original_colors,
                    "reduction_percent": reduction_percent,
                    "reduction_occurred": reduction_occurred,
                    "tolerance": self.config.COLOR_COUNT_TOLERANCE,
                    "min_reduction_required": self.config.MIN_COLOR_REDUCTION_PERCENT
                }
            )
            
        except Exception as e:
            logger.error(f"Color validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="color_count",
                expected=expected_colors,
                actual=None,
                error_message=f"Color validation failed: {str(e)}",
                details={"exception": str(e)}
            )
    
    def validate_timing_preservation(
        self,
        input_path: Path,
        output_path: Path,
        wrapper_metadata: dict[str, Any]
    ) -> ValidationResult:
        """Validate timing/FPS preservation.
        
        Compares the frame timing between input and output GIFs to ensure
        animation speed is preserved appropriately.
        
        Args:
            input_path: Original GIF file
            output_path: Compressed GIF file
            wrapper_metadata: Metadata from wrapper execution
            
        Returns:
            ValidationResult with timing preservation validation details
        """
        try:
            input_metadata = extract_gif_metadata(input_path)
            output_metadata = extract_gif_metadata(output_path)
            
            # Compare FPS (with tolerance for frame reduction effects)
            input_fps = input_metadata.orig_fps
            output_fps = output_metadata.orig_fps
            
            # Validate FPS within reasonable bounds
            fps_in_bounds = (self.config.MIN_FPS <= output_fps <= self.config.MAX_FPS)
            
            # For frame reduction, expect FPS to remain similar (timing should be preserved)
            if input_fps > 0:
                fps_diff = abs(output_fps - input_fps) / input_fps
                fps_preserved = fps_diff <= self.config.FPS_TOLERANCE
            else:
                fps_preserved = output_fps > 0  # Any positive FPS is better than zero
                fps_diff = float('inf') if input_fps == 0 and output_fps > 0 else 0
            
            is_valid = fps_in_bounds and fps_preserved
            
            error_parts = []
            if not fps_in_bounds:
                error_parts.append(f"FPS {output_fps:.2f} outside valid range [{self.config.MIN_FPS}, {self.config.MAX_FPS}]")
            if not fps_preserved:
                error_parts.append(f"FPS changed from {input_fps:.2f} to {output_fps:.2f} (diff: {fps_diff:.1%}, tolerance: {self.config.FPS_TOLERANCE:.1%})")
            
            return ValidationResult(
                is_valid=is_valid,
                validation_type="timing_preservation",
                expected={"fps": input_fps},
                actual={"fps": output_fps},
                error_message=None if is_valid else "; ".join(error_parts),
                details={
                    "fps_difference_percent": fps_diff,
                    "fps_tolerance": self.config.FPS_TOLERANCE,
                    "fps_bounds": (self.config.MIN_FPS, self.config.MAX_FPS),
                    "fps_preserved": fps_preserved,
                    "fps_in_bounds": fps_in_bounds
                }
            )
            
        except Exception as e:
            logger.error(f"Timing validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="timing_preservation",
                expected=None,
                actual=None,
                error_message=f"Timing validation failed: {str(e)}",
                details={"exception": str(e)}
            )
    
    def validate_file_integrity(
        self,
        output_path: Path,
        wrapper_metadata: dict[str, Any]
    ) -> ValidationResult:
        """Validate output file format integrity.
        
        Performs basic checks to ensure the output file is a valid, non-corrupted GIF.
        
        Args:
            output_path: Output GIF file to validate
            wrapper_metadata: Metadata from wrapper execution
            
        Returns:
            ValidationResult with file integrity validation details
        """
        try:
            # Basic file existence and size checks
            if not output_path.exists():
                return ValidationResult(
                    is_valid=False,
                    validation_type="file_integrity",
                    expected="file_exists",
                    actual="file_missing",
                    error_message="Output file does not exist"
                )
            
            file_size = output_path.stat().st_size
            if file_size < self.config.MIN_FILE_SIZE_BYTES:
                return ValidationResult(
                    is_valid=False,
                    validation_type="file_integrity",
                    expected=f"size >= {self.config.MIN_FILE_SIZE_BYTES} bytes",
                    actual=f"{file_size} bytes",
                    error_message=f"Output file too small: {file_size} bytes < {self.config.MIN_FILE_SIZE_BYTES} bytes minimum"
                )
            
            max_size_bytes = int(self.config.MAX_FILE_SIZE_MB * 1024 * 1024)
            if file_size > max_size_bytes:
                return ValidationResult(
                    is_valid=False,
                    validation_type="file_integrity",
                    expected=f"size <= {self.config.MAX_FILE_SIZE_MB}MB",
                    actual=f"{file_size / 1024 / 1024:.1f}MB",
                    error_message=f"Output file too large: {file_size / 1024 / 1024:.1f}MB > {self.config.MAX_FILE_SIZE_MB}MB maximum"
                )
            
            # Try to open and validate as GIF
            try:
                with Image.open(output_path) as img:
                    if img.format != "GIF":
                        return ValidationResult(
                            is_valid=False,
                            validation_type="file_integrity",
                            expected="GIF format",
                            actual=img.format or "unknown",
                            error_message=f"Output file is not a GIF (format: {img.format})"
                        )
                    
                    # Basic integrity checks
                    width, height = img.size
                    if width <= 0 or height <= 0:
                        return ValidationResult(
                            is_valid=False,
                            validation_type="file_integrity",
                            expected="positive dimensions",
                            actual=f"{width}x{height}",
                            error_message=f"Invalid dimensions: {width}x{height}"
                        )
                    
                    # Check if we can read at least the first frame
                    img.seek(0)
                    img.load()  # This will raise an exception if the image is corrupted
                    
            except Exception as img_error:
                return ValidationResult(
                    is_valid=False,
                    validation_type="file_integrity",
                    expected="valid GIF file",
                    actual="corrupted or invalid",
                    error_message=f"Cannot read output file as valid GIF: {str(img_error)}",
                    details={"image_error": str(img_error)}
                )
            
            # All checks passed
            return ValidationResult(
                is_valid=True,
                validation_type="file_integrity",
                expected="valid GIF file",
                actual="valid GIF file",
                details={
                    "file_size_bytes": file_size,
                    "file_size_mb": file_size / 1024 / 1024,
                    "dimensions": f"{width}x{height}"
                }
            )
            
        except Exception as e:
            logger.error(f"File integrity validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="file_integrity",
                expected="valid file",
                actual="validation error",
                error_message=f"File integrity validation failed: {str(e)}",
                details={"exception": str(e)}
            )
    
    def validate_wrapper_output(
        self,
        input_path: Path,
        output_path: Path,
        wrapper_params: dict[str, Any],
        wrapper_metadata: dict[str, Any],
        wrapper_type: str
    ) -> list[ValidationResult]:
        """Run comprehensive validation for wrapper output.
        
        This is the main entry point that runs appropriate validations based
        on the wrapper type and parameters.
        
        Args:
            input_path: Original input file
            output_path: Wrapper output file
            wrapper_params: Parameters passed to wrapper
            wrapper_metadata: Metadata returned by wrapper
            wrapper_type: Type of wrapper ("frame_reduction", "color_reduction", "lossy_compression")
            
        Returns:
            List of ValidationResult objects for all applicable validations
        """
        validations = []
        
        # Always check file integrity first
        integrity_result = self.validate_file_integrity(output_path, wrapper_metadata)
        validations.append(integrity_result)
        
        # If file integrity failed, skip other validations
        if not integrity_result.is_valid:
            return validations
        
        # Always check timing preservation
        timing_result = self.validate_timing_preservation(input_path, output_path, wrapper_metadata)
        validations.append(timing_result)
        
        # Type-specific validations
        if wrapper_type == "frame_reduction" and "ratio" in wrapper_params:
            frame_result = self.validate_frame_reduction(
                input_path, output_path, wrapper_params["ratio"], wrapper_metadata
            )
            validations.append(frame_result)
            
        elif wrapper_type == "color_reduction" and "colors" in wrapper_params:
            color_result = self.validate_color_reduction(
                input_path, output_path, wrapper_params["colors"], wrapper_metadata
            )
            validations.append(color_result)
        
        # Add quality degradation validation for all operations
        # This detects catastrophic quality failures using existing metrics
        try:
            quality_result = self.quality_validator.validate_quality_degradation(
                input_path, output_path, wrapper_metadata, wrapper_type
            )
            validations.append(quality_result)
        except Exception as e:
            logger.warning(f"Quality validation skipped due to error: {e}")
            # Don't fail the entire validation if quality validation has issues
        
        return validations
    
    def _count_unique_colors(self, gif_path: Path) -> int:
        """Count unique colors in GIF using PIL palette analysis.
        
        Args:
            gif_path: Path to GIF file
            
        Returns:
            Number of unique colors, or estimated count if exact count unavailable
        """
        try:
            with Image.open(gif_path) as img:
                # Try direct color counting first
                if hasattr(img, 'getcolors'):
                    colors = img.getcolors(maxcolors=256*256*256)
                    if colors:
                        return len(colors)
                
                # Fallback: analyze palette
                if hasattr(img, 'palette') and img.palette:
                    # Palette mode - count palette entries
                    palette_data = img.palette.getdata()[1]
                    if palette_data:
                        # Count non-duplicate colors in palette
                        unique_colors = set()
                        for i in range(0, len(palette_data), 3):
                            if i + 2 < len(palette_data):
                                color = (palette_data[i], palette_data[i+1], palette_data[i+2])
                                unique_colors.add(color)
                        return len(unique_colors)
                
                # Convert to RGB and sample for color estimation
                rgb_img = img.convert("RGB")
                colors = rgb_img.getcolors(maxcolors=256*256*256)
                if colors:
                    return len(colors)
                
                # Final fallback - assume 256 colors for indexed images
                if img.mode in ('P', 'L'):
                    return 256
                else:
                    return 16777216  # 24-bit color space
                    
        except Exception as e:
            logger.warning(f"Could not count colors in {gif_path}: {e}")
            return 256  # Conservative fallback
