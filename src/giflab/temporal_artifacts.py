"""
Enhanced temporal artifact detection for GIF compression validation.

This module implements advanced temporal artifact detection metrics designed specifically
for debugging compression failures, including:
- Flicker excess detection using LPIPS-T between consecutive frames
- Flat-region flicker detection for background stability validation
- Enhanced disposal artifact detection with better background tracking
- Temporal pumping detection for quality oscillation
"""

import logging
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    logger.warning(
        "LPIPS not available. Temporal artifact detection will use fallback methods."
    )
    LPIPS_AVAILABLE = False


class TemporalArtifactDetector:
    """Enhanced temporal artifact detector using perceptual metrics."""

    def __init__(self, device: str | None = None):
        """Initialize temporal artifact detector.

        Args:
            device: PyTorch device ('cpu', 'cuda', etc.). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model: None | Any | bool = None

    def _get_lpips_model(self) -> Any | None:
        """Lazy initialization of LPIPS model."""
        if self._lpips_model is None and LPIPS_AVAILABLE:
            try:
                model = lpips.LPIPS(net="alex", spatial=False).to(self.device)
                model.eval()
                self._lpips_model = model
            except Exception as e:
                logger.warning(f"Failed to initialize LPIPS model: {e}")
                self._lpips_model = False
        return self._lpips_model if self._lpips_model is not False else None

    def preprocess_for_lpips(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for LPIPS calculation.

        Args:
            frame: RGB frame as numpy array [H, W, 3]

        Returns:
            Preprocessed tensor ready for LPIPS [1, 3, H, W]
        """
        # Ensure frame is RGB and float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32) / 255.0

        # Convert to torch tensor and normalize to [-1, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor * 2.0 - 1.0

        return frame_tensor.to(self.device)

    def calculate_lpips_temporal(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Calculate LPIPS between consecutive frames for temporal consistency.

        Args:
            frames: List of RGB frames as numpy arrays

        Returns:
            Dictionary with LPIPS temporal metrics
        """
        if len(frames) < 2:
            return {
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "lpips_t_max": 0.0,
                "lpips_frame_count": len(frames),
            }

        lpips_model = self._get_lpips_model()

        if lpips_model is None:
            # Fallback to MSE-based temporal difference
            logger.debug("Using MSE fallback for LPIPS temporal calculation")
            return self._calculate_mse_temporal_fallback(frames)

        lpips_scores = []

        try:
            with torch.no_grad():
                for i in range(len(frames) - 1):
                    frame1_tensor = self.preprocess_for_lpips(frames[i])
                    frame2_tensor = self.preprocess_for_lpips(frames[i + 1])

                    # Calculate perceptual distance
                    distance = lpips_model(frame1_tensor, frame2_tensor)
                    lpips_scores.append(float(distance.cpu().item()))

        except Exception as e:
            logger.error(f"Error calculating LPIPS temporal: {e}")
            return self._calculate_mse_temporal_fallback(frames)

        if not lpips_scores:
            return {
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "lpips_t_max": 0.0,
                "lpips_frame_count": len(frames),
            }

        return {
            "lpips_t_mean": float(np.mean(lpips_scores)),
            "lpips_t_p95": float(np.percentile(lpips_scores, 95)),
            "lpips_t_max": float(np.max(lpips_scores)),
            "lpips_frame_count": len(frames),
        }

    def _calculate_mse_temporal_fallback(
        self, frames: list[np.ndarray]
    ) -> dict[str, float]:
        """Fallback MSE-based temporal calculation when LPIPS is unavailable."""
        mse_scores = []

        for i in range(len(frames) - 1):
            frame1 = frames[i].astype(np.float32) / 255.0
            frame2 = frames[i + 1].astype(np.float32) / 255.0

            mse = float(np.mean((frame1 - frame2) ** 2))
            # Normalize MSE to approximate LPIPS range [0, 1]
            normalized_mse = min(mse * 10.0, 1.0)
            mse_scores.append(normalized_mse)

        if not mse_scores:
            return {
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "lpips_t_max": 0.0,
                "lpips_frame_count": len(frames),
            }

        return {
            "lpips_t_mean": float(np.mean(mse_scores)),
            "lpips_t_p95": float(np.percentile(mse_scores, 95)),
            "lpips_t_max": float(np.max(mse_scores)),
            "lpips_frame_count": len(frames),
        }

    def detect_flicker_excess(
        self, frames: list[np.ndarray], threshold: float = 0.02
    ) -> dict[str, float]:
        """Detect flicker excess using LPIPS-T between consecutive frames.

        Flicker manifests as high perceptual differences between consecutive frames
        that don't correlate with intended animation patterns.

        Args:
            frames: List of RGB frames
            threshold: LPIPS-T threshold above which flicker is detected

        Returns:
            Dictionary with flicker excess metrics
        """
        lpips_metrics = self.calculate_lpips_temporal(frames)

        # Calculate flicker excess - how much perceptual difference exceeds expected
        lpips_scores = []
        if len(frames) >= 2:
            lpips_model = self._get_lpips_model()
            if lpips_model is not None:
                try:
                    with torch.no_grad():
                        for i in range(len(frames) - 1):
                            frame1_tensor = self.preprocess_for_lpips(frames[i])
                            frame2_tensor = self.preprocess_for_lpips(frames[i + 1])
                            distance = lpips_model(frame1_tensor, frame2_tensor)
                            lpips_scores.append(float(distance.cpu().item()))
                except Exception as e:
                    logger.error(f"Error in flicker detection: {e}")

        # Analyze flicker patterns
        flicker_excess = 0.0
        flicker_frames = 0

        if lpips_scores:
            # Identify frames with excessive perceptual difference
            excessive_frames = [score > threshold for score in lpips_scores]
            flicker_frames = sum(excessive_frames)

            # Calculate excess beyond threshold
            excess_values = [max(0, score - threshold) for score in lpips_scores]
            flicker_excess = float(np.mean(excess_values))

        return {
            "flicker_excess": flicker_excess,
            "flicker_frame_count": flicker_frames,
            "flicker_frame_ratio": flicker_frames / max(1, len(frames) - 1),
            "lpips_t_mean": lpips_metrics["lpips_t_mean"],
            "lpips_t_p95": lpips_metrics["lpips_t_p95"],
        }

    def identify_flat_regions(
        self, frame: np.ndarray, variance_threshold: float = 10.0
    ) -> list[tuple[int, int, int, int]]:
        """Identify flat/stable regions in a frame that should not flicker.

        Args:
            frame: RGB frame as numpy array
            variance_threshold: Pixel variance threshold for flat regions

        Returns:
            List of (x, y, width, height) tuples for flat regions
        """
        # Convert to grayscale for variance calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Use sliding window to find low-variance regions
        kernel_size = 16
        flat_regions = []

        h, w = gray.shape
        step = kernel_size // 2

        for y in range(0, h - kernel_size, step):
            for x in range(0, w - kernel_size, step):
                patch = gray[y : y + kernel_size, x : x + kernel_size]
                variance = float(np.var(patch))

                if variance < variance_threshold:
                    flat_regions.append((x, y, kernel_size, kernel_size))

        # Merge overlapping regions (simplified approach)
        if len(flat_regions) > 1:
            flat_regions = self._merge_overlapping_regions(flat_regions)

        return flat_regions

    def _merge_overlapping_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping rectangular regions."""
        if not regions:
            return []

        # Simple merge: combine regions that overlap by >50%
        merged = [regions[0]]

        for region in regions[1:]:
            x, y, w, h = region
            merged_with_existing = False

            for i, existing in enumerate(merged):
                ex, ey, ew, eh = existing

                # Check for overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y

                region_area = w * h
                existing_area = ew * eh

                if overlap_area > 0.5 * min(region_area, existing_area):
                    # Merge regions
                    new_x = min(x, ex)
                    new_y = min(y, ey)
                    new_w = max(x + w, ex + ew) - new_x
                    new_h = max(y + h, ey + eh) - new_y
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_with_existing = True
                    break

            if not merged_with_existing:
                merged.append(region)

        return merged

    def detect_flat_region_flicker(
        self, frames: list[np.ndarray], variance_threshold: float = 10.0
    ) -> dict[str, float]:
        """Detect flicker in regions that should be stable (background areas).

        Args:
            frames: List of RGB frames
            variance_threshold: Threshold for identifying flat regions

        Returns:
            Dictionary with flat region flicker metrics
        """
        if len(frames) < 2:
            return {
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "flat_region_variance_mean": 0.0,
            }

        # Identify flat regions in first frame
        flat_regions = self.identify_flat_regions(frames[0], variance_threshold)

        if not flat_regions:
            return {
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "flat_region_variance_mean": 0.0,
            }

        # Track variance in these regions across frames
        region_variances = []

        for region in flat_regions:
            x, y, w, h = region

            # Extract region pixels from all frames
            region_pixels = []
            for frame in frames:
                # Ensure bounds are within frame
                actual_h, actual_w = frame.shape[:2]
                x_end = min(x + w, actual_w)
                y_end = min(y + h, actual_h)

                if x < actual_w and y < actual_h:
                    patch = frame[y:y_end, x:x_end]
                    region_pixels.append(patch.astype(np.float32))

            if len(region_pixels) >= 2:
                # Calculate temporal variance for this region
                temporal_variance = self._calculate_temporal_variance(region_pixels)
                region_variances.append(temporal_variance)

        if not region_variances:
            return {
                "flat_flicker_ratio": 0.0,
                "flat_region_count": len(flat_regions),
                "flat_region_variance_mean": 0.0,
            }

        mean_variance = float(np.mean(region_variances))

        # High variance in flat regions indicates flicker
        # Use adaptive threshold based on overall image characteristics
        flicker_threshold = variance_threshold * 2.0
        flickering_regions = sum(
            1 for var in region_variances if var > flicker_threshold
        )

        return {
            "flat_flicker_ratio": flickering_regions / len(flat_regions),
            "flat_region_count": len(flat_regions),
            "flat_region_variance_mean": mean_variance,
        }

    def _calculate_temporal_variance(self, region_pixels: list[np.ndarray]) -> float:
        """Calculate temporal variance for a region across frames."""
        if len(region_pixels) < 2:
            return 0.0

        # Stack patches and calculate variance across time
        try:
            # Ensure all patches have the same shape
            min_h = min(patch.shape[0] for patch in region_pixels)
            min_w = min(patch.shape[1] for patch in region_pixels)

            normalized_patches = []
            for patch in region_pixels:
                resized_patch = patch[:min_h, :min_w]
                normalized_patches.append(resized_patch)

            # Stack along time axis
            temporal_stack = np.stack(
                normalized_patches, axis=0
            )  # [time, h, w, channels]

            # Calculate variance across time axis for each pixel
            temporal_variance = np.var(temporal_stack, axis=0)

            # Return mean variance across all pixels in region
            return float(np.mean(temporal_variance))

        except Exception as e:
            logger.error(f"Error calculating temporal variance: {e}")
            return 0.0

    def detect_temporal_pumping(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Detect temporal pumping - quality oscillation between frames.

        Temporal pumping manifests as cyclic quality variations that create
        a "breathing" or "pumping" visual effect.

        Args:
            frames: List of RGB frames

        Returns:
            Dictionary with temporal pumping metrics
        """
        if len(frames) < 3:
            return {
                "temporal_pumping_score": 0.0,
                "quality_oscillation_frequency": 0.0,
                "quality_variance": 0.0,
            }

        # Calculate frame-wise quality metrics
        frame_qualities = []

        for frame in frames:
            # Use multiple quality indicators
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Edge density as quality indicator
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)

            # Local variance as quality indicator
            local_variance = float(np.var(gray))

            # Combine metrics
            quality = edge_density * 0.7 + (local_variance / 10000.0) * 0.3
            frame_qualities.append(quality)

        # Analyze quality oscillation
        quality_variance = float(np.var(frame_qualities))

        # Detect periodic patterns in quality
        quality_diffs = np.diff(frame_qualities)
        oscillation_score = 0.0

        if len(quality_diffs) >= 2:
            # Look for alternating pattern in quality differences
            sign_changes = 0
            for i in range(len(quality_diffs) - 1):
                if np.sign(quality_diffs[i]) != np.sign(quality_diffs[i + 1]):
                    sign_changes += 1

            # High sign changes indicate oscillation
            oscillation_frequency = sign_changes / len(quality_diffs)

            # Combine frequency with magnitude
            oscillation_magnitude = float(np.std(quality_diffs))
            oscillation_score = oscillation_frequency * oscillation_magnitude

        return {
            "temporal_pumping_score": oscillation_score,
            "quality_oscillation_frequency": oscillation_frequency
            if "oscillation_frequency" in locals()
            else 0.0,
            "quality_variance": quality_variance,
        }


def calculate_enhanced_temporal_metrics(
    original_frames: list[np.ndarray],
    compressed_frames: list[np.ndarray],
    device: str | None = None,
) -> dict[str, float]:
    """Calculate comprehensive temporal artifact metrics for validation.

    This is the main entry point for temporal artifact detection that integrates
    all the enhanced detection methods.

    Args:
        original_frames: List of original RGB frames
        compressed_frames: List of compressed RGB frames
        device: PyTorch device for LPIPS computation

    Returns:
        Dictionary with all temporal artifact metrics
    """
    detector = TemporalArtifactDetector(device=device)

    # Ensure frames are aligned
    min_frame_count = min(len(original_frames), len(compressed_frames))
    if min_frame_count == 0:
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }

    original_frames = original_frames[:min_frame_count]
    compressed_frames = compressed_frames[:min_frame_count]

    # Calculate temporal metrics on compressed frames
    flicker_metrics = detector.detect_flicker_excess(compressed_frames)
    flat_flicker_metrics = detector.detect_flat_region_flicker(compressed_frames)
    pumping_metrics = detector.detect_temporal_pumping(compressed_frames)

    # Combine all metrics
    combined_metrics = {
        "flicker_excess": flicker_metrics["flicker_excess"],
        "flicker_frame_ratio": flicker_metrics["flicker_frame_ratio"],
        "flat_flicker_ratio": flat_flicker_metrics["flat_flicker_ratio"],
        "flat_region_count": flat_flicker_metrics["flat_region_count"],
        "temporal_pumping_score": pumping_metrics["temporal_pumping_score"],
        "quality_oscillation_frequency": pumping_metrics[
            "quality_oscillation_frequency"
        ],
        "lpips_t_mean": flicker_metrics["lpips_t_mean"],
        "lpips_t_p95": flicker_metrics["lpips_t_p95"],
        "frame_count": min_frame_count,
    }

    return combined_metrics


def calculate_enhanced_temporal_metrics_from_paths(
    original_path: str, compressed_path: str, device: str | None = None
) -> dict[str, float]:
    """Calculate enhanced temporal metrics from GIF file paths.
    
    This is a convenience wrapper around calculate_enhanced_temporal_metrics
    that handles frame extraction from file paths.
    
    Args:
        original_path: Path to original GIF file
        compressed_path: Path to compressed GIF file
        device: PyTorch device for LPIPS computation
        
    Returns:
        Dictionary with temporal artifact metrics
    """
    try:
        # Import frame extraction utility
        from pathlib import Path

        from .metrics import extract_gif_frames
        
        # Extract frames from both files
        original_result = extract_gif_frames(Path(original_path))
        compressed_result = extract_gif_frames(Path(compressed_path))
        
        original_frames = original_result.frames
        compressed_frames = compressed_result.frames
        
        return calculate_enhanced_temporal_metrics(
            original_frames, compressed_frames, device=device
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate temporal metrics from paths: {e}")
        return {
            "flicker_excess": 0.0,
            "flat_flicker_ratio": 0.0,
            "temporal_pumping_score": 0.0,
            "lpips_t_mean": 0.0,
            "lpips_t_p95": 0.0,
        }
