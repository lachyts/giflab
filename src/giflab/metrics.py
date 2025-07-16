"""Quality metrics and comparison functionality for GIF analysis."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import math
from skimage.feature import local_binary_pattern

from .config import DEFAULT_METRICS_CONFIG, MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractResult:
    """Result of frame extraction from a GIF."""
    frames: list[np.ndarray]
    frame_count: int
    dimensions: tuple[int, int]  # (width, height)
    duration_ms: int


def extract_gif_frames(gif_path: Path, max_frames: int | None = None) -> FrameExtractResult:
    """Extract frames from a GIF file.

    Args:
        gif_path: Path to GIF file
        max_frames: Maximum number of frames to extract (None for all)

    Returns:
        FrameExtractResult with extracted frames and metadata

    Raises:
        IOError: If GIF cannot be read
        ValueError: If GIF is invalid or corrupted
    """
    try:
        with Image.open(gif_path) as img:
            if not hasattr(img, 'n_frames') or img.n_frames == 1:
                # Single frame image (PNG, JPEG, etc.) or single-frame GIF
                frame = np.array(img.convert('RGB'))
                return FrameExtractResult(
                    frames=[frame],
                    frame_count=1,
                    dimensions=(img.width, img.height),
                    duration_ms=0
                )

            total_frames = img.n_frames

            # Memory protection: limit frame extraction for very large GIFs
            memory_limit_frames = 500  # Reasonable limit to prevent memory issues
            if max_frames is None:
                frames_to_extract = min(total_frames, memory_limit_frames)
            else:
                frames_to_extract = min(total_frames, max_frames, memory_limit_frames)

            # Additional memory check based on image dimensions
            width, height = img.size
            pixels_per_frame = width * height * 3  # RGB
            estimated_memory_mb = (frames_to_extract * pixels_per_frame) / (1024 * 1024)

            # Limit memory usage to ~500MB for frame extraction
            if estimated_memory_mb > 500:
                max_safe_frames = int(500 * 1024 * 1024 / pixels_per_frame)
                frames_to_extract = min(frames_to_extract, max(1, max_safe_frames))

            frames = []
            total_duration = 0

            for i in range(frames_to_extract):
                img.seek(i)
                frame = np.array(img.convert('RGB'))
                frames.append(frame)

                # Get frame duration
                duration = img.info.get('duration', 100)  # Default 100ms
                total_duration += duration

            return FrameExtractResult(
                frames=frames,
                frame_count=len(frames),
                dimensions=(img.width, img.height),
                duration_ms=total_duration
            )

    except Exception as e:
        raise OSError(f"Failed to extract frames from {gif_path}: {e}") from e


def resize_to_common_dimensions(frames1: list[np.ndarray], frames2: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Resize frames to common dimensions (smallest common size).

    Args:
        frames1: First set of frames
        frames2: Second set of frames

    Returns:
        Tuple of (resized_frames1, resized_frames2)
    """
    if not frames1 or not frames2:
        return frames1, frames2

    # Validate frames have proper dimensions
    if len(frames1[0].shape) < 2 or len(frames2[0].shape) < 2:
        raise ValueError("Frames must have at least 2 dimensions")

    # Get dimensions
    h1, w1 = frames1[0].shape[:2]
    h2, w2 = frames2[0].shape[:2]

    # Validate dimensions are positive
    if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
        raise ValueError(f"Invalid frame dimensions: {h1}x{w1} and {h2}x{w2}")

    # Use smallest common dimensions
    target_h = min(h1, h2)
    target_w = min(w1, w2)

    # Ensure minimum dimensions for processing
    target_h = max(target_h, 1)
    target_w = max(target_w, 1)

    # Resize if necessary
    resized_frames1 = []
    for frame in frames1:
        if len(frame.shape) < 2:
            raise ValueError("Frame has invalid shape")

        if frame.shape[:2] != (target_h, target_w):
            try:
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                resized_frames1.append(resized)
            except Exception as e:
                raise ValueError(f"Failed to resize frame: {e}") from e
        else:
            resized_frames1.append(frame)

    resized_frames2 = []
    for frame in frames2:
        if len(frame.shape) < 2:
            raise ValueError("Frame has invalid shape")

        if frame.shape[:2] != (target_h, target_w):
            try:
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                resized_frames2.append(resized)
            except Exception as e:
                raise ValueError(f"Failed to resize frame: {e}") from e
        else:
            resized_frames2.append(frame)

    return resized_frames1, resized_frames2





def calculate_frame_mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate mean squared error between two frames."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    return float(np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2))


def align_frames_content_based(original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Content-based alignment - find most similar frames using MSE.

    This is the most robust alignment method as it finds actual visual matches
    regardless of temporal position or compression patterns."""
    if not original_frames or not compressed_frames:
        return []

    aligned_pairs = []
    used_compressed_indices = set()

    for orig_frame in original_frames:
        best_match_idx = -1
        best_mse = float('inf')

        for comp_idx, comp_frame in enumerate(compressed_frames):
            if comp_idx in used_compressed_indices:
                continue

            try:
                mse = calculate_frame_mse(orig_frame, comp_frame)
                # Validate MSE is finite and reasonable
                if not np.isfinite(mse):
                    logger.warning(f"Non-finite MSE calculated for frame pair {comp_idx}")
                    continue

                if mse < best_mse:
                    best_mse = mse
                    best_match_idx = comp_idx
            except Exception as e:
                logger.warning(f"MSE calculation failed for frame {comp_idx}: {e}")
                continue

        # Only add pair if we found a valid match with finite MSE
        if best_match_idx >= 0 and np.isfinite(best_mse):
            aligned_pairs.append((orig_frame, compressed_frames[best_match_idx]))
            used_compressed_indices.add(best_match_idx)
        else:
            logger.warning(f"No valid frame match found for original frame (best_mse={best_mse})")

    return aligned_pairs



def align_frames(original_frames: list[np.ndarray], compressed_frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Align frames using content-based matching (most robust approach).

    Args:
        original_frames: Original GIF frames
        compressed_frames: Compressed GIF frames

    Returns:
        List of aligned frame pairs based on visual similarity
    """
    return align_frames_content_based(original_frames, compressed_frames)


def calculate_ms_ssim(frame1: np.ndarray, frame2: np.ndarray, scales: int = 5) -> float:
    """Calculate Multi-Scale SSIM (MS-SSIM) between two frames.

    Args:
        frame1: First frame
        frame2: Second frame
        scales: Number of scales for MS-SSIM (default 5)

    Returns:
        MS-SSIM value between 0.0 and 1.0
    """
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Convert to grayscale for MS-SSIM calculation
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    else:
        frame1_gray = frame1
        frame2_gray = frame2

    # Calculate MS-SSIM using pyramid approach
    ssim_values = []
    current_frame1 = frame1_gray.astype(np.float32)
    current_frame2 = frame2_gray.astype(np.float32)

    # Safety check: limit scales to prevent infinite loops
    max_possible_scales = min(scales, 10)  # Hard limit to prevent runaway loops

    for scale in range(max_possible_scales):
        # Calculate SSIM at current scale
        try:
            scale_ssim = ssim(current_frame1, current_frame2, data_range=255.0)
            ssim_values.append(scale_ssim)
        except (ValueError, RuntimeError) as e:
            # If SSIM calculation fails due to invalid data or computation error, use a default value
            logger.warning(f"SSIM calculation failed at scale {scale}: {e}")
            ssim_values.append(0.0)

        # Downsample for next scale (if not the last scale)
        if scale < max_possible_scales - 1:
            prev_shape = current_frame1.shape
            current_frame1 = cv2.resize(current_frame1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            current_frame2 = cv2.resize(current_frame2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # Stop if frames become too small OR if size didn't change (safety check)
            if (current_frame1.shape[0] < 8 or current_frame1.shape[1] < 8 or
                current_frame1.shape == prev_shape):
                break

    # Weighted average of SSIM values across scales
    if ssim_values:
        weights = [0.4, 0.25, 0.15, 0.1, 0.1][:len(ssim_values)]
        weights = np.array(weights)

        # Protect against division by zero in weight normalization
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum  # Normalize weights
            return np.average(ssim_values, weights=weights)
        else:
            # If all weights are zero, use uniform weighting
            return np.mean(ssim_values)
    else:
        return 0.0


def calculate_temporal_consistency(frames: list[np.ndarray]) -> float:
    """Calculate temporal consistency (animation smoothness) of frames.

    Args:
        frames: List of consecutive frames

    Returns:
        Temporal consistency value between 0.0 and 1.0 (higher = more consistent)
    """
    if len(frames) < 2:
        return 1.0  # Single frame is perfectly consistent

    frame_differences = []

    for i in range(len(frames) - 1):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)

        # Calculate frame-to-frame difference
        diff = float(np.mean(np.abs(frame1 - frame2)))
        frame_differences.append(diff)

    if not frame_differences:
        return 1.0

    # Calculate consistency as inverse of variance in frame differences
    # More consistent animations have lower variance in frame-to-frame changes
    mean_diff = float(np.mean(frame_differences))
    variance_diff = float(np.var(frame_differences))

    # Normalize to 0-1 range (higher = more consistent)
    if mean_diff == 0:
        return 1.0

    # Protect against division by zero and handle edge cases
    epsilon = 1e-8
    if variance_diff == 0:
        # Perfect consistency - no variance in frame differences
        return 1.0

    # Use max to ensure we don't divide by zero
    denominator = max(mean_diff, epsilon)
    consistency = 1.0 / (1.0 + variance_diff / denominator)
    return float(max(0.0, min(1.0, consistency)))


# Legacy compatibility functions
def calculate_ssim(original_path: Path, compressed_path: Path) -> float:
    """Calculate Structural Similarity Index (SSIM) between two GIFs.

    Legacy function - use calculate_comprehensive_metrics for full functionality.
    """
    try:
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)
        return metrics["ssim"]
    except Exception as e:
        logger.error(f"SSIM calculation failed: {e}")
        return 0.0


def calculate_file_size_kb(file_path: Path) -> float:
    """Calculate file size in kilobytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in kilobytes (KB)

    Raises:
        IOError: If file cannot be accessed
    """
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / 1024.0  # Convert bytes to KB
    except OSError as e:
        raise OSError(f"Cannot access file {file_path}: {e}") from e


def measure_render_time(func, *args, **kwargs) -> tuple[Any, int]:
    """Measure execution time of a function in milliseconds.

    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (function_result, execution_time_ms)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
    execution_time_ms = min(int(elapsed_seconds * 1000), 86400000)
    return result, execution_time_ms


def compare_gif_frames(gif1_path: Path, gif2_path: Path) -> dict[str, Any]:
    """Compare frames between two GIF files for quality analysis.

    Legacy function - use calculate_comprehensive_metrics for full functionality.
    """
    try:
        metrics = calculate_comprehensive_metrics(gif1_path, gif2_path)
        return {
            "frame_count_original": len(extract_gif_frames(gif1_path).frames),
            "frame_count_compressed": len(extract_gif_frames(gif2_path).frames),
            "quality_metrics": metrics
        }
    except Exception as e:
        logger.error(f"Frame comparison failed: {e}")
        return {"error": str(e)}


def calculate_compression_ratio(original_size_kb: float, compressed_size_kb: float) -> float:
    """Calculate compression ratio between original and compressed files.

    Args:
        original_size_kb: Original file size in KB
        compressed_size_kb: Compressed file size in KB

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    if compressed_size_kb <= 0:
        raise ValueError("Compressed size must be positive")

    return original_size_kb / compressed_size_kb

# ---------------- New helper and metric functions (Stage-1) ---------------- #

def _resize_if_needed(frame1: np.ndarray, frame2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize both frames to the smallest common size if their shapes differ.

    The function keeps the aspect ratio by simply resizing to the *minimum* of the
    two input shapes. This avoids any padding/cropping artefacts while ensuring
    metric functions receive arrays with identical dimensions.
    """
    if frame1.shape[:2] == frame2.shape[:2]:
        return frame1, frame2

    target_h = min(frame1.shape[0], frame2.shape[0])
    target_w = min(frame1.shape[1], frame2.shape[1])

    try:
        frame1_resized = cv2.resize(frame1, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except Exception as exc:  # pragma: no cover – surface as ValueError for callers
        raise ValueError(f"Failed to resize frames to common size: {exc}") from exc

    return frame1_resized, frame2_resized


def mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Mean-Squared Error (lower is better).

    Returns a non-negative float. Identical frames ⇒ 0.0.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)
    return float(np.mean((f1.astype(np.float32) - f2.astype(np.float32)) ** 2))


def rmse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Root-Mean-Squared Error (sqrt of MSE)."""
    return math.sqrt(mse(frame1, frame2))


def fsim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Feature Similarity Index (approximated implementation).

    This lightweight approximation returns the *mean* of the combined
    gradient-magnitude and phase-congruency similarity maps, which empirically
    yields higher scores for identical images and lower scores for dissimilar
    ones.
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    # Grayscale conversion.
    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    # Gradient magnitude (Sobel).
    def _grad_mag(img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    G1 = _grad_mag(gray1)
    G2 = _grad_mag(gray2)

    # Phase-congruency proxy using Laplacian magnitude.
    PC1 = np.abs(cv2.Laplacian(gray1, cv2.CV_32F))
    PC2 = np.abs(cv2.Laplacian(gray2, cv2.CV_32F))

    T1 = 1e-3
    T2 = 1e-3
    gradient_sim = (2 * G1 * G2 + T1) / (G1 ** 2 + G2 ** 2 + T1)
    pc_sim = (2 * PC1 * PC2 + T2) / (PC1 ** 2 + PC2 ** 2 + T2)

    fsim_map = gradient_sim * pc_sim

    return float(np.clip(np.mean(fsim_map), 0.0, 1.0))


def gmsd(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Gradient Magnitude Similarity Deviation (lower is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    # Prewitt kernels.
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 3.0

    def _prewitt(img: np.ndarray) -> np.ndarray:
        gx = cv2.filter2D(img, -1, prewitt_x)
        gy = cv2.filter2D(img, -1, prewitt_y)
        return np.sqrt(gx ** 2 + gy ** 2)

    M1 = _prewitt(gray1)
    M2 = _prewitt(gray2)

    C = 1e-3  # stability constant
    gms_map = (2 * M1 * M2 + C) / (M1 ** 2 + M2 ** 2 + C)

    return float(np.std(gms_map))


def chist(frame1: np.ndarray, frame2: np.ndarray, bins: int = 32) -> float:
    """Colour-Histogram Correlation (0-1, higher is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)
    scores: list[float] = []
    for ch in range(3):  # R,G,B channels
        h1 = cv2.calcHist([f1], [ch], None, [bins], [0, 256])
        h2 = cv2.calcHist([f2], [ch], None, [bins], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        scores.append(corr)
    # cv2 correlation in [-1,1] – map to [0,1]
    return float(np.clip((np.mean(scores) + 1) / 2.0, 0.0, 1.0))


def edge_similarity(frame1: np.ndarray, frame2: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> float:
    """Edge-Map Jaccard similarity (0-1, higher is better).

    Args:
        frame1: First frame (RGB or grayscale)
        frame2: Second frame (RGB or grayscale)
        threshold1: Lower Canny threshold
        threshold2: Upper Canny threshold
    """
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    edges1 = cv2.Canny(gray1, threshold1, threshold2)
    edges2 = cv2.Canny(gray2, threshold1, threshold2)

    intersection = np.logical_and(edges1 > 0, edges2 > 0).sum()
    union = np.logical_or(edges1 > 0, edges2 > 0).sum()
    if union == 0:
        return 1.0  # no edges at all
    return float(intersection / union)


def texture_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Texture-Histogram correlation using uniform LBP (0-1, higher is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    radius = 1
    n_points = 8 * radius
    lbp1 = local_binary_pattern(gray1, n_points, radius, "uniform")
    lbp2 = local_binary_pattern(gray2, n_points, radius, "uniform")

    hist1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, n_points + 2), density=True)
    hist2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, n_points + 2), density=True)

    if np.std(hist1) == 0 or np.std(hist2) == 0:
        return 1.0  # completely uniform textures

    corr = np.corrcoef(hist1, hist2)[0, 1]
    return float(np.clip((corr + 1) / 2.0, 0.0, 1.0))


def sharpness_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Sharpness similarity based on Laplacian variance ratio (0-1, higher is better)."""
    f1, f2 = _resize_if_needed(frame1, frame2)

    if f1.ndim == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = f1, f2

    var1 = float(np.var(cv2.Laplacian(gray1, cv2.CV_64F)))
    var2 = float(np.var(cv2.Laplacian(gray2, cv2.CV_64F)))

    # Both completely flat ⇒ identical sharpness.
    if var1 == 0 and var2 == 0:
        return 1.0
    if max(var1, var2) == 0:
        return 0.0

    return float(min(var1, var2) / max(var1, var2))

# --------------------------------------------------------------------------- #


def _aggregate_metric(values: list[float], metric_name: str) -> dict[str, float]:
    """Aggregate frame-level metric values into descriptive statistics.
    
    Args:
        values: List of frame-level metric values
        metric_name: Name of the metric for key generation
        
    Returns:
        Dictionary with mean, std, min, max for the metric
    """
    if not values:
        return {
            metric_name: 0.0,
            f"{metric_name}_std": 0.0,
            f"{metric_name}_min": 0.0,
            f"{metric_name}_max": 0.0,
        }
    
    values_array = np.array(values)
    
    # Handle edge case of single frame
    if len(values) == 1:
        return {
            metric_name: float(values[0]),
            f"{metric_name}_std": 0.0,
            f"{metric_name}_min": float(values[0]),
            f"{metric_name}_max": float(values[0]),
        }
    
    return {
        metric_name: float(np.mean(values_array)),
        f"{metric_name}_std": float(np.std(values_array)),
        f"{metric_name}_min": float(np.min(values_array)),
        f"{metric_name}_max": float(np.max(values_array)),
    }


def _calculate_positional_samples(aligned_pairs: list, metric_func, metric_name: str) -> dict[str, float]:
    """Calculate metrics for first, middle, and last frames to understand positional effects.
    
    This function provides insights into how frame position affects quality metrics,
    which is crucial for determining optimal sampling strategies in production.
    
    Args:
        aligned_pairs: List of (original_frame, compressed_frame) tuples
        metric_func: Function to calculate the metric (e.g., ssim, mse, fsim)
        metric_name: Name of the metric for key generation
        
    Returns:
        Dictionary with positional samples and variance:
        {
            "metric_first": float,      # Metric value for first frame
            "metric_middle": float,     # Metric value for middle frame  
            "metric_last": float,       # Metric value for last frame
            "metric_positional_variance": float  # Variance across positions
        }
    """
    if not aligned_pairs:
        return {
            f"{metric_name}_first": 0.0,
            f"{metric_name}_middle": 0.0,
            f"{metric_name}_last": 0.0,
            f"{metric_name}_positional_variance": 0.0,
        }
    
    n_frames = len(aligned_pairs)
    
    try:
        # Calculate for 3 key positions
        first_val = float(metric_func(*aligned_pairs[0]))
        middle_val = float(metric_func(*aligned_pairs[n_frames // 2]))
        last_val = float(metric_func(*aligned_pairs[-1]))
        
        # Calculate positional variance (how much does position matter?)
        pos_values = [first_val, middle_val, last_val]
        positional_variance = float(np.var(pos_values))
        
        return {
            f"{metric_name}_first": first_val,
            f"{metric_name}_middle": middle_val,
            f"{metric_name}_last": last_val,
            f"{metric_name}_positional_variance": positional_variance,
        }
        
    except Exception as e:
        logger.warning(f"Positional sampling failed for {metric_name}: {e}")
        return {
            f"{metric_name}_first": 0.0,
            f"{metric_name}_middle": 0.0,
            f"{metric_name}_last": 0.0,
            f"{metric_name}_positional_variance": 0.0,
        }


def calculate_comprehensive_metrics(original_path: Path, compressed_path: Path, config: MetricsConfig | None = None) -> dict[str, float]:
    """Calculate comprehensive quality metrics between original and compressed GIFs.

    This is the main function that addresses the frame alignment problem and provides
    multi-metric quality assessment with all available metrics.

    Args:
        original_path: Path to original GIF file
        compressed_path: Path to compressed GIF file
        config: Optional metrics configuration (uses default if None)

    Returns:
        Dictionary with comprehensive metrics including:
        - Traditional metrics: ssim, ms_ssim, psnr, temporal_consistency
        - New metrics: mse, rmse, fsim, gmsd, chist, edge_similarity, texture_similarity, sharpness_similarity
        - Aggregation descriptors: *_std, *_min, *_max for each metric
        - Optional raw values: *_raw for each metric (if config.RAW_METRICS=True)
        - System metrics: render_ms, kilobytes
        - Composite quality score

    Raises:
        IOError: If either GIF file cannot be read
        ValueError: If GIFs are invalid or processing fails
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG

    start_time = time.perf_counter()

    try:
        # Extract frames from both GIFs
        original_result = extract_gif_frames(original_path, config.SSIM_MAX_FRAMES)
        compressed_result = extract_gif_frames(compressed_path, config.SSIM_MAX_FRAMES)

        # Resize frames to common dimensions
        original_frames, compressed_frames = resize_to_common_dimensions(
            original_result.frames, compressed_result.frames
        )

        # Align frames using content-based method (most robust)
        aligned_pairs = align_frames(original_frames, compressed_frames)

        if not aligned_pairs:
            raise ValueError("No frame pairs could be aligned")

        # Calculate all frame-level metrics
        metric_values = {
            'ssim': [],
            'ms_ssim': [],
            'psnr': [],
            'mse': [],
            'rmse': [],
            'fsim': [],
            'gmsd': [],
            'chist': [],
            'edge_similarity': [],
            'texture_similarity': [],
            'sharpness_similarity': [],
        }

        # Store raw (un-normalised) metric values where necessary
        raw_metric_values = {
            'psnr': [],  # PSNR is normalised for main reporting; keep raw values separately
        }

        for orig_frame, comp_frame in aligned_pairs:
            # Traditional SSIM calculation
            try:
                if len(orig_frame.shape) == 3:
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                    comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
                else:
                    orig_gray = orig_frame
                    comp_gray = comp_frame

                frame_ssim = ssim(orig_gray, comp_gray, data_range=255.0)
                metric_values['ssim'].append(max(0.0, min(1.0, frame_ssim)))
            except Exception as e:
                logger.warning(f"SSIM calculation failed for frame: {e}")
                metric_values['ssim'].append(0.0)

            # MS-SSIM calculation
            try:
                frame_ms_ssim = calculate_ms_ssim(orig_frame, comp_frame)
                metric_values['ms_ssim'].append(frame_ms_ssim)
            except Exception as e:
                logger.warning(f"MS-SSIM calculation failed for frame: {e}")
                metric_values['ms_ssim'].append(0.0)

            # PSNR calculation
            try:
                frame_psnr = psnr(orig_frame, comp_frame, data_range=255.0)
                # Keep un-scaled PSNR for optional raw metrics output
                raw_metric_values['psnr'].append(frame_psnr)

                # Normalize PSNR using configurable upper bound
                normalized_psnr = min(frame_psnr / float(config.PSNR_MAX_DB), 1.0)
                metric_values['psnr'].append(max(0.0, normalized_psnr))
            except Exception as e:
                logger.warning(f"PSNR calculation failed for frame: {e}")
                metric_values['psnr'].append(0.0)
                raw_metric_values['psnr'].append(0.0)

            # New metrics - MSE and RMSE
            try:
                frame_mse = mse(orig_frame, comp_frame)
                metric_values['mse'].append(frame_mse)
                
                frame_rmse = rmse(orig_frame, comp_frame)
                metric_values['rmse'].append(frame_rmse)
            except Exception as e:
                logger.warning(f"MSE/RMSE calculation failed for frame: {e}")
                metric_values['mse'].append(0.0)
                metric_values['rmse'].append(0.0)

            # FSIM calculation
            try:
                frame_fsim = fsim(orig_frame, comp_frame)
                metric_values['fsim'].append(frame_fsim)
            except Exception as e:
                logger.warning(f"FSIM calculation failed for frame: {e}")
                metric_values['fsim'].append(0.0)

            # GMSD calculation
            try:
                frame_gmsd = gmsd(orig_frame, comp_frame)
                metric_values['gmsd'].append(frame_gmsd)
            except Exception as e:
                logger.warning(f"GMSD calculation failed for frame: {e}")
                metric_values['gmsd'].append(0.0)

            # Color histogram correlation
            try:
                frame_chist = chist(orig_frame, comp_frame)
                metric_values['chist'].append(frame_chist)
            except Exception as e:
                logger.warning(f"Color histogram calculation failed for frame: {e}")
                metric_values['chist'].append(0.0)

            # Edge similarity
            try:
                frame_edge = edge_similarity(
                    orig_frame, comp_frame,
                    config.EDGE_CANNY_THRESHOLD1,
                    config.EDGE_CANNY_THRESHOLD2)
                metric_values['edge_similarity'].append(frame_edge)
            except Exception as e:
                logger.warning(f"Edge similarity calculation failed for frame: {e}")
                metric_values['edge_similarity'].append(0.0)

            # Texture similarity
            try:
                frame_texture = texture_similarity(orig_frame, comp_frame)
                metric_values['texture_similarity'].append(frame_texture)
            except Exception as e:
                logger.warning(f"Texture similarity calculation failed for frame: {e}")
                metric_values['texture_similarity'].append(0.0)

            # Sharpness similarity
            try:
                frame_sharpness = sharpness_similarity(orig_frame, comp_frame)
                metric_values['sharpness_similarity'].append(frame_sharpness)
            except Exception as e:
                logger.warning(f"Sharpness similarity calculation failed for frame: {e}")
                metric_values['sharpness_similarity'].append(0.0)

        # Calculate temporal consistency for original and compressed GIFs
        temporal_pre = 0.0
        temporal_post = 0.0
        if config.TEMPORAL_CONSISTENCY_ENABLED:
            temporal_pre = calculate_temporal_consistency(original_frames)
            temporal_post = calculate_temporal_consistency(compressed_frames)

        temporal_delta = abs(temporal_post - temporal_pre)

        # Aggregate all metrics with descriptive statistics
        result = {}
        
        # Add aggregated metrics
        for metric_name, values in metric_values.items():
            result.update(_aggregate_metric(values, metric_name))
        
        # Add temporal consistency (single value, not frame-level)
        # Keep legacy key pointing to *post*-compression value for backward compatibility
        result['temporal_consistency'] = float(temporal_post)
        result['temporal_consistency_std'] = 0.0
        result['temporal_consistency_min'] = float(temporal_post)
        result['temporal_consistency_max'] = float(temporal_post)

        # New keys: pre, post (explicit) and delta
        result['temporal_consistency_pre'] = float(temporal_pre)
        result['temporal_consistency_post'] = float(temporal_post)
        result['temporal_consistency_delta'] = float(temporal_delta)

        # Calculate composite quality using traditional metrics only
        composite_quality = (
            config.SSIM_WEIGHT * result['ssim'] +
            config.MS_SSIM_WEIGHT * result['ms_ssim'] +
            config.PSNR_WEIGHT * result['psnr'] +
            config.TEMPORAL_WEIGHT * result['temporal_consistency']
        )
        result['composite_quality'] = float(composite_quality)

        # Add system metrics
        result['kilobytes'] = float(calculate_file_size_kb(compressed_path))
        
        # Calculate processing time
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        result['render_ms'] = min(int(elapsed_seconds * 1000), 86400000)

        # Add positional sampling if enabled
        if config.ENABLE_POSITIONAL_SAMPLING:
            # Map metric names to their functions
            metric_functions = {
                'ssim': lambda f1, f2: ssim(
                    cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY) if len(f1.shape) == 3 else f1,
                    cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY) if len(f2.shape) == 3 else f2,
                    data_range=255.0
                ),
                'mse': mse,
                'rmse': rmse,
                'fsim': fsim,
                'gmsd': gmsd,
                'chist': chist,
                'edge_similarity': lambda f1, f2: edge_similarity(
                    f1, f2,
                    config.EDGE_CANNY_THRESHOLD1,
                    config.EDGE_CANNY_THRESHOLD2),
                'texture_similarity': texture_similarity,
                'sharpness_similarity': sharpness_similarity,
            }
            
            # Calculate positional samples for configured metrics
            for metric_name in config.POSITIONAL_METRICS:
                if metric_name in metric_functions:
                    try:
                        positional_data = _calculate_positional_samples(
                            aligned_pairs, 
                            metric_functions[metric_name], 
                            metric_name
                        )
                        result.update(positional_data)
                    except Exception as e:
                        logger.warning(f"Failed to calculate positional samples for {metric_name}: {e}")

        # Add raw metrics if requested
        if config.RAW_METRICS:
            # Metrics already reported in raw (un-scaled) form – directly copy.
            raw_equivalent_metrics = [
                'ssim', 'ms_ssim', 'mse', 'rmse', 'fsim', 'gmsd',
                'chist', 'edge_similarity', 'texture_similarity',
                'sharpness_similarity', 'temporal_consistency',
                'temporal_consistency_pre', 'temporal_consistency_post', 'temporal_consistency_delta'
            ]

            for metric_name in raw_equivalent_metrics:
                result[f"{metric_name}_raw"] = result[metric_name]

            # Handle PSNR separately: use un-scaled mean value
            if raw_metric_values['psnr']:
                result['psnr_raw'] = float(np.mean(raw_metric_values['psnr']))
            else:
                result['psnr_raw'] = 0.0

            # Raw copies for temporal consistency variants
            result['temporal_consistency_pre_raw'] = result['temporal_consistency_pre']
            result['temporal_consistency_post_raw'] = result['temporal_consistency_post']
            result['temporal_consistency_delta_raw'] = result['temporal_consistency_delta']

        return result

    except Exception as e:
        logger.error(f"Failed to calculate comprehensive metrics: {e}")
        raise ValueError(f"Metrics calculation failed: {e}") from e
