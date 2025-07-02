"""Quality metrics and comparison functionality for GIF analysis."""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time
import logging
from dataclasses import dataclass

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

from .config import DEFAULT_METRICS_CONFIG, MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractResult:
    """Result of frame extraction from a GIF."""
    frames: List[np.ndarray]
    frame_count: int
    dimensions: Tuple[int, int]  # (width, height)
    duration_ms: int


def extract_gif_frames(gif_path: Path, max_frames: Optional[int] = None) -> FrameExtractResult:
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
            frames_to_extract = min(total_frames, max_frames) if max_frames else total_frames
            
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
        raise IOError(f"Failed to extract frames from {gif_path}: {e}")


def resize_to_common_dimensions(frames1: List[np.ndarray], frames2: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Resize frames to common dimensions (smallest common size).
    
    Args:
        frames1: First set of frames
        frames2: Second set of frames
        
    Returns:
        Tuple of (resized_frames1, resized_frames2)
    """
    if not frames1 or not frames2:
        return frames1, frames2
    
    # Get dimensions
    h1, w1 = frames1[0].shape[:2]
    h2, w2 = frames2[0].shape[:2]
    
    # Use smallest common dimensions
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    # Resize if necessary
    resized_frames1 = []
    for frame in frames1:
        if frame.shape[:2] != (target_h, target_w):
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized_frames1.append(resized)
        else:
            resized_frames1.append(frame)
    
    resized_frames2 = []
    for frame in frames2:
        if frame.shape[:2] != (target_h, target_w):
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized_frames2.append(resized)
        else:
            resized_frames2.append(frame)
    
    return resized_frames1, resized_frames2





def calculate_frame_mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate mean squared error between two frames."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    return float(np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2))


def align_frames_content_based(original_frames: List[np.ndarray], compressed_frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
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



def align_frames(original_frames: List[np.ndarray], compressed_frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def calculate_temporal_consistency(frames: List[np.ndarray]) -> float:
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
        diff = np.mean(np.abs(frame1 - frame2))
        frame_differences.append(diff)
    
    if not frame_differences:
        return 1.0
    
    # Calculate consistency as inverse of variance in frame differences
    # More consistent animations have lower variance in frame-to-frame changes
    mean_diff = np.mean(frame_differences)
    variance_diff = np.var(frame_differences)
    
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
    return max(0.0, min(1.0, consistency))


def calculate_comprehensive_metrics(original_path: Path, compressed_path: Path, config: Optional[MetricsConfig] = None) -> Dict[str, float]:
    """Calculate comprehensive quality metrics between original and compressed GIFs.
    
    This is the main function that addresses the frame alignment problem and provides
    multi-metric quality assessment.
    
    Args:
        original_path: Path to original GIF file
        compressed_path: Path to compressed GIF file
        config: Optional metrics configuration (uses default if None)
        
    Returns:
        Dictionary with comprehensive metrics:
        {
            "ssim": float,                    # Traditional SSIM (0.0-1.0)
            "ms_ssim": float,                 # Multi-scale SSIM (0.0-1.0) 
            "psnr": float,                    # PSNR normalized (0.0-1.0)
            "temporal_consistency": float,     # Animation smoothness (0.0-1.0)
            "composite_quality": float,        # Weighted combination (0.0-1.0)
            "render_ms": int,                 # Processing time in milliseconds
            "kilobytes": float                # File size in KB
        }
        
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
        
        # Calculate individual metrics
        ssim_values = []
        ms_ssim_values = []
        psnr_values = []
        
        for orig_frame, comp_frame in aligned_pairs:
            # SSIM calculation
            try:
                if len(orig_frame.shape) == 3:
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                    comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
                else:
                    orig_gray = orig_frame
                    comp_gray = comp_frame
                
                frame_ssim = ssim(orig_gray, comp_gray, data_range=255.0)
                ssim_values.append(max(0.0, min(1.0, frame_ssim)))
            except Exception as e:
                logger.warning(f"SSIM calculation failed for frame: {e}")
                ssim_values.append(0.0)
            
            # MS-SSIM calculation
            try:
                frame_ms_ssim = calculate_ms_ssim(orig_frame, comp_frame)
                ms_ssim_values.append(frame_ms_ssim)
            except Exception as e:
                logger.warning(f"MS-SSIM calculation failed for frame: {e}")
                ms_ssim_values.append(0.0)
            
            # PSNR calculation
            try:
                frame_psnr = psnr(orig_frame, comp_frame, data_range=255.0)
                # Normalize PSNR to 0-1 range (assume max useful PSNR is 50dB)
                normalized_psnr = min(frame_psnr / 50.0, 1.0)
                psnr_values.append(max(0.0, normalized_psnr))
            except Exception as e:
                logger.warning(f"PSNR calculation failed for frame: {e}")
                psnr_values.append(0.0)
        
        # Aggregate metrics
        avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
        avg_ms_ssim = np.mean(ms_ssim_values) if ms_ssim_values else 0.0
        avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
        
        # Calculate temporal consistency
        temporal_consistency = 0.0
        if config.TEMPORAL_CONSISTENCY_ENABLED:
            temporal_consistency = calculate_temporal_consistency(compressed_frames)
        
        # Calculate composite quality using exact formula from instructions
        composite_quality = (
            config.SSIM_WEIGHT * avg_ssim +
            config.MS_SSIM_WEIGHT * avg_ms_ssim +
            config.PSNR_WEIGHT * avg_psnr +
            config.TEMPORAL_WEIGHT * temporal_consistency
        )
        
        # Calculate file size
        file_size_kb = calculate_file_size_kb(compressed_path)
        
        # Calculate processing time
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        # Cap at reasonable maximum to prevent overflow (24 hours = 86400000 ms)
        render_ms = min(int(elapsed_seconds * 1000), 86400000)
        
        return {
            "ssim": float(avg_ssim),
            "ms_ssim": float(avg_ms_ssim),
            "psnr": float(avg_psnr),
            "temporal_consistency": float(temporal_consistency),
            "composite_quality": float(composite_quality),
            "render_ms": render_ms,
            "kilobytes": float(file_size_kb)
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate comprehensive metrics: {e}")
        raise ValueError(f"Metrics calculation failed: {e}")


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
        raise IOError(f"Cannot access file {file_path}: {e}")


def measure_render_time(func, *args, **kwargs) -> Tuple[Any, int]:
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


def compare_gif_frames(gif1_path: Path, gif2_path: Path) -> Dict[str, Any]:
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