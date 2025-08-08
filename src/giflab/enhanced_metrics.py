"""Enhanced quality metrics calculation with comprehensive 11-metric composite quality.

This module provides enhanced composite quality calculation using all 11 available
quality metrics instead of the traditional 4-metric approach. It also includes
the user-requested efficiency metric calculation.
"""

import numpy as np
from typing import Dict, Any, Optional
from .config import MetricsConfig, DEFAULT_METRICS_CONFIG


def normalize_metric(value: float, metric_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a metric value to 0-1 range with appropriate direction.
    
    Args:
        value: Raw metric value
        metric_name: Name of the metric (determines if higher/lower is better)
        min_val: Minimum possible value for normalization
        max_val: Maximum possible value for normalization
        
    Returns:
        Normalized value between 0 and 1
    """
    # Handle special cases with known ranges
    if metric_name == 'mse_mean':
        # MSE can be very large, use log normalization
        if value <= 0:
            return 1.0  # Perfect score for zero MSE
        # Normalize MSE using log scale, then invert (lower MSE is better)
        normalized = 1.0 / (1.0 + np.log10(max(value, 1.0)))
        return max(0.0, min(1.0, normalized))
    
    elif metric_name == 'gmsd_mean':
        # GMSD: lower is better, typical range 0-0.5
        max_val = 0.5
        normalized = 1.0 - min(value, max_val) / max_val
        return max(0.0, min(1.0, normalized))
    
    elif metric_name == 'ms_ssim_mean':
        # MS-SSIM can go negative, handle appropriately
        if value < 0:
            return 0.0  # Negative MS-SSIM indicates very poor quality
        return max(0.0, min(1.0, value))
    
    else:
        # Standard 0-1 normalization for metrics where higher is better
        if min_val >= max_val:
            return 1.0  # Avoid division by zero
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))


def calculate_enhanced_composite_quality(
    metrics: Dict[str, float], 
    config: Optional[MetricsConfig] = None
) -> float:
    """Calculate enhanced composite quality using all 11 available metrics.
    
    This function implements the comprehensive approach (Approach B) using
    research-based weights across all quality dimensions.
    
    Args:
        metrics: Dictionary containing all metric values
        config: Metrics configuration (uses default if None)
        
    Returns:
        Enhanced composite quality score (0-1)
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG
    
    if not config.USE_ENHANCED_COMPOSITE_QUALITY:
        # Fall back to legacy 4-metric calculation
        return calculate_legacy_composite_quality(metrics, config)
    
    composite_quality = 0.0
    total_weight = 0.0  # Track actual weights used (for missing metrics)
    
    # DEBUG: Track calculation steps
    debug_steps = []
    
    # Core structural similarity metrics (40% total)
    if 'ssim_mean' in metrics:
        raw_value = metrics['ssim_mean']
        normalized = normalize_metric(raw_value, 'ssim_mean')
        contribution = config.ENHANCED_SSIM_WEIGHT * normalized
        composite_quality += contribution
        total_weight += config.ENHANCED_SSIM_WEIGHT
        debug_steps.append(f"SSIM: {raw_value:.3f} → {normalized:.3f} × {config.ENHANCED_SSIM_WEIGHT} = {contribution:.3f}")
    
    if 'ms_ssim_mean' in metrics:
        raw_value = metrics['ms_ssim_mean']
        normalized = normalize_metric(raw_value, 'ms_ssim_mean')
        contribution = config.ENHANCED_MS_SSIM_WEIGHT * normalized
        composite_quality += contribution
        total_weight += config.ENHANCED_MS_SSIM_WEIGHT
        debug_steps.append(f"MS-SSIM: {raw_value:.3f} → {normalized:.3f} × {config.ENHANCED_MS_SSIM_WEIGHT} = {contribution:.3f}")
    
    # Signal quality metrics (25% total)
    if 'psnr_mean' in metrics:
        normalized = normalize_metric(metrics['psnr_mean'], 'psnr_mean')
        composite_quality += config.ENHANCED_PSNR_WEIGHT * normalized
        total_weight += config.ENHANCED_PSNR_WEIGHT
    
    if 'mse_mean' in metrics:
        normalized = normalize_metric(metrics['mse_mean'], 'mse_mean')
        composite_quality += config.ENHANCED_MSE_WEIGHT * normalized
        total_weight += config.ENHANCED_MSE_WEIGHT
    
    # Advanced structural metrics (20% total)
    if 'fsim_mean' in metrics:
        normalized = normalize_metric(metrics['fsim_mean'], 'fsim_mean')
        composite_quality += config.ENHANCED_FSIM_WEIGHT * normalized
        total_weight += config.ENHANCED_FSIM_WEIGHT
    
    if 'edge_similarity_mean' in metrics:
        normalized = normalize_metric(metrics['edge_similarity_mean'], 'edge_similarity_mean')
        composite_quality += config.ENHANCED_EDGE_WEIGHT * normalized
        total_weight += config.ENHANCED_EDGE_WEIGHT
    
    if 'gmsd_mean' in metrics:
        normalized = normalize_metric(metrics['gmsd_mean'], 'gmsd_mean')
        composite_quality += config.ENHANCED_GMSD_WEIGHT * normalized
        total_weight += config.ENHANCED_GMSD_WEIGHT
    
    # Perceptual quality metrics (10% total)
    if 'chist_mean' in metrics:
        normalized = normalize_metric(metrics['chist_mean'], 'chist_mean')
        composite_quality += config.ENHANCED_CHIST_WEIGHT * normalized
        total_weight += config.ENHANCED_CHIST_WEIGHT
    
    if 'sharpness_similarity_mean' in metrics:
        normalized = normalize_metric(metrics['sharpness_similarity_mean'], 'sharpness_similarity_mean')
        composite_quality += config.ENHANCED_SHARPNESS_WEIGHT * normalized
        total_weight += config.ENHANCED_SHARPNESS_WEIGHT
    
    if 'texture_similarity_mean' in metrics:
        normalized = normalize_metric(metrics['texture_similarity_mean'], 'texture_similarity_mean')
        composite_quality += config.ENHANCED_TEXTURE_WEIGHT * normalized
        total_weight += config.ENHANCED_TEXTURE_WEIGHT
    
    # Temporal consistency (5% total)
    if 'temporal_consistency' in metrics:
        normalized = normalize_metric(metrics['temporal_consistency'], 'temporal_consistency')
        composite_quality += config.ENHANCED_TEMPORAL_WEIGHT * normalized
        total_weight += config.ENHANCED_TEMPORAL_WEIGHT
    
    # Normalize by actual weights used (handles missing metrics gracefully)
    raw_composite = composite_quality
    if total_weight > 0:
        composite_quality = composite_quality / total_weight
    
    # DEBUG: Log final calculation for debugging
    final_result = max(0.0, min(1.0, composite_quality))
    if 'ssim_mean' in metrics and metrics.get('ssim_mean', 0) > 0.5:  # Only log for significant cases
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Enhanced composite quality calculation:")
        for step in debug_steps[:3]:  # Log first few steps
            logger.info(f"  {step}")
        logger.info(f"  Raw total: {raw_composite:.3f}, Total weight: {total_weight:.3f}")
        logger.info(f"  Normalized: {composite_quality:.3f}, Final (clamped): {final_result:.3f}")
    
    return final_result


def calculate_legacy_composite_quality(
    metrics: Dict[str, float], 
    config: Optional[MetricsConfig] = None
) -> float:
    """Calculate legacy 4-metric composite quality for backward compatibility.
    
    Args:
        metrics: Dictionary containing metric values
        config: Metrics configuration (uses default if None)
        
    Returns:
        Legacy composite quality score (0-1)
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG
    
    # Use existing logic from experimental runner
    composite_quality = (
        config.SSIM_WEIGHT * metrics.get('ssim_mean', 0.0) +
        config.MS_SSIM_WEIGHT * metrics.get('ms_ssim_mean', 0.0) +
        config.PSNR_WEIGHT * metrics.get('psnr_mean', 0.0) +
        config.TEMPORAL_WEIGHT * metrics.get('temporal_consistency', 0.0)
    )
    
    return max(0.0, min(1.0, composite_quality))


def calculate_efficiency_metric(
    compression_ratio: float, 
    composite_quality: float
) -> float:
    """Calculate the user-requested efficiency metric.
    
    Efficiency = compression_ratio × composite_quality
    
    This balances compression performance with quality retention.
    Higher values indicate better overall efficiency.
    
    Args:
        compression_ratio: Compression ratio (original_size / compressed_size)
        composite_quality: Composite quality score (0-1)
        
    Returns:
        Efficiency score (higher is better)
    """
    if compression_ratio <= 0 or composite_quality < 0:
        return 0.0
    
    return compression_ratio * composite_quality


def process_metrics_with_enhanced_quality(
    result: Dict[str, Any],
    config: Optional[MetricsConfig] = None
) -> Dict[str, Any]:
    """Process a metrics result dictionary to add enhanced quality metrics.
    
    Args:
        result: Dictionary containing raw metric values
        config: Metrics configuration (uses default if None)
        
    Returns:
        Enhanced result dictionary with new metrics added
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG
    
    # Calculate enhanced composite quality
    enhanced_quality = calculate_enhanced_composite_quality(result, config)
    result['enhanced_composite_quality'] = enhanced_quality
    
    # Calculate efficiency metric if compression data is available
    if 'compression_ratio' in result:
        # Use enhanced composite quality for efficiency calculation
        efficiency = calculate_efficiency_metric(
            result['compression_ratio'], 
            enhanced_quality
        )
        result['efficiency'] = efficiency
    
    # Also add legacy composite quality for comparison (if not already present)
    if 'composite_quality' not in result:
        legacy_quality = calculate_legacy_composite_quality(result, config)
        result['composite_quality'] = legacy_quality
    
    return result


def get_enhanced_weights_info(config: Optional[MetricsConfig] = None) -> Dict[str, Any]:
    """Get information about the enhanced weighting scheme.
    
    Args:
        config: Metrics configuration (uses default if None)
        
    Returns:
        Dictionary with weight distribution information
    """
    if config is None:
        config = DEFAULT_METRICS_CONFIG
    
    weights_info = {
        'core_structural': {
            'ssim_mean': config.ENHANCED_SSIM_WEIGHT,
            'ms_ssim_mean': config.ENHANCED_MS_SSIM_WEIGHT,
            'total': config.ENHANCED_SSIM_WEIGHT + config.ENHANCED_MS_SSIM_WEIGHT
        },
        'signal_quality': {
            'psnr_mean': config.ENHANCED_PSNR_WEIGHT,
            'mse_mean': config.ENHANCED_MSE_WEIGHT,
            'total': config.ENHANCED_PSNR_WEIGHT + config.ENHANCED_MSE_WEIGHT
        },
        'advanced_structural': {
            'fsim_mean': config.ENHANCED_FSIM_WEIGHT,
            'edge_similarity_mean': config.ENHANCED_EDGE_WEIGHT,
            'gmsd_mean': config.ENHANCED_GMSD_WEIGHT,
            'total': config.ENHANCED_FSIM_WEIGHT + config.ENHANCED_EDGE_WEIGHT + config.ENHANCED_GMSD_WEIGHT
        },
        'perceptual_quality': {
            'chist_mean': config.ENHANCED_CHIST_WEIGHT,
            'sharpness_similarity_mean': config.ENHANCED_SHARPNESS_WEIGHT,
            'texture_similarity_mean': config.ENHANCED_TEXTURE_WEIGHT,
            'total': config.ENHANCED_CHIST_WEIGHT + config.ENHANCED_SHARPNESS_WEIGHT + config.ENHANCED_TEXTURE_WEIGHT
        },
        'temporal_consistency': {
            'temporal_consistency': config.ENHANCED_TEMPORAL_WEIGHT,
            'total': config.ENHANCED_TEMPORAL_WEIGHT
        },
        'grand_total': (
            config.ENHANCED_SSIM_WEIGHT + config.ENHANCED_MS_SSIM_WEIGHT +
            config.ENHANCED_PSNR_WEIGHT + config.ENHANCED_MSE_WEIGHT +
            config.ENHANCED_FSIM_WEIGHT + config.ENHANCED_EDGE_WEIGHT + 
            config.ENHANCED_GMSD_WEIGHT + config.ENHANCED_CHIST_WEIGHT +
            config.ENHANCED_SHARPNESS_WEIGHT + config.ENHANCED_TEXTURE_WEIGHT +
            config.ENHANCED_TEMPORAL_WEIGHT
        )
    }
    
    return weights_info