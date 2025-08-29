"""
Core validation system for GifLab compression pipeline.

Provides comprehensive validation of compression results with content-type awareness,
configurable thresholds, and multi-metric validation combinations.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..meta import GifMetadata
from .data_structures import (
    ValidationResult, ValidationStatus, ValidationIssue, ValidationWarning,
    ValidationMetrics, ValidationConfig
)
from .config import load_validation_config

logger = logging.getLogger(__name__)


class ValidationChecker:
    """
    Pipeline-integrated validation system for automated compression issue detection.
    
    Validates compression results against configurable thresholds with content-type
    awareness and multi-metric combination detection.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ValidationChecker with configuration.
        
        Args:
            config_path: Optional path to validation configuration file.
                        If None, uses default configuration with content-type adjustments.
        """
        self.config = load_validation_config(config_path)
        self.validation_results: Dict[str, ValidationResult] = {}
        
        logger.info("ValidationChecker initialized with configuration")
        logger.debug(f"Base quality floor: {self.config.minimum_quality_floor}")
        logger.debug(f"Content-type adjustments: {len(self.config.content_type_adjustments)} types")
    
    def validate_compression_result(
        self,
        original_metadata: GifMetadata,
        compression_metrics: Dict[str, Any], 
        pipeline_id: str,
        gif_name: str,
        content_type: str = "unknown",
        pipeline_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Main validation method called during pipeline execution.
        
        Args:
            original_metadata: GifMetadata from original GIF
            compression_metrics: Dict from calculate_comprehensive_metrics()
            pipeline_id: Identifier of the compression pipeline used
            gif_name: Name of the GIF being validated
            content_type: Content type classification (e.g., 'animation_heavy', 'smooth_gradient')
            pipeline_type: Type of pipeline (e.g., 'frame_reduction', 'color_reduction')
            
        Returns:
            ValidationResult with comprehensive validation status
        """
        start_time = time.perf_counter()
        
        # Get effective thresholds for this content type and pipeline
        effective_thresholds = self.config.get_effective_thresholds(content_type, pipeline_type)
        
        logger.debug(f"Validating {gif_name} ({content_type}) with pipeline {pipeline_id}")
        
        # Initialize validation result
        result = ValidationResult(
            pipeline_id=pipeline_id,
            gif_name=gif_name,
            content_type=content_type,
            status=ValidationStatus.PASS,
            effective_thresholds=effective_thresholds
        )
        
        # Extract and validate metrics
        try:
            # Build validation metrics summary
            result.metrics = self._extract_validation_metrics(original_metadata, compression_metrics)
            
            # Perform individual validation checks
            self._validate_frame_reduction(result, original_metadata, compression_metrics, effective_thresholds)
            self._validate_fps_consistency(result, original_metadata, compression_metrics, effective_thresholds)
            self._validate_quality_thresholds(result, compression_metrics, effective_thresholds)
            self._validate_efficiency_thresholds(result, compression_metrics, effective_thresholds)
            self._validate_disposal_artifacts(result, compression_metrics, effective_thresholds)
            self._validate_temporal_consistency(result, compression_metrics, effective_thresholds)
            
            # Perform multi-metric combination validation
            self._validate_multi_metric_combinations(result, original_metadata, compression_metrics)
            
            # Determine final validation status
            if result.issues:
                # Check for artifact-specific issues
                artifact_categories = {'disposal_artifacts', 'temporal_corruption', 'animation_corruption'}
                has_artifacts = any(issue.category in artifact_categories for issue in result.issues)
                result.status = ValidationStatus.ARTIFACT if has_artifacts else ValidationStatus.ERROR
            elif result.warnings:
                result.status = ValidationStatus.WARNING
            else:
                result.status = ValidationStatus.PASS
            
        except Exception as e:
            logger.error(f"Validation failed for {gif_name}: {e}")
            result.status = ValidationStatus.UNKNOWN
            result.issues.append(ValidationIssue(
                category="validation_error",
                message=f"Validation system error: {str(e)}",
                severity="CRITICAL"
            ))
        
        # Record timing and store result
        end_time = time.perf_counter()
        result.validation_time_ms = int((end_time - start_time) * 1000)
        
        self.validation_results[f"{gif_name}_{pipeline_id}"] = result
        
        logger.debug(f"Validation completed for {gif_name}: {result.status.value} "
                    f"({len(result.issues)} issues, {len(result.warnings)} warnings)")
        
        return result
    
    def _extract_validation_metrics(
        self,
        original_metadata: GifMetadata,
        compression_metrics: Dict[str, Any]
    ) -> ValidationMetrics:
        """Extract key metrics for validation from metadata and compression results."""
        
        # Calculate FPS deviation
        original_fps = original_metadata.orig_fps
        compressed_fps = compression_metrics.get('orig_fps', original_fps)  # Fallback to original if not available
        
        fps_deviation = None
        if original_fps > 0:
            fps_deviation = abs(compressed_fps - original_fps) / original_fps * 100
        
        # Calculate frame reduction ratio
        original_frames = original_metadata.orig_frames
        compressed_frames = compression_metrics.get('compressed_frame_count', original_frames)
        frame_reduction_ratio = compressed_frames / original_frames if original_frames > 0 else None
        
        return ValidationMetrics(
            # Frame and timing
            original_frame_count=original_frames,
            compressed_frame_count=compressed_frames,
            original_fps=original_fps,
            compressed_fps=compressed_fps,
            frame_reduction_ratio=frame_reduction_ratio,
            fps_deviation_percent=fps_deviation,
            
            # Quality
            composite_quality=compression_metrics.get('composite_quality'),
            efficiency=compression_metrics.get('efficiency'),
            compression_ratio=compression_metrics.get('compression_ratio'),
            
            # Temporal and artifacts
            temporal_consistency_pre=compression_metrics.get('temporal_consistency_pre'),
            temporal_consistency_post=compression_metrics.get('temporal_consistency_post'),
            disposal_artifacts_pre=compression_metrics.get('disposal_artifacts_pre'),
            disposal_artifacts_post=compression_metrics.get('disposal_artifacts_post'),
            disposal_artifacts_delta=compression_metrics.get('disposal_artifacts_delta'),
            
            # Size
            original_size_kb=original_metadata.orig_kilobytes,
            compressed_size_kb=compression_metrics.get('kilobytes')
        )
    
    def _validate_frame_reduction(
        self,
        result: ValidationResult,
        original_metadata: GifMetadata,
        compression_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> None:
        """Validate frame count reduction accuracy."""
        
        original_frames = original_metadata.orig_frames
        compressed_frames = compression_metrics.get('compressed_frame_count', original_frames)
        
        if not original_frames or not compressed_frames:
            result.warnings.append(ValidationWarning(
                category="frame_count",
                message="Frame count data unavailable for validation"
            ))
            return
        
        actual_ratio = compressed_frames / original_frames
        tolerance = thresholds['frame_reduction_tolerance']
        
        # Check if we have an expected ratio from pipeline parameters
        expected_ratio = compression_metrics.get('expected_frame_ratio')
        
        if expected_ratio and abs(actual_ratio - expected_ratio) > tolerance:
            expected_frames = int(original_frames * expected_ratio)
            result.issues.append(ValidationIssue(
                category="frame_reduction",
                message=f"Frame count mismatch: Expected {expected_frames} frames "
                       f"({expected_ratio*100:.0f}%), got {compressed_frames} ({actual_ratio*100:.1f}%)",
                expected_value=expected_frames,
                actual_value=compressed_frames,
                threshold=tolerance
            ))
        
        # Warn about significant frame reduction without explicit expectation
        elif expected_ratio is None and actual_ratio < 0.5:
            result.warnings.append(ValidationWarning(
                category="frame_reduction",
                message=f"Significant frame reduction detected: {original_frames} → {compressed_frames} "
                       f"({actual_ratio*100:.1f}% retained)",
                actual_value=actual_ratio,
                recommendation="Verify frame reduction is intentional"
            ))
    
    def _validate_fps_consistency(
        self,
        result: ValidationResult,
        original_metadata: GifMetadata,
        compression_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> None:
        """Validate FPS consistency between original and compressed GIFs."""
        
        original_fps = original_metadata.orig_fps
        compressed_fps = compression_metrics.get('orig_fps', original_fps)
        
        if not original_fps or not compressed_fps or original_fps <= 0:
            result.warnings.append(ValidationWarning(
                category="fps_consistency",
                message="FPS data unavailable for validation"
            ))
            return
        
        deviation = abs(compressed_fps - original_fps) / original_fps
        error_threshold = thresholds['fps_deviation_tolerance']
        warning_threshold = thresholds['fps_warning_threshold']
        
        if deviation > error_threshold:
            percent_change = (compressed_fps - original_fps) / original_fps * 100
            result.issues.append(ValidationIssue(
                category="fps_consistency",
                message=f"FPS deviation too large: {original_fps:.1f}fps → {compressed_fps:.1f}fps "
                       f"({percent_change:+.1f}% change)",
                expected_value=original_fps,
                actual_value=compressed_fps,
                threshold=error_threshold
            ))
        elif deviation > warning_threshold:
            percent_change = (compressed_fps - original_fps) / original_fps * 100
            result.warnings.append(ValidationWarning(
                category="fps_consistency",
                message=f"FPS change detected: {original_fps:.1f}fps → {compressed_fps:.1f}fps "
                       f"({percent_change:+.1f}% change)",
                expected_value=original_fps,
                actual_value=compressed_fps,
                threshold=warning_threshold,
                recommendation="Verify FPS change is intentional"
            ))
    
    def _validate_quality_thresholds(
        self,
        result: ValidationResult,
        compression_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> None:
        """Validate composite quality against thresholds."""
        
        composite_quality = compression_metrics.get('composite_quality')
        
        if composite_quality is None:
            result.warnings.append(ValidationWarning(
                category="quality_threshold",
                message="Quality data unavailable for validation"
            ))
            return
        
        minimum_floor = thresholds['minimum_quality_floor']
        warning_threshold = thresholds['quality_warning_threshold']
        
        if composite_quality < minimum_floor:
            result.issues.append(ValidationIssue(
                category="quality_threshold",
                message=f"Quality below minimum threshold: {composite_quality:.3f} < {minimum_floor:.3f}",
                expected_value=minimum_floor,
                actual_value=composite_quality,
                threshold=minimum_floor
            ))
        elif composite_quality < warning_threshold:
            result.warnings.append(ValidationWarning(
                category="quality_threshold",
                message=f"Quality below warning threshold: {composite_quality:.3f} < {warning_threshold:.3f}",
                expected_value=warning_threshold,
                actual_value=composite_quality,
                threshold=warning_threshold,
                recommendation="Consider adjusting compression parameters for better quality"
            ))
    
    def _validate_efficiency_thresholds(
        self,
        result: ValidationResult,
        compression_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> None:
        """Validate efficiency metrics against thresholds."""
        
        efficiency = compression_metrics.get('efficiency')
        
        if efficiency is None:
            result.warnings.append(ValidationWarning(
                category="efficiency_threshold",
                message="Efficiency data unavailable for validation"
            ))
            return
        
        # Check for unreasonably high efficiency (may indicate calculation errors)
        max_reasonable_efficiency = 0.95
        if efficiency > max_reasonable_efficiency:
            result.warnings.append(ValidationWarning(
                category="efficiency_threshold", 
                message=f"Efficiency unexpectedly high: {efficiency:.3f} > {max_reasonable_efficiency:.3f}",
                expected_value=max_reasonable_efficiency,
                actual_value=efficiency,
                threshold=max_reasonable_efficiency,
                recommendation="Verify quality metrics and compression calculations are correct"
            ))
        
        # Note: No minimum efficiency validation - efficiency is purely informational
    
    def _validate_disposal_artifacts(
        self,
        result: ValidationResult,
        compression_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> None:
        """Validate disposal artifact detection scores."""
        
        disposal_pre = compression_metrics.get('disposal_artifacts_pre')
        disposal_post = compression_metrics.get('disposal_artifacts_post')
        disposal_delta = compression_metrics.get('disposal_artifacts_delta')
        
        if not disposal_pre or not disposal_post:
            result.warnings.append(ValidationWarning(
                category="disposal_artifacts",
                message="Disposal artifact data unavailable for validation"
            ))
            return
        
        artifact_threshold = thresholds['disposal_artifact_threshold']
        delta_threshold = thresholds['disposal_artifact_delta_threshold']
        
        if (disposal_pre < artifact_threshold or disposal_post < artifact_threshold or
            (disposal_delta is not None and abs(disposal_delta) > delta_threshold)):
            
            result.issues.append(ValidationIssue(
                category="disposal_artifacts",
                message=f"Disposal artifacts detected: Pre={disposal_pre:.3f}, Post={disposal_post:.3f}"
                       f"{f', Delta={disposal_delta:.3f}' if disposal_delta is not None else ''}",
                expected_value=artifact_threshold,
                actual_value=min(disposal_pre, disposal_post),
                threshold=artifact_threshold
            ))
    
    def _validate_temporal_consistency(
        self,
        result: ValidationResult,
        compression_metrics: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> None:
        """Validate temporal consistency scores."""
        
        temporal_score = compression_metrics.get('temporal_consistency_post')
        
        if temporal_score is None:
            result.warnings.append(ValidationWarning(
                category="temporal_consistency",
                message="Temporal consistency data unavailable for validation"
            ))
            return
        
        threshold = thresholds['temporal_consistency_threshold']
        
        if temporal_score < threshold:
            result.warnings.append(ValidationWarning(
                category="temporal_consistency",
                message=f"Low temporal consistency: {temporal_score:.3f} < {threshold:.3f}",
                expected_value=threshold,
                actual_value=temporal_score,
                threshold=threshold,
                recommendation="Consider using different compression parameters to preserve temporal consistency"
            ))
    
    def _validate_multi_metric_combinations(
        self,
        result: ValidationResult,
        original_metadata: GifMetadata,
        compression_metrics: Dict[str, Any]
    ) -> None:
        """Validate complex multi-metric combinations for advanced issue detection."""
        
        # Combination 1: Low quality + High compression ratio = Suspicious efficiency
        composite_quality = compression_metrics.get('composite_quality')
        compression_ratio = compression_metrics.get('compression_ratio')
        
        if composite_quality and compression_ratio:
            quality_compression_ratio = composite_quality / min(compression_ratio / 10, 1)
            if quality_compression_ratio < 0.6:
                result.warnings.append(ValidationWarning(
                    category="quality_compression_mismatch",
                    message=f"Suspicious quality/compression ratio: Quality={composite_quality:.3f}, "
                           f"Compression={compression_ratio:.1f}x",
                    recommendation="Review compression parameters - quality may be unnecessarily low"
                ))
        
        # Combination 2: Significant FPS change + No frame reduction = Timing issue
        original_fps = original_metadata.orig_fps
        compressed_fps = compression_metrics.get('orig_fps', original_fps)
        original_frames = original_metadata.orig_frames
        compressed_frames = compression_metrics.get('compressed_frame_count', original_frames)
        
        if (original_fps and compressed_fps and original_frames and compressed_frames and
            original_frames == compressed_frames):  # No frame reduction
            
            fps_change = abs(compressed_fps - original_fps) / original_fps if original_fps > 0 else 0
            if fps_change > 0.1:  # 10% FPS change without frame reduction
                result.issues.append(ValidationIssue(
                    category="fps_timing_issue",
                    message=f"FPS changed without frame reduction: {original_fps:.1f}fps → {compressed_fps:.1f}fps",
                    expected_value=original_fps,
                    actual_value=compressed_fps
                ))
        
        # Combination 3: High disposal artifacts + Low temporal consistency = Animation corruption
        disposal_pre = compression_metrics.get('disposal_artifacts_pre')
        temporal_consistency = compression_metrics.get('temporal_consistency_post')
        
        if disposal_pre and temporal_consistency:
            if disposal_pre < 0.8 and temporal_consistency < 0.7:
                result.issues.append(ValidationIssue(
                    category="animation_corruption",
                    message=f"Animation corruption detected: Poor disposal artifacts ({disposal_pre:.3f}) + "
                           f"Low temporal consistency ({temporal_consistency:.3f})",
                    severity="CRITICAL"
                ))
        
        # Combination 4: Extreme frame reduction + Poor quality = Over-compression
        if (original_frames and compressed_frames and composite_quality):
            frame_reduction = compressed_frames / original_frames
            if frame_reduction < 0.3 and composite_quality < 0.6:
                result.warnings.append(ValidationWarning(
                    category="over_compression",
                    message=f"Possible over-compression: {frame_reduction*100:.0f}% frames retained, "
                           f"{composite_quality*100:.0f}% quality",
                    recommendation="Consider reducing compression aggressiveness"
                ))
    
    def get_validation_result(self, gif_name: str, pipeline_id: str) -> Optional[ValidationResult]:
        """Get validation result for a specific GIF and pipeline combination."""
        key = f"{gif_name}_{pipeline_id}"
        return self.validation_results.get(key)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all validation results."""
        if not self.validation_results:
            return {"total_validations": 0}
        
        status_counts = {status.value: 0 for status in ValidationStatus}
        total_issues = 0
        total_warnings = 0
        
        for result in self.validation_results.values():
            status_counts[result.status.value] += 1
            total_issues += len(result.issues)
            total_warnings += len(result.warnings)
        
        return {
            "total_validations": len(self.validation_results),
            "status_breakdown": status_counts,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "pass_rate": status_counts[ValidationStatus.PASS.value] / len(self.validation_results) * 100
        }
    
    def clear_results(self) -> None:
        """Clear all stored validation results."""
        self.validation_results.clear()
        logger.info("Validation results cleared")