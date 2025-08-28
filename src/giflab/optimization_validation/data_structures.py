"""
Data structures for GifLab validation system.

Defines the core data types used for validation results, statuses, and configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path


class ValidationStatus(Enum):
    """Validation status levels for compression results."""
    
    PASS = "PASS"           # All validation checks passed
    WARNING = "WARNING"     # Minor issues detected, compression acceptable
    ERROR = "ERROR"         # Significant issues, compression problematic  
    ARTIFACT = "ARTIFACT"   # Disposal artifacts or corruption detected
    UNKNOWN = "UNKNOWN"     # Validation could not be performed


@dataclass
class ValidationIssue:
    """Represents a validation error or critical issue."""
    
    category: str           # e.g., "fps_consistency", "quality_threshold", "frame_count"
    message: str           # Human-readable description
    expected_value: Optional[Any] = None  # What was expected
    actual_value: Optional[Any] = None    # What was measured
    threshold: Optional[float] = None     # Threshold that was violated
    severity: str = "ERROR"               # ERROR, CRITICAL
    

@dataclass  
class ValidationWarning:
    """Represents a validation warning or minor issue."""
    
    category: str           # e.g., "efficiency_suboptimal", "quality_below_ideal"
    message: str           # Human-readable description
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None  # Suggested improvement


@dataclass
class ValidationMetrics:
    """Summary of key metrics used in validation."""
    
    # Frame and timing metrics
    original_frame_count: int
    compressed_frame_count: int
    original_fps: float
    compressed_fps: float
    frame_reduction_ratio: Optional[float] = None
    fps_deviation_percent: Optional[float] = None
    
    # Quality metrics
    composite_quality: Optional[float] = None
    efficiency: Optional[float] = None
    compression_ratio: Optional[float] = None
    
    # Temporal and artifact metrics
    temporal_consistency_pre: Optional[float] = None
    temporal_consistency_post: Optional[float] = None
    disposal_artifacts_pre: Optional[float] = None
    disposal_artifacts_post: Optional[float] = None
    disposal_artifacts_delta: Optional[float] = None
    
    # Size metrics
    original_size_kb: Optional[float] = None
    compressed_size_kb: Optional[float] = None


@dataclass
class ValidationResult:
    """Complete validation result for a compression operation."""
    
    # Identification
    pipeline_id: str
    gif_name: str
    content_type: str
    
    # Overall validation status
    status: ValidationStatus
    
    # Issues and warnings
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    
    # Metrics summary
    metrics: Optional[ValidationMetrics] = None
    
    # Validation configuration used
    effective_thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    validation_time_ms: Optional[int] = None
    config_version: Optional[str] = None
    
    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return self.status in (ValidationStatus.ERROR, ValidationStatus.ARTIFACT)
    
    def has_warnings(self) -> bool:
        """Check if validation found any warnings."""
        return len(self.warnings) > 0 or self.status == ValidationStatus.WARNING
    
    def is_acceptable(self) -> bool:
        """Check if compression result is acceptable despite any warnings."""
        return self.status in (ValidationStatus.PASS, ValidationStatus.WARNING)
    
    def get_summary(self) -> str:
        """Get a brief summary of validation results."""
        status_emoji = {
            ValidationStatus.PASS: "🟢",
            ValidationStatus.WARNING: "🟡", 
            ValidationStatus.ERROR: "🔴",
            ValidationStatus.ARTIFACT: "⚠️",
            ValidationStatus.UNKNOWN: "❓"
        }
        
        emoji = status_emoji.get(self.status, "❓")
        error_count = len(self.issues)
        warning_count = len(self.warnings)
        
        summary = f"{emoji} {self.status.value}"
        if error_count > 0:
            summary += f" ({error_count} errors"
            if warning_count > 0:
                summary += f", {warning_count} warnings"
            summary += ")"
        elif warning_count > 0:
            summary += f" ({warning_count} warnings)"
            
        return summary
    
    def get_detailed_report(self) -> str:
        """Get a detailed validation report for terminal output."""
        lines = []
        lines.append(f"Validation Report: {self.gif_name} ({self.content_type})")
        lines.append(f"Pipeline: {self.pipeline_id}")
        lines.append(f"Status: {self.get_summary()}")
        lines.append("")
        
        if self.metrics:
            lines.append("Key Metrics:")
            lines.append(f"  Frames: {self.metrics.original_frame_count} → {self.metrics.compressed_frame_count}")
            lines.append(f"  FPS: {self.metrics.original_fps:.1f} → {self.metrics.compressed_fps:.1f}")
            if self.metrics.composite_quality is not None:
                lines.append(f"  Quality: {self.metrics.composite_quality:.3f}")
            if self.metrics.efficiency is not None:
                lines.append(f"  Efficiency: {self.metrics.efficiency:.3f}")
            lines.append("")
        
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  ❌ {issue.category}: {issue.message}")
            lines.append("")
            
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning.category}: {warning.message}")
                if warning.recommendation:
                    lines.append(f"     Suggestion: {warning.recommendation}")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class ValidationConfig:
    """Configuration for validation thresholds and rules."""
    
    # Frame validation
    frame_reduction_tolerance: float = 0.1  # 10% tolerance for frame reduction accuracy
    
    # FPS validation
    fps_deviation_tolerance: float = 0.1    # 10% tolerance for FPS changes
    fps_warning_threshold: float = 0.05     # 5% warning threshold for FPS changes
    
    # Quality thresholds
    minimum_quality_floor: float = 0.7      # Minimum acceptable composite quality
    quality_warning_threshold: float = 0.8  # Warning threshold for quality
    
    # Efficiency thresholds
    minimum_efficiency: float = 0.6         # Minimum acceptable efficiency
    efficiency_warning_threshold: float = 0.7  # Warning threshold for efficiency
    
    # Disposal artifact thresholds
    disposal_artifact_threshold: float = 0.85    # Minimum acceptable disposal score
    disposal_artifact_delta_threshold: float = 0.1  # Max change in disposal score
    
    # Temporal consistency thresholds
    temporal_consistency_threshold: float = 0.75  # Minimum temporal consistency
    
    # Content-type specific adjustments
    content_type_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Pipeline-specific rules  
    pipeline_specific_rules: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'ValidationConfig':
        """Load validation configuration from YAML file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def get_effective_thresholds(self, content_type: Optional[str] = None, 
                               pipeline_type: Optional[str] = None) -> Dict[str, float]:
        """Get effective thresholds with content-type and pipeline adjustments."""
        # Start with base thresholds
        thresholds = {
            'frame_reduction_tolerance': self.frame_reduction_tolerance,
            'fps_deviation_tolerance': self.fps_deviation_tolerance,
            'fps_warning_threshold': self.fps_warning_threshold,
            'minimum_quality_floor': self.minimum_quality_floor,
            'quality_warning_threshold': self.quality_warning_threshold,
            'minimum_efficiency': self.minimum_efficiency,
            'efficiency_warning_threshold': self.efficiency_warning_threshold,
            'disposal_artifact_threshold': self.disposal_artifact_threshold,
            'disposal_artifact_delta_threshold': self.disposal_artifact_delta_threshold,
            'temporal_consistency_threshold': self.temporal_consistency_threshold,
        }
        
        # Apply content-type specific adjustments
        if content_type and content_type in self.content_type_adjustments:
            adjustments = self.content_type_adjustments[content_type]
            thresholds.update(adjustments)
        
        # Apply pipeline-specific rules
        if pipeline_type and pipeline_type in self.pipeline_specific_rules:
            rules = self.pipeline_specific_rules[pipeline_type]
            thresholds.update(rules)
        
        return thresholds