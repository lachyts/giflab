"""
Configuration system for GifLab validation.

Provides default validation configurations and content-type specific adjustments.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

from .data_structures import ValidationConfig

logger = logging.getLogger(__name__)


def get_default_validation_config() -> ValidationConfig:
    """Get the default validation configuration with content-type adjustments."""
    
    # Content-type specific threshold adjustments
    content_type_adjustments = {
        'animation_heavy': {
            'minimum_quality_floor': 0.75,        # Higher quality expectations for animation
            'fps_deviation_tolerance': 0.15,      # More tolerant of FPS changes due to complexity
            'temporal_consistency_threshold': 0.8, # Higher temporal consistency requirement
            'efficiency_warning_threshold': 0.6   # Lower efficiency expectations due to complexity
        },
        
        'smooth_gradient': {
            'minimum_quality_floor': 0.8,         # Very high quality for gradients  
            'disposal_artifact_threshold': 0.9,   # Stricter artifact checking
            'efficiency_warning_threshold': 0.75, # Expect good compression
            'fps_deviation_tolerance': 0.05        # Very strict FPS consistency
        },
        
        'static_minimal_change': {
            'minimum_quality_floor': 0.85,        # Highest quality for static content
            'fps_deviation_tolerance': 0.05,      # Very strict FPS consistency
            'minimum_efficiency': 0.8,            # Should compress very well
            'frame_reduction_tolerance': 0.05,    # Very precise frame reduction
            'efficiency_warning_threshold': 0.85   # High efficiency expectations
        },
        
        'high_frequency_detail': {
            'minimum_quality_floor': 0.6,         # More lenient quality due to complexity
            'minimum_efficiency': 0.5,            # Lower efficiency expectations
            'temporal_consistency_threshold': 0.7, # More lenient temporal requirements
            'fps_deviation_tolerance': 0.15       # More tolerant due to detail complexity
        },
        
        'photographic_noise': {
            'minimum_quality_floor': 0.65,        # Moderate quality expectations
            'disposal_artifact_threshold': 0.8,   # More lenient artifact checking
            'efficiency_warning_threshold': 0.6,  # Lower efficiency expectations
            'minimum_efficiency': 0.55            # Challenging content to compress
        },
        
        'few_colors': {
            'minimum_efficiency': 0.85,           # Should compress extremely well
            'disposal_artifact_threshold': 0.9,   # High artifact standards
            'minimum_quality_floor': 0.8,         # High quality for simple content
            'efficiency_warning_threshold': 0.9   # Very high efficiency expectations
        },
        
        'many_colors': {
            'minimum_efficiency': 0.5,            # Lower efficiency due to complexity
            'minimum_quality_floor': 0.65,        # More lenient quality
            'disposal_artifact_threshold': 0.8,   # Moderate artifact tolerance
            'efficiency_warning_threshold': 0.6   # Lower efficiency expectations
        },
        
        'geometric_patterns': {
            'minimum_quality_floor': 0.85,        # Should maintain geometric precision
            'disposal_artifact_threshold': 0.9,   # High artifact standards for patterns
            'minimum_efficiency': 0.75,           # Should compress well
            'temporal_consistency_threshold': 0.8  # Patterns should be temporally consistent
        },
        
        'texture_complex': {
            'minimum_quality_floor': 0.6,         # Challenging content for quality
            'minimum_efficiency': 0.5,            # Lower efficiency due to texture complexity
            'fps_deviation_tolerance': 0.15,      # More tolerant due to complexity
            'disposal_artifact_threshold': 0.8    # More lenient artifact checking
        }
    }
    
    # Pipeline-specific validation rules
    pipeline_specific_rules = {
        'frame_reduction': {
            'frame_reduction_tolerance': 0.05,    # Very precise for frame-focused pipelines
            'fps_deviation_tolerance': 0.2,       # More tolerant of FPS changes when frames change
            'temporal_consistency_threshold': 0.7  # May reduce temporal consistency
        },
        
        'color_reduction': {
            'disposal_artifact_threshold': 0.9,   # Stricter artifact checking
            'minimum_quality_floor': 0.75,        # Higher quality expectations
            'temporal_consistency_threshold': 0.8  # Color changes shouldn't affect temporal consistency
        },
        
        'lossy_compression': {
            'minimum_quality_floor': 0.6,         # More lenient quality due to lossy nature
            'minimum_efficiency': 0.75,           # Should achieve good compression
            'temporal_consistency_threshold': 0.7, # More lenient temporal requirements
            'disposal_artifact_threshold': 0.8    # Lossy may introduce some artifacts
        },
        
        'hybrid_optimization': {
            'minimum_quality_floor': 0.7,         # Balanced expectations
            'minimum_efficiency': 0.7,            # Good compression expected
            'fps_deviation_tolerance': 0.08,      # Moderate FPS tolerance
            'frame_reduction_tolerance': 0.08     # Moderate frame tolerance
        }
    }
    
    return ValidationConfig(
        content_type_adjustments=content_type_adjustments,
        pipeline_specific_rules=pipeline_specific_rules
    )


def load_validation_config(config_path: Optional[Path] = None) -> ValidationConfig:
    """Load validation configuration from file or return defaults."""
    
    if config_path is None:
        # Look for config in standard locations
        potential_paths = [
            Path.cwd() / "validation_config.yaml",
            Path.home() / ".giflab" / "validation_config.yaml",
            Path(__file__).parent / "default_config.yaml"
        ]
        
        config_path = None
        for path in potential_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and config_path.exists():
        try:
            logger.info(f"Loading validation config from: {config_path}")
            return ValidationConfig.load_from_file(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.warning("Falling back to default configuration")
    
    return get_default_validation_config()


def save_validation_config(config: ValidationConfig, config_path: Path) -> None:
    """Save validation configuration to YAML file."""
    
    # Convert config to dictionary for YAML serialization
    config_dict = {
        'frame_reduction_tolerance': config.frame_reduction_tolerance,
        'fps_deviation_tolerance': config.fps_deviation_tolerance,
        'fps_warning_threshold': config.fps_warning_threshold,
        'minimum_quality_floor': config.minimum_quality_floor,
        'quality_warning_threshold': config.quality_warning_threshold,
        'minimum_efficiency': config.minimum_efficiency,
        'efficiency_warning_threshold': config.efficiency_warning_threshold,
        'disposal_artifact_threshold': config.disposal_artifact_threshold,
        'disposal_artifact_delta_threshold': config.disposal_artifact_delta_threshold,
        'temporal_consistency_threshold': config.temporal_consistency_threshold,
        'content_type_adjustments': config.content_type_adjustments,
        'pipeline_specific_rules': config.pipeline_specific_rules
    }
    
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Validation configuration saved to: {config_path}")


def create_default_config_file(config_path: Path) -> None:
    """Create a default configuration file for user customization."""
    
    default_config = get_default_validation_config()
    save_validation_config(default_config, config_path)
    
    logger.info(f"Created default validation config at: {config_path}")
    logger.info("You can customize the thresholds by editing this file")