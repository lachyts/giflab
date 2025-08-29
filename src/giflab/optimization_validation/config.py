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
    
    # Content-type specific threshold adjustments - Updated for real-world scenarios
    content_type_adjustments = {
        'animation_heavy': {
            'minimum_quality_floor': 0.35,        # Lower quality expectations for complex animation
            'fps_deviation_tolerance': 0.15,      # More tolerant of FPS changes due to complexity
            'temporal_consistency_threshold': 0.3, # More lenient temporal consistency
            'minimum_efficiency': 0.25            # Lower efficiency expectations due to complexity
        },
        
        'smooth_gradient': {
            'minimum_quality_floor': 0.45,        # Moderate quality for gradients  
            'disposal_artifact_threshold': 0.9,   # Stricter artifact checking
            'minimum_efficiency': 0.4,            # Gradients can be challenging to compress efficiently
            'fps_deviation_tolerance': 0.05        # Very strict FPS consistency
        },
        
        'complex_gradient': {
            'minimum_quality_floor': 0.4,         # Similar to smooth but more complex
            'disposal_artifact_threshold': 0.85,  # Slightly more lenient artifacts
            'minimum_efficiency': 0.3,            # Lower efficiency for complex gradients
            'temporal_consistency_threshold': 0.35 # More lenient temporal consistency
        },
        
        'static_minimal_change': {
            'minimum_quality_floor': 0.6,         # Higher quality for static content
            'fps_deviation_tolerance': 0.05,      # Very strict FPS consistency
            'minimum_efficiency': 0.5,            # Should compress reasonably well
            'frame_reduction_tolerance': 0.05,    # Very precise frame reduction
        },
        
        'high_frequency_detail': {
            'minimum_quality_floor': 0.3,         # Very lenient quality due to complexity
            'minimum_efficiency': 0.25,           # Very low efficiency expectations
            'temporal_consistency_threshold': 0.3, # More lenient temporal requirements
            'fps_deviation_tolerance': 0.15       # More tolerant due to detail complexity
        },
        
        'photographic_noise': {
            'minimum_quality_floor': 0.35,        # Moderate quality expectations
            'disposal_artifact_threshold': 0.8,   # More lenient artifact checking
            'minimum_efficiency': 0.25,           # Very challenging content to compress
            'temporal_consistency_threshold': 0.3  # More lenient temporal requirements
        },
        
        'few_colors': {
            'minimum_efficiency': 0.6,            # Should still compress well but realistic
            'disposal_artifact_threshold': 0.9,   # High artifact standards
            'minimum_quality_floor': 0.5,         # Good but realistic quality for simple content
        },
        
        'many_colors': {
            'minimum_efficiency': 0.25,           # Very low efficiency due to complexity
            'minimum_quality_floor': 0.35,        # Very lenient quality
            'disposal_artifact_threshold': 0.8,   # Moderate artifact tolerance
        },
        
        'geometric_patterns': {
            'minimum_quality_floor': 0.5,         # Should maintain reasonable precision
            'disposal_artifact_threshold': 0.9,   # High artifact standards for patterns
            'minimum_efficiency': 0.4,            # Should compress reasonably well
            'temporal_consistency_threshold': 0.35 # Patterns should be somewhat consistent
        },
        
        'texture_complex': {
            'minimum_quality_floor': 0.3,         # Very challenging content for quality
            'minimum_efficiency': 0.2,            # Very low efficiency due to texture complexity
            'fps_deviation_tolerance': 0.15,      # More tolerant due to complexity
            'disposal_artifact_threshold': 0.8,   # More lenient artifact checking
            'temporal_consistency_threshold': 0.3  # Very lenient temporal requirements
        }
    }
    
    # Pipeline-specific validation rules
    pipeline_specific_rules = {
        'frame_reduction': {
            'frame_reduction_tolerance': 0.05,    # Very precise for frame-focused pipelines
            'fps_deviation_tolerance': 0.2,       # More tolerant of FPS changes when frames change
            'temporal_consistency_threshold': 0.3  # May reduce temporal consistency
        },
        
        'color_reduction': {
            'disposal_artifact_threshold': 0.9,   # Stricter artifact checking
            'minimum_quality_floor': 0.4,         # Realistic quality expectations
            'temporal_consistency_threshold': 0.35 # Color changes shouldn't severely affect temporal consistency
        },
        
        'lossy_compression': {
            'minimum_quality_floor': 0.3,         # More lenient quality due to lossy nature
            'minimum_efficiency': 0.4,            # Should achieve reasonable compression
            'temporal_consistency_threshold': 0.3, # More lenient temporal requirements
            'disposal_artifact_threshold': 0.8    # Lossy may introduce some artifacts
        },
        
        'hybrid_optimization': {
            'minimum_quality_floor': 0.35,        # Balanced but realistic expectations
            'minimum_efficiency': 0.35,           # Realistic compression expected
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