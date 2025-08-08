"""Main presets interface module for GifLab experiments.

This module provides the main interface for accessing experiment presets,
acting as a central point for preset retrieval and management.
"""

from .targeted_presets import ExperimentPreset, SlotConfiguration, PRESET_REGISTRY
from .builtin_presets import TARGETED_PRESETS, register_builtin_presets

# Re-export important classes and constants
__all__ = [
    'ExperimentPreset',
    'SlotConfiguration', 
    'PRESET_REGISTRY',
    'TARGETED_PRESETS',
    'EXPERIMENT_PRESETS',
    'get_preset',
    'list_presets'
]

# Alias for backward compatibility
EXPERIMENT_PRESETS = TARGETED_PRESETS


def get_preset(preset_id: str) -> ExperimentPreset:
    """Get an experiment preset by ID.
    
    Args:
        preset_id: The preset identifier (e.g., 'frame-focus', 'color-optimization')
        
    Returns:
        The ExperimentPreset object
        
    Raises:
        ValueError: If the preset ID is not found
    """
    return PRESET_REGISTRY.get(preset_id)


def list_presets() -> dict[str, str]:
    """List all available presets with their descriptions.
    
    Returns:
        Dictionary mapping preset IDs to descriptions
    """
    return PRESET_REGISTRY.list_presets()
