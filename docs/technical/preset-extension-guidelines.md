# Preset Extension Guidelines

This document provides technical guidelines for creating custom presets and extending the targeted experiment presets system.

## Overview

The preset system is designed to be easily extensible. Developers can create custom presets programmatically, register them with the global registry, and integrate them into existing workflows.

## Creating Custom Presets

### Basic Preset Structure

All presets follow the same basic structure using the `ExperimentPreset` dataclass:

```python
from giflab.core.targeted_presets import ExperimentPreset, SlotConfiguration

preset = ExperimentPreset(
    name="Descriptive Name",
    description="Clear research purpose and methodology",
    
    # Required: Three slot configurations
    frame_slot=SlotConfiguration(...),
    color_slot=SlotConfiguration(...), 
    lossy_slot=SlotConfiguration(...),
    
    # Optional: Enhancement parameters
    custom_sampling=None,
    max_combinations=None,
    tags=[],
    author=None,
    version="1.0"
)
```

### Slot Configuration Types

Each slot must be configured as either **variable** or **locked**:

#### Variable Slot Configuration
```python
# Test multiple algorithms in this dimension
variable_slot = SlotConfiguration(
    type="variable",
    scope=["*"],  # All tools OR ["tool1", "tool2"] for specific tools
    parameters={"param": [val1, val2, val3]}  # Parameter ranges
)
```

#### Locked Slot Configuration
```python
# Fix to specific algorithm and parameters
locked_slot = SlotConfiguration(
    type="locked",
    implementation="specific-tool-name",  # Exact tool name
    parameters={"param": value}  # Fixed parameters
)
```

### Validation Requirements

Custom presets must meet these validation criteria:

1. **At least one variable slot**: Cannot have all three slots locked
2. **Valid tool names**: All implementations must exist in capability registry
3. **Valid parameter values**: Parameters must be within acceptable ranges
4. **Valid sampling strategy**: Custom sampling (if specified) must be recognized
5. **Positive max_combinations**: If specified, must be > 0

### Parameter Guidelines

#### Frame Slot Parameters
```python
# Variable frame slot parameters
parameters = {
    "ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # 0.0-1.0
}

# Locked frame slot parameters  
parameters = {
    "ratio": 0.8  # Single value 0.0-1.0
}
```

#### Color Slot Parameters
```python
# Variable color slot parameters
parameters = {
    "colors": [256, 128, 64, 32, 16, 8]  # Color counts 1-256
}

# Locked color slot parameters
parameters = {
    "colors": 32  # Single color count 1-256
}
```

#### Lossy Slot Parameters  
```python
# Variable lossy slot parameters
parameters = {
    "levels": [0, 20, 40, 60, 80, 100, 120, 140, 160]  # 0-200
}

# Locked lossy slot parameters
parameters = {
    "level": 40  # Single level 0-200
}
```

## Preset Categories and Design Patterns

### Research Presets
**Purpose**: Single-dimension algorithm comparisons
**Pattern**: One variable slot, two locked slots

```python
# Frame algorithm comparison
research_preset = ExperimentPreset(
    name="Custom Frame Study",
    description="Compare specific frame algorithms",
    frame_slot=SlotConfiguration(
        type="variable", 
        scope=["animately-frame", "ffmpeg-frame", "gifsicle-frame"]
    ),
    color_slot=SlotConfiguration(
        type="locked",
        implementation="ffmpeg-color",
        parameters={"colors": 32}
    ),
    lossy_slot=SlotConfiguration(
        type="locked", 
        implementation="none-lossy",
        parameters={"level": 0}
    ),
    tags=["research", "frame-comparison"]
)
```

### Specialized Presets
**Purpose**: Focused optimization for specific scenarios
**Pattern**: Targeted tool selection with specific parameters

```python
# High-quality animation preset
specialized_preset = ExperimentPreset(
    name="High Quality Animation",
    description="Optimize for high-quality animated content",
    frame_slot=SlotConfiguration(
        type="locked",
        implementation="animately-frame",
        parameters={"ratio": 0.9}  # Minimal frame reduction
    ),
    color_slot=SlotConfiguration(
        type="variable",
        scope=["ffmpeg-color", "imagemagick-color-floyd"],  # High-quality options
        parameters={"colors": [128, 64]}  # Higher color counts
    ),
    lossy_slot=SlotConfiguration(
        type="variable",
        scope=["gifski-lossy", "animately-advanced-lossy"],  # Quality-focused
        parameters={"levels": [80, 100]}  # Higher quality levels
    ),
    tags=["specialized", "high-quality", "animation"]
)
```

### Baseline Presets
**Purpose**: Comprehensive multi-dimensional comparisons
**Pattern**: Multiple variable slots with conservative parameters

```python
# Custom baseline comparison
baseline_preset = ExperimentPreset(
    name="Custom Tool Baseline",
    description="Compare specific tool combinations",
    frame_slot=SlotConfiguration(
        type="variable",
        scope=["animately-frame", "ffmpeg-frame"],
        parameters={"ratios": [1.0, 0.8, 0.5]}
    ),
    color_slot=SlotConfiguration(
        type="variable", 
        scope=["animately-color", "ffmpeg-color"],
        parameters={"colors": [64, 32]}
    ),
    lossy_slot=SlotConfiguration(
        type="variable",
        scope=["animately-advanced-lossy", "ffmpeg-lossy"],
        parameters={"levels": [0, 40, 80]}
    ),
    max_combinations=100,  # Limit explosion
    tags=["baseline", "comparison"]
)
```

### Development Presets
**Purpose**: Fast testing and debugging
**Pattern**: Minimal combinations with reliable tools

```python
# Custom development preset
dev_preset = ExperimentPreset(
    name="Custom Quick Test",
    description="Fast validation for specific tools",
    frame_slot=SlotConfiguration(
        type="variable",
        scope=["animately-frame"],  # Single reliable tool
        parameters={"ratios": [1.0]}  # Single ratio
    ),
    color_slot=SlotConfiguration(
        type="locked",
        implementation="ffmpeg-color",
        parameters={"colors": 32}
    ),
    lossy_slot=SlotConfiguration(
        type="locked",
        implementation="none-lossy", 
        parameters={"level": 0}
    ),
    max_combinations=5,
    tags=["development", "testing"]
)
```

## Registration and Lifecycle Management

### Registering Custom Presets

```python
from giflab.experimental.targeted_presets import PRESET_REGISTRY

# Create custom preset
custom_preset = ExperimentPreset(...)

# Register with global registry
PRESET_REGISTRY.register("custom-preset-id", custom_preset)

# Verify registration
available = PRESET_REGISTRY.list_presets()
assert "custom-preset-id" in available
```

### Preset Registry Features

#### Conflict Detection
```python
# Registry detects and warns about similar presets
similar = PRESET_REGISTRY.find_similar_presets(new_preset)
if similar:
    print(f"Similar presets found: {similar}")
```

#### Validation During Registration
```python
try:
    PRESET_REGISTRY.register("invalid-preset", bad_preset)
except ValueError as e:
    print(f"Registration failed: {e}")
```

### Preset Lifecycle

1. **Creation**: Define preset with required slots and parameters
2. **Validation**: Automatic validation during creation and registration  
3. **Registration**: Add to global registry for system-wide access
4. **Usage**: Access via registry or CLI for experiment execution
5. **Management**: Update, replace, or remove as needed

## Advanced Preset Features

### Custom Sampling Integration
```python
preset_with_sampling = ExperimentPreset(
    name="Custom Sampling Study", 
    description="Preset with custom sampling strategy",
    frame_slot=SlotConfiguration(type="variable", scope=["*"]),
    color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
    lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),
    custom_sampling="factorial",  # Apply additional sampling
    max_combinations=200
)
```

### Pipeline Count Estimation
```python
# Estimate pipeline count before execution
estimated_count = preset.estimate_pipeline_count()
print(f"Estimated pipelines: {estimated_count}")

# Validate feasibility
from giflab.experimental.targeted_generator import TargetedPipelineGenerator
generator = TargetedPipelineGenerator()
validation = generator.validate_preset_feasibility(preset)
print(f"Valid: {validation['valid']}, Count: {validation['estimated_pipelines']}")
```

### Parameter Range Optimization
```python
# Optimize parameter ranges for specific content
content_optimized_preset = ExperimentPreset(
    name="Content-Specific Optimization",
    description="Optimized for specific content characteristics",
    frame_slot=SlotConfiguration(
        type="variable",
        scope=["*"],
        parameters={
            # Optimize ratios for high-motion content
            "ratios": [1.0, 0.8, 0.6, 0.4, 0.2]  # More aggressive reduction
        }
    ),
    color_slot=SlotConfiguration(
        type="variable",
        scope=["ffmpeg-color", "ffmpeg-color-floyd"],
        parameters={
            # Optimize colors for photographic content  
            "colors": [128, 64, 32]  # Higher color counts for photos
        }
    ),
    lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy")
)
```

## Tool Integration

### Dynamic Tool Discovery
```python
from giflab.capability_registry import tools_for

# Get available tools for dynamic preset creation
frame_tools = [tool.NAME for tool in tools_for("frame_reduction")]
color_tools = [tool.NAME for tool in tools_for("color_reduction")]  
lossy_tools = [tool.NAME for tool in tools_for("lossy_compression")]

# Create preset with all available tools
comprehensive_preset = ExperimentPreset(
    name="All Available Tools",
    description="Test all available algorithms",
    frame_slot=SlotConfiguration(type="variable", scope=frame_tools[:3]),  # Limit to first 3
    color_slot=SlotConfiguration(type="variable", scope=color_tools[:5]),  # Limit to first 5
    lossy_slot=SlotConfiguration(type="locked", implementation=lossy_tools[0])  # Use first
)
```

### Tool Availability Validation
```python
def validate_tool_availability(preset: ExperimentPreset) -> bool:
    """Validate that all tools in preset are available."""
    generator = TargetedPipelineGenerator()
    validation = generator.validate_preset_feasibility(preset)
    
    if not validation["valid"]:
        print(f"Validation errors: {validation['errors']}")
        return False
        
    print(f"All tools available. Estimated pipelines: {validation['estimated_pipelines']}")
    return True
```

## Best Practices

### Preset Design Guidelines

1. **Clear Research Purpose**: Each preset should answer a specific research question
2. **Balanced Efficiency**: Aim for meaningful comparisons without excessive combinations
3. **Parameter Relevance**: Include only parameter values relevant to the research goal
4. **Tool Selection**: Choose tools appropriate for the comparison scope
5. **Resource Awareness**: Consider computational resources when setting max_combinations

### Naming Conventions
```python
# Good preset naming
"frame-algorithm-comparison"     # Clear purpose
"high-quality-animation"         # Specific use case
"development-quick-test"         # Context and speed
"photographic-content-optimized" # Content-specific

# Avoid generic naming
"test-preset"
"custom"
"experiment"
```

### Parameter Optimization
```python
# Optimize parameters for research goals

# For algorithm comparison (keep parameters fixed)
algorithm_comparison = {
    "ratios": [0.8],        # Single representative ratio
    "colors": [32],         # Single representative color count
    "levels": [40]          # Single representative lossy level
}

# For parameter optimization (vary parameters)
parameter_optimization = {
    "ratios": [1.0, 0.8, 0.6, 0.4, 0.2],    # Full range
    "colors": [256, 128, 64, 32, 16, 8],     # Full range
    "levels": [0, 20, 40, 60, 80, 100, 120]  # Full range
}
```

### Error Handling
```python
def create_safe_preset(name: str, config: dict) -> Optional[ExperimentPreset]:
    """Create preset with comprehensive error handling."""
    try:
        preset = ExperimentPreset(
            name=name,
            description=config.get("description", "Custom preset"),
            frame_slot=SlotConfiguration(**config["frame_slot"]),
            color_slot=SlotConfiguration(**config["color_slot"]),
            lossy_slot=SlotConfiguration(**config["lossy_slot"]),
            max_combinations=config.get("max_combinations")
        )
        
        # Validate before returning
        generator = TargetedPipelineGenerator()
        validation = generator.validate_preset_feasibility(preset)
        
        if not validation["valid"]:
            raise ValueError(f"Preset validation failed: {validation['errors']}")
            
        return preset
        
    except Exception as e:
        print(f"Failed to create preset '{name}': {e}")
        return None
```

## Testing Custom Presets

### Unit Testing
```python
import pytest
from giflab.core.targeted_presets import ExperimentPreset, SlotConfiguration

def test_custom_preset_validation():
    """Test that custom preset validates correctly."""
    preset = ExperimentPreset(
        name="Test Preset",
        description="Test preset for validation",
        frame_slot=SlotConfiguration(type="variable", scope=["*"]),
        color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),
        lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy")
    )
    
    # Should not raise validation errors
    assert preset.name == "Test Preset"
    assert len(preset.get_variable_slots()) == 1
    assert "frame" in preset.get_variable_slots()

def test_preset_pipeline_generation():
    """Test that preset generates expected number of pipelines."""
    from giflab.experimental.targeted_generator import TargetedPipelineGenerator
    
    generator = TargetedPipelineGenerator()
    preset = ExperimentPreset(...)  # Define test preset
    
    pipelines = generator.generate_targeted_pipelines(preset)
    assert len(pipelines) > 0
    assert all(hasattr(p, 'steps') for p in pipelines)
```

### Integration Testing
```python
def test_preset_with_experimental_runner():
    """Test preset integration with GifLabRunner."""
    from giflab.core.runner import GifLabRunner
    from giflab.experimental.targeted_presets import PRESET_REGISTRY
    
    # Register custom preset
    custom_preset = ExperimentPreset(...)
    PRESET_REGISTRY.register("test-integration", custom_preset)
    
    # Test with runner
    runner = GifLabRunner(use_cache=False)
    pipelines = runner.generate_targeted_pipelines("test-integration")
    
    assert len(pipelines) > 0
```

## Migration and Compatibility

### Upgrading Existing Presets
```python
# Version 1.0 preset
old_preset = ExperimentPreset(
    name="Old Preset",
    description="Legacy preset",
    frame_slot=SlotConfiguration(...),
    color_slot=SlotConfiguration(...),
    lossy_slot=SlotConfiguration(...),
    version="1.0"
)

# Version 2.0 with enhancements
new_preset = ExperimentPreset(
    name="Old Preset",  # Keep same name for compatibility
    description="Enhanced legacy preset with improved parameters",
    frame_slot=SlotConfiguration(...),  # Enhanced configuration
    color_slot=SlotConfiguration(...),  # Enhanced configuration
    lossy_slot=SlotConfiguration(...),  # Enhanced configuration
    max_combinations=100,  # New optimization
    tags=["upgraded", "legacy-compatible"],
    version="2.0"
)

# Replace in registry
PRESET_REGISTRY.register("old-preset", new_preset)  # Overwrites old version
```

### Backward Compatibility
- Preset IDs should remain stable across versions
- Parameter formats should be backward compatible
- API changes should be additive, not breaking

## Performance Considerations

### Pipeline Count Management
```python
# Estimate and limit pipeline count
def create_efficient_preset(max_pipelines: int = 100) -> ExperimentPreset:
    """Create preset with pipeline count constraint."""
    base_preset = ExperimentPreset(...)
    
    # Check estimated count
    estimated = base_preset.estimate_pipeline_count()
    
    if estimated > max_pipelines:
        # Apply max_combinations limit
        return ExperimentPreset(
            **base_preset.__dict__,
            max_combinations=max_pipelines
        )
    
    return base_preset
```

### Memory Optimization
- Use specific tool scopes instead of wildcards when possible
- Limit parameter ranges to essential values
- Set appropriate max_combinations limits
- Consider using custom_sampling for large combination sets

This guide provides the foundation for creating effective custom presets that integrate seamlessly with the targeted experiment presets system.