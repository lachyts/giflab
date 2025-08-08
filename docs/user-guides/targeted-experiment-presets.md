# Targeted Experiment Presets User Guide

This guide explains how to use the targeted experiment presets system for efficient GIF compression research with GifLab.

## Overview

The targeted experiment presets system allows you to run focused compression experiments by generating only the specific pipeline combinations you need, rather than generating all possible combinations and then sampling from them. This results in 80-99% efficiency gains for focused research studies.

## Quick Start

### Using Built-in Presets

The simplest way to run a targeted experiment is to use one of the built-in presets:

```bash
# List all available presets
poetry run python -m giflab experiment --list-presets

# Run a frame algorithm comparison (tests all frame reduction methods)
poetry run python -m giflab experiment --preset frame-focus

# Run a color optimization study (tests all color reduction methods)
poetry run python -m giflab experiment --preset color-optimization

# Quick test for development (minimal pipelines)
poetry run python -m giflab experiment --preset quick-test
```

### Custom Slot Configuration

For more control, you can specify custom slot configurations:

```bash
# Test all frame algorithms with specific color and lossy settings
poetry run python -m giflab experiment \
  --variable-slot frame=* \
  --lock-slot color=ffmpeg-color \
  --lock-slot lossy=animately-advanced-lossy \
  --slot-params color=colors:32 \
  --slot-params lossy=level:40

# Test specific color algorithms only
poetry run python -m giflab experiment \
  --variable-slot color=ffmpeg-color,gifsicle-color \
  --lock-slot frame=animately-frame \
  --lock-slot lossy=none-lossy
```

## Understanding Slot-Based Configuration

The targeted preset system is built around the concept of **slots** - the three main stages of GIF compression:

1. **Frame Slot**: Frame reduction algorithms (animately-frame, ffmpeg-frame, etc.)
2. **Color Slot**: Color reduction algorithms (ffmpeg-color, gifsicle-color, etc.)  
3. **Lossy Slot**: Lossy compression algorithms (animately-advanced-lossy, gifski-lossy, etc.)

Each slot can be configured as either:
- **Variable**: Test multiple algorithms in this slot
- **Locked**: Use a specific algorithm with fixed parameters

## Built-in Presets Reference

### Research Presets

#### `frame-focus`
- **Purpose**: Compare all frame reduction algorithms
- **Configuration**: Variable frame slot, locked color/lossy
- **Pipeline Count**: ~5 pipelines (99.5% efficiency vs full generation)
- **Use Case**: Determine which frame algorithm works best for your content

```bash
poetry run python -m giflab experiment --preset frame-focus
```

#### `color-optimization`
- **Purpose**: Compare all color reduction techniques and dithering methods
- **Configuration**: Variable color slot, locked frame/lossy
- **Pipeline Count**: ~17 pipelines (98.2% efficiency)
- **Use Case**: Find optimal color quantization approach

```bash
poetry run python -m giflab experiment --preset color-optimization
```

#### `lossy-quality-sweep`
- **Purpose**: Evaluate lossy compression effectiveness
- **Configuration**: Variable lossy slot, locked frame/color
- **Pipeline Count**: ~11 pipelines (98.8% efficiency)
- **Use Case**: Determine quality/size tradeoffs for lossy compression

```bash
poetry run python -m giflab experiment --preset lossy-quality-sweep
```

#### `tool-comparison-baseline`
- **Purpose**: Fair comparison across complete toolchain engines
- **Configuration**: All slots variable with conservative parameters
- **Pipeline Count**: ~64 pipelines (93.2% efficiency)
- **Use Case**: Compare animately vs ffmpeg vs gifsicle vs imagemagick

```bash
poetry run python -m giflab experiment --preset tool-comparison-baseline
```

### Specialized Presets

#### `dithering-focus`
- **Purpose**: Compare dithering algorithms specifically
- **Configuration**: Variable color slot with dithering-focused tools
- **Pipeline Count**: ~6 pipelines
- **Use Case**: Fine-tune dithering for specific content types

```bash
poetry run python -m giflab experiment --preset dithering-focus
```

#### `png-optimization`
- **Purpose**: Optimize PNG sequence workflows (gifski + animately-advanced)
- **Configuration**: Variable color/lossy focused on PNG-optimized tools
- **Pipeline Count**: ~4 pipelines
- **Use Case**: High-quality GIF creation from PNG sequences

```bash
poetry run python -m giflab experiment --preset png-optimization
```

### Development Preset

#### `quick-test`
- **Purpose**: Fast development and debugging
- **Configuration**: Minimal pipeline combinations
- **Pipeline Count**: ~2 pipelines
- **Use Case**: Quick validation during development

```bash
poetry run python -m giflab experiment --preset quick-test
```

## Custom Slot Configuration

### Variable Slot Options

Use `--variable-slot` to specify which slots should test multiple algorithms:

```bash
# Test all algorithms in a slot
--variable-slot frame=*
--variable-slot color=*
--variable-slot lossy=*

# Test specific algorithms only
--variable-slot frame=animately-frame,ffmpeg-frame
--variable-slot color=ffmpeg-color,gifsicle-color

# Multiple variable slots
--variable-slot frame=* --variable-slot color=*
```

### Lock Slot Options

Use `--lock-slot` to fix specific algorithms:

```bash
# Lock to specific implementations
--lock-slot frame=animately-frame
--lock-slot color=ffmpeg-color
--lock-slot lossy=animately-advanced-lossy

# Multiple locked slots
--lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

### Parameter Configuration

Use `--slot-params` to specify algorithm parameters:

```bash
# Integer parameters
--slot-params color=colors:32
--slot-params lossy=level:60

# List parameters (for variable slots)
--slot-params frame=ratios:[1.0,0.8,0.5,0.2]
--slot-params color=colors:[128,64,32,16]

# Multiple parameter specifications
--slot-params color=colors:32 --slot-params lossy=level:40
```

## Advanced Usage

### Combining with Traditional Options

Targeted presets can be combined with other experiment options:

```bash
# Use preset with custom quality threshold
poetry run python -m giflab experiment --preset frame-focus --quality-threshold 0.1

# Use preset with specific output directory
poetry run python -m giflab experiment --preset color-optimization --output-dir color_study_results

# Use preset with GPU acceleration
poetry run python -m giflab experiment --preset tool-comparison-baseline --use-gpu
```

### Programmatic Usage

You can also use targeted presets programmatically:

```python
from giflab.experimental.runner import ExperimentalRunner
from pathlib import Path

# Set up experiment
runner = ExperimentalRunner(
    output_dir=Path("results"),
    use_cache=True
)

# List available presets
presets = runner.list_available_presets()
print(f"Available presets: {list(presets.keys())}")

# Generate targeted pipelines
pipelines = runner.generate_targeted_pipelines("frame-focus")
print(f"Generated {len(pipelines)} targeted pipelines")

# Run experiment
result = runner.run_targeted_experiment(
    preset_id="frame-focus",
    quality_threshold=0.05
)

print(f"Completed {result.total_jobs_run} pipeline tests")
```

### Custom Preset Creation

For advanced users, you can create custom presets programmatically:

```python
from giflab.experimental.targeted_presets import ExperimentPreset, SlotConfiguration, PRESET_REGISTRY

# Create custom preset
custom_preset = ExperimentPreset(
    name="My Custom Study",
    description="Custom frame algorithm comparison",
    frame_slot=SlotConfiguration(
        type="variable",
        scope=["animately-frame", "ffmpeg-frame"],
        parameters={"ratios": [1.0, 0.7, 0.4]}
    ),
    color_slot=SlotConfiguration(
        type="locked",
        implementation="ffmpeg-color",
        parameters={"colors": 64}
    ),
    lossy_slot=SlotConfiguration(
        type="locked",
        implementation="none-lossy",
        parameters={"level": 0}
    ),
    max_combinations=50
)

# Register the preset
PRESET_REGISTRY.register("my-custom-study", custom_preset)

# Use the custom preset
runner = ExperimentalRunner()
pipelines = runner.generate_targeted_pipelines("my-custom-study")
```

## Performance Benefits

The targeted preset system provides significant efficiency gains:

| Approach | Pipelines Generated | Efficiency Gain |
|----------|-------------------|-----------------|
| Traditional (generate_all + quick sampling) | 935 → 46 used | 95.1% waste |
| Traditional (generate_all + representative) | 935 → 140 used | 85.0% waste |
| **Targeted: frame-focus** | **5 generated** | **99.5% efficient** |
| **Targeted: color-optimization** | **17 generated** | **98.2% efficient** |
| **Targeted: tool-comparison** | **64 generated** | **93.2% efficient** |

### Benefits:
- **Faster startup**: No sampling phase required
- **Lower memory usage**: Generate only what's needed
- **Clearer intent**: Configuration directly expresses research goals
- **Better resource utilization**: Focus computation on relevant comparisons

## Troubleshooting

### Common Issues

#### "Unknown preset: preset-name"
Make sure the preset name is correct. List available presets:
```bash
poetry run python -m giflab experiment --list-presets
```

#### "Invalid slot name: slot-name"
Only `frame`, `color`, and `lossy` are valid slot names.

#### "At least one slot must be variable"
You cannot lock all three slots. At least one slot must be variable to create meaningful comparisons.

#### "Invalid parameter format"
Parameter specifications must use the format `param:value`. For lists, use `param:[value1,value2,value3]`.

### Tool Name Issues

If you get errors about unknown tool implementations, check available tools:

```python
from giflab.capability_registry import tools_for

# List available tools for each slot
frame_tools = [tool.NAME for tool in tools_for("frame_reduction")]
color_tools = [tool.NAME for tool in tools_for("color_reduction")]
lossy_tools = [tool.NAME for tool in tools_for("lossy_compression")]

print(f"Frame tools: {frame_tools}")
print(f"Color tools: {color_tools}")
print(f"Lossy tools: {lossy_tools}")
```

### Performance Issues

If experiments are running slowly:

1. **Use smaller presets**: Start with `quick-test` for development
2. **Limit combinations**: Use `max_combinations` parameter in custom presets
3. **Enable caching**: Use `--use-cache` flag
4. **GPU acceleration**: Use `--use-gpu` if available

### Getting Help

- Use `--help` to see all available options
- Check the troubleshooting section in the main documentation
- Review example commands in this guide
- Examine the built-in presets for configuration patterns

## Migration from Traditional Sampling

If you're currently using the traditional `generate_all_pipelines() + sampling` approach, here's how to migrate:

### Before (Traditional)
```bash
# Old approach: generate all, then sample
poetry run python -m giflab experiment --sampling quick --max-combinations 50
```

### After (Targeted)
```bash
# New approach: generate only what's needed
poetry run python -m giflab experiment --preset quick-test

# Or custom configuration
poetry run python -m giflab experiment \
  --variable-slot frame=* \
  --lock-slot color=ffmpeg-color \
  --lock-slot lossy=animately-advanced-lossy
```

The targeted approach provides the same focused testing but with much better efficiency and clearer configuration.

## Next Steps

1. **Start simple**: Try the built-in presets first
2. **Understand your needs**: Identify which algorithm dimension you want to study
3. **Use custom slots**: Create specific configurations for your research
4. **Measure results**: Compare efficiency gains vs traditional approaches
5. **Create custom presets**: Define reusable configurations for common studies

For more advanced usage and API details, see the technical documentation and CLI reference guide.