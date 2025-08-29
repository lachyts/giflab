# Performance Tuning: Targeted Presets

This document provides comprehensive guidelines for optimizing performance when using the targeted presets system.

## Performance Overview

The targeted presets system is optimized for focused testing:

- **Minimal pipeline generation** overhead
- **Fast experiment startup** times
- **70-90% lower** memory usage during generation phase
- **Direct resource targeting** eliminating waste on irrelevant combinations

## Optimization Strategies

### 1. Preset Selection Optimization

#### Choose Appropriate Preset Scope

**Best Practice**: Match preset scope to research requirements

```python
# For algorithm comparison: Use single-dimension presets
--preset frame-focus           # 5 pipelines (99.5% efficiency)
--preset color-optimization    # 17 pipelines (98.2% efficiency)  
--preset lossy-quality-sweep   # 11 pipelines (98.8% efficiency)

# For comprehensive analysis: Use multi-dimension presets
--preset tool-comparison-baseline  # 64 pipelines (93.2% efficiency)

# For development: Use minimal presets
--preset quick-test               # 2 pipelines (99.8% efficiency)
```

**Pipeline Count Planning**:
```bash
# Check estimated pipeline count before execution
poetry run python -c "
from giflab.experimental.targeted_presets import PRESET_REGISTRY
from giflab.experimental.targeted_generator import TargetedPipelineGenerator

preset = PRESET_REGISTRY.get('frame-focus')
generator = TargetedPipelineGenerator()
validation = generator.validate_preset_feasibility(preset)
print(f'Estimated pipelines: {validation[\"estimated_pipelines\"]}')
print(f'Efficiency gain: {validation[\"efficiency_gain\"]:.1%}')
"
```

#### Custom Scope Optimization

**Specific Tool Lists vs Wildcards**:
```python
# More efficient: Specific tools (if you know what you want)
--variable-slot frame=animately-frame,ffmpeg-frame  # Only 2 tools
--variable-slot color=ffmpeg-color,gifsicle-color   # Only 2 tools

# Less efficient: Wildcard expansion (tests everything)  
--variable-slot frame=*     # All available frame tools (~5 tools)
--variable-slot color=*     # All available color tools (~17 tools)
```

**Parameter Range Optimization**:
```python
# Focused parameter ranges for faster execution
--slot-params frame=ratios:[1.0,0.8,0.5]           # 3 ratios
--slot-params color=colors:[64,32]                  # 2 color counts
--slot-params lossy=levels:[0,40,80]                # 3 levels

# vs broader ranges (slower but more comprehensive)
--slot-params frame=ratios:[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]  # 9 ratios
--slot-params color=colors:[256,128,64,32,16,8]                     # 6 counts
--slot-params lossy=levels:[0,20,40,60,80,100,120,140,160]          # 9 levels
```

### 2. Pipeline Count Management

#### Use max_combinations Limits

**Prevent Pipeline Explosion**:
```python
# Custom preset with hard limit
custom_preset = ExperimentPreset(
    name="Limited Multi-Dimension Study",
    description="Comprehensive but bounded experiment",
    frame_slot=SlotConfiguration(type="variable", scope=["*"]),      # ~5 tools
    color_slot=SlotConfiguration(type="variable", scope=["*"]),      # ~17 tools  
    lossy_slot=SlotConfiguration(type="variable", scope=["*"]),      # ~11 tools
    max_combinations=100  # Limit: 5×17×11=hundreds → 100 (significant reduction)
)
```

**CLI Pipeline Limiting**:
```bash
# Create custom preset with limits via CLI
poetry run python -c "
import sys
from giflab.experimental.targeted_presets import ExperimentPreset, SlotConfiguration

preset = ExperimentPreset(
    name='Limited Study',
    description='Performance-optimized study',
    frame_slot=SlotConfiguration(type='variable', scope=['*']),
    color_slot=SlotConfiguration(type='variable', scope=['*']),
    lossy_slot=SlotConfiguration(type='locked', implementation='none-lossy'),
    max_combinations=50
)

from giflab.experimental.targeted_generator import TargetedPipelineGenerator
generator = TargetedPipelineGenerator()
pipelines = generator.generate_targeted_pipelines(preset)
print(f'Generated {len(pipelines)} pipelines (limited from potential {5*17})')
"
```

#### Progressive Preset Approach

**Start Small, Scale Up**:
```bash
# Phase 1: Quick validation (2 pipelines)
poetry run python -m giflab run --preset quick-test

# Phase 2: Single dimension focus (5-17 pipelines)
poetry run python -m giflab run --preset frame-focus

# Phase 3: Multi-dimension if needed (64+ pipelines)
poetry run python -m giflab run --preset tool-comparison-baseline
```

### 3. Memory Optimization

#### Enable Caching

**⚠️ Note**: As of recent updates, **caching is disabled by default** (`use_cache=False`) to ensure predictable behavior and avoid stale cache issues during development.

**Result Caching**:
```bash
# Enable caching for repeated experiments (RECOMMENDED for production)
poetry run python -m giflab run --preset frame-focus --use-cache

# Cache benefits compound over multiple runs
poetry run python -m giflab run --preset color-optimization --use-cache
poetry run python -m giflab run --preset lossy-quality-sweep --use-cache
```

**When to Enable Caching**:
- **Production runs**: Always use `--use-cache` for large experiments
- **Development**: Leave caching off to ensure fresh results
- **Repeated testing**: Enable caching when running the same presets multiple times

**Tool Resolution Caching**:
```python
# Tool caching is automatic in TargetedPipelineGenerator
generator = TargetedPipelineGenerator()  # Initializes _tool_cache
# First preset resolves tools and caches them
pipelines1 = generator.generate_targeted_pipelines(preset1)  
# Second preset reuses cached tool information
pipelines2 = generator.generate_targeted_pipelines(preset2)  # Faster
```

#### Use Targeted GIF Sets

**Reduce Test Data Size**:
```bash
# Use smaller, focused GIF set for testing
poetry run python -m giflab run --preset frame-focus --use-targeted-gifs

# Benefits:
# - Faster I/O operations
# - Lower memory usage for image processing
# - Reduced disk space requirements
# - Faster overall execution
```

#### Memory-Aware Preset Design

**Memory-Efficient Patterns**:
```python
# Memory-efficient: Single variable dimension
memory_optimized = ExperimentPreset(
    name="Memory Efficient Study", 
    description="Low memory usage configuration",
    frame_slot=SlotConfiguration(type="variable", scope=["animately-frame", "ffmpeg-frame"]),  # 2 tools
    color_slot=SlotConfiguration(type="locked", implementation="ffmpeg-color"),  # Fixed
    lossy_slot=SlotConfiguration(type="locked", implementation="none-lossy"),    # Fixed
    max_combinations=10
)
# Result: 2 pipelines, minimal memory usage

# Memory-intensive: Multiple variable dimensions  
memory_intensive = ExperimentPreset(
    name="Memory Intensive Study",
    description="High memory usage configuration", 
    frame_slot=SlotConfiguration(type="variable", scope=["*"]),      # ~5 tools
    color_slot=SlotConfiguration(type="variable", scope=["*"]),      # ~17 tools
    lossy_slot=SlotConfiguration(type="variable", scope=["*"])       # ~11 tools
)
# Result: hundreds of pipelines, high memory usage
```

### 4. Execution Speed Optimization

#### GPU Acceleration

**Enable GPU When Available**:
```bash
# GPU acceleration for supported operations
poetry run python -m giflab run --preset frame-focus --use-gpu

# Check GPU availability first
poetry run python -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name()}')
"
```

#### Parallel Execution Patterns

**Concurrent Preset Execution**:
```python
# Execute multiple presets in parallel
import concurrent.futures
from giflab.experimental.runner import GifLabRunner

def run_preset_experiment(preset_id, output_dir):
    runner = GifLabRunner(output_dir=output_dir, use_cache=True)
    return runner.run_targeted_experiment(preset_id, use_targeted_gifs=True)

presets = ["frame-focus", "color-optimization", "lossy-quality-sweep"]
results = {}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    future_to_preset = {
        executor.submit(run_preset_experiment, preset, f"results_{preset}"): preset 
        for preset in presets
    }
    
    for future in concurrent.futures.as_completed(future_to_preset):
        preset = future_to_preset[future]
        results[preset] = future.result()

print(f"Completed {len(results)} experiments in parallel")
```

#### Quality Threshold Optimization

**Balance Quality vs Speed**:
```bash
# Strict quality (slower, higher precision)
poetry run python -m giflab run --preset frame-focus --quality-threshold 0.02

# Balanced quality (good balance)
poetry run python -m giflab run --preset frame-focus --quality-threshold 0.05

# Permissive quality (faster, lower precision)  
poetry run python -m giflab run --preset frame-focus --quality-threshold 0.1

# Development/testing (fastest)
poetry run python -m giflab run --preset quick-test --quality-threshold 0.2
```

### 5. Tool Selection Optimization

#### High-Performance Tool Combinations

**Fast Tool Combinations**:
```python
# Speed-optimized preset
speed_preset = ExperimentPreset(
    name="Speed Optimized Study",
    description="Fastest execution configuration",
    frame_slot=SlotConfiguration(
        type="variable", 
        scope=["animately-frame", "gifsicle-frame"],  # Fast frame tools
        parameters={"ratios": [1.0, 0.5]}  # Limited ratios
    ),
    color_slot=SlotConfiguration(
        type="locked",
        implementation="ffmpeg-color",  # Fast color quantization
        parameters={"colors": 32}       # Conservative color count
    ),
    lossy_slot=SlotConfiguration(
        type="locked", 
        implementation="none-lossy",    # No lossy processing (fastest)
        parameters={"level": 0}
    ),
    max_combinations=5
)
```

**Quality-Focused Tool Combinations**:
```python
# Quality-optimized preset (slower but better results)
quality_preset = ExperimentPreset(
    name="Quality Optimized Study",
    description="Highest quality configuration",
    frame_slot=SlotConfiguration(
        type="locked",
        implementation="animately-frame",  # High-quality frame processing
        parameters={"ratio": 0.9}          # Minimal frame reduction
    ),
    color_slot=SlotConfiguration(
        type="variable",
        scope=["ffmpeg-color-floyd", "imagemagick-color-floyd"],  # High-quality dithering
        parameters={"colors": [128, 64]}   # Higher color counts
    ),
    lossy_slot=SlotConfiguration(
        type="variable",
        scope=["gifski-lossy", "animately-advanced-lossy"],  # Quality-focused tools
        parameters={"levels": [80, 100]}   # High quality levels
    )
)
```

### 6. Parameter Optimization

#### Efficient Parameter Ranges

**Algorithm Comparison Parameters**:
```python
# For algorithm comparison: Use fixed parameters
algorithm_comparison_params = {
    "frame": {"ratios": [0.8]},      # Single representative ratio
    "color": {"colors": [32]},       # Single representative count  
    "lossy": {"levels": [40]}        # Single representative level
}

# Focus on algorithm differences, not parameter variations
```

**Parameter Sweep Parameters**:
```python
# For parameter optimization: Use comprehensive ranges
parameter_sweep_params = {
    "frame": {"ratios": [1.0, 0.8, 0.6, 0.4, 0.2]},        # Full range
    "color": {"colors": [256, 128, 64, 32, 16, 8]},         # Full range
    "lossy": {"levels": [0, 25, 50, 75, 100, 125, 150]}     # Full range
}

# Test parameter impact with fixed algorithm
```

#### Parameter Range Sizing

**Small Ranges (Fast)**:
```bash
--slot-params frame=ratios:[1.0,0.5]          # 2 values
--slot-params color=colors:[64,32]             # 2 values  
--slot-params lossy=levels:[0,80]              # 2 values
# Pipeline multiplication: Tools × 2 × 2 × 2 = Tools × 8
```

**Medium Ranges (Balanced)**:
```bash
--slot-params frame=ratios:[1.0,0.8,0.5]      # 3 values
--slot-params color=colors:[128,64,32]         # 3 values
--slot-params lossy=levels:[0,40,80]           # 3 values  
# Pipeline multiplication: Tools × 3 × 3 × 3 = Tools × 27
```

**Large Ranges (Comprehensive)**:
```bash
--slot-params frame=ratios:[1.0,0.9,0.8,0.7,0.6,0.5]    # 6 values
--slot-params color=colors:[256,128,64,32,16,8]           # 6 values
--slot-params lossy=levels:[0,20,40,60,80,100]           # 6 values
# Pipeline multiplication: Tools × 6 × 6 × 6 = Tools × 216
```

## Performance Monitoring

### 1. Built-in Performance Metrics

#### Pipeline Count Tracking

**Pipeline Analysis**:
```python
from giflab.experimental.targeted_generator import TargetedPipelineGenerator

preset = PRESET_REGISTRY.get("frame-focus")
generator = TargetedPipelineGenerator()
targeted_pipelines = generator.generate_targeted_pipelines(preset)
targeted_count = len(targeted_pipelines)

print(f"Generated {targeted_count} targeted pipelines for {preset.name}")
```

#### Execution Time Profiling

**Generation Time Measurement**:
```python
import time
from giflab.core.runner import GifLabRunner

runner = GifLabRunner()

# Profile targeted pipeline generation
start_time = time.time()
targeted_pipelines = runner.generate_targeted_pipelines("frame-focus")
generation_time = time.time() - start_time

print(f"Pipeline generation time: {generation_time:.3f}s")
print(f"Average time per pipeline: {generation_time/len(targeted_pipelines):.4f}s")
```

### 2. Performance Benchmarking

#### Preset Performance Comparison

**Benchmark Multiple Presets**:
```python
import time
from pathlib import Path

def benchmark_preset(preset_id: str, iterations: int = 3) -> dict:
    """Benchmark preset performance across multiple runs."""
    times = []
    pipeline_counts = []
    
    for i in range(iterations):
        start_time = time.time()
        runner = GifLabRunner(output_dir=Path(f"benchmark_{preset_id}_{i}"))
        pipelines = runner.generate_targeted_pipelines(preset_id)
        generation_time = time.time() - start_time
        
        times.append(generation_time)
        pipeline_counts.append(len(pipelines))
    
    return {
        "preset_id": preset_id,
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times), 
        "pipeline_count": pipeline_counts[0],  # Should be consistent
        "consistency": len(set(pipeline_counts)) == 1
    }

# Benchmark key presets
presets = ["quick-test", "frame-focus", "color-optimization", "tool-comparison-baseline"]
results = [benchmark_preset(preset) for preset in presets]

for result in sorted(results, key=lambda x: x["pipeline_count"]):
    print(f"{result['preset_id']}: {result['pipeline_count']} pipelines, "
          f"{result['avg_time']:.3f}s avg, consistent: {result['consistency']}")
```

### 3. Resource Usage Monitoring

#### Memory Usage Profiling

**Memory Tracking**:
```python
import psutil
import os
from giflab.experimental.runner import GifLabRunner

def profile_memory_usage(preset_id: str) -> dict:
    """Profile memory usage during preset execution."""
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute preset
    runner = GifLabRunner(use_cache=False)  # Explicitly disable for isolated testing
    pipelines = runner.generate_targeted_pipelines(preset_id)
    
    # Peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    return {
        "preset_id": preset_id,
        "pipeline_count": len(pipelines),
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory, 
        "memory_increase_mb": memory_increase,
        "memory_per_pipeline_mb": memory_increase / len(pipelines) if pipelines else 0
    }

# Profile memory usage across presets
presets = ["quick-test", "frame-focus", "color-optimization"]
for preset in presets:
    profile = profile_memory_usage(preset)
    print(f"{profile['preset_id']}: {profile['memory_increase_mb']:.1f}MB "
          f"({profile['memory_per_pipeline_mb']:.3f}MB per pipeline)")
```

## Performance Best Practices

### 1. Development Workflow Optimization

**Iterative Development Pattern**:
```bash
# Phase 1: Quick validation
poetry run python -m giflab run --preset quick-test --use-cache --use-targeted-gifs

# Phase 2: Focused testing  
poetry run python -m giflab run --preset frame-focus --use-cache --use-targeted-gifs

# Phase 3: Production run
poetry run python -m giflab run --preset frame-focus --use-cache
```

**Configuration Testing**:
```python
# Test custom configuration efficiency before full run
def test_configuration_efficiency(preset: ExperimentPreset) -> bool:
    """Test if configuration is reasonable before execution."""
    generator = TargetedPipelineGenerator()
    validation = generator.validate_preset_feasibility(preset)
    
    if not validation["valid"]:
        print(f"Configuration invalid: {validation['errors']}")
        return False
    
    pipeline_count = validation["estimated_pipelines"]
    efficiency_gain = validation["efficiency_gain"]
    
    print(f"Estimated pipelines: {pipeline_count}")
    print(f"Efficiency gain: {efficiency_gain:.1%}")
    
    # Performance thresholds
    if pipeline_count > 200:
        print("Warning: High pipeline count may impact performance")
    if efficiency_gain < 0.8:
        print("Warning: Low efficiency optimization achieved")
    
    return pipeline_count <= 500 and efficiency_gain >= 0.5
```

### 2. Production Optimization

**Batch Execution Strategy**:
```python
# Execute related presets in batches
batch_1_presets = ["frame-focus", "color-optimization", "lossy-quality-sweep"]  # Research
batch_2_presets = ["tool-comparison-baseline"]  # Comprehensive
batch_3_presets = ["dithering-focus", "png-optimization"]  # Specialized

def execute_preset_batch(presets: list, base_output_dir: str):
    """Execute preset batch with optimized resource usage."""
    for preset_id in presets:
        output_dir = f"{base_output_dir}/{preset_id}"
        runner = GifLabRunner(
            output_dir=Path(output_dir),
            use_cache=True,  # Enable caching for production runs
            use_gpu=True     # GPU acceleration
        )
        
        result = runner.run_targeted_experiment(
            preset_id=preset_id,
            quality_threshold=0.05,
            use_targeted_gifs=False  # Full dataset for production
        )
        
        print(f"Completed {preset_id}: {result.total_jobs_run} jobs")
```

**Resource-Aware Scheduling**:
```python
def schedule_experiments_by_resource_usage():
    """Schedule experiments based on resource requirements."""
    
    # Low resource experiments (can run concurrently)
    low_resource = ["quick-test", "frame-focus", "color-optimization"]
    
    # Medium resource experiments (run individually)  
    medium_resource = ["lossy-quality-sweep", "dithering-focus", "png-optimization"]
    
    # High resource experiments (run with full system resources)
    high_resource = ["tool-comparison-baseline"]
    
    # Execute low resource in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_experiment, preset) for preset in low_resource]
        concurrent.futures.wait(futures)
    
    # Execute medium resource sequentially
    for preset in medium_resource:
        run_experiment(preset)
    
    # Execute high resource with full system
    for preset in high_resource:
        run_experiment_full_resources(preset)
```

This performance tuning guide enables users to optimize their targeted experiment workflows for maximum efficiency while maintaining result quality.