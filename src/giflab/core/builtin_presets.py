"""Built-in presets for common research scenarios.

This module provides pre-configured presets for typical GIF compression
research studies, implementing the use cases defined in Phase 1 analysis.

The built-in presets are automatically registered when this module is imported,
providing immediate access to common research configurations without requiring
manual preset definition.

Available Preset Categories:
    - Research Presets: Single-dimension algorithm comparisons
    - Baseline Presets: Multi-dimensional comprehensive comparisons
    - Specialized Presets: Focused studies for specific optimization scenarios
    - Development Presets: Minimal configurations for testing and debugging

Efficiency Gains:
    Most presets achieve 93-99% efficiency improvements over traditional
    generate_all_pipelines() + sampling approaches by generating only
    the specific pipeline combinations needed for focused studies.

Usage:
    # Access presets via global registry
    from giflab.core.targeted_presets import PRESET_REGISTRY
    preset = PRESET_REGISTRY.get("frame-focus")
    
    # Or use via CLI
    poetry run python -m giflab run --preset frame-focus
"""

from .targeted_presets import PRESET_REGISTRY, ExperimentPreset, SlotConfiguration

# Built-in preset definitions based on research use cases
#
# Each preset is designed for a specific research question and optimized for
# maximum efficiency by generating only relevant pipeline combinations.

TARGETED_PRESETS = {
    # Research Presets: Single-dimension algorithm comparisons
    "frame-focus": ExperimentPreset(
        name="Frame Removal Focus Study",
        description="Compare all frame reduction algorithms with locked color and lossy settings",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=["*"],  # All available frame algorithms
            parameters={"ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]},
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        ),
        lossy_slot=SlotConfiguration(
            type="locked",
            implementation="animately-advanced-lossy",
            parameters={"level": 40},
        ),
        custom_sampling="quick",
        max_combinations=100,
        tags=["research", "frame-optimization", "comparison"],
        author="GifLab Research Team",
    ),
    "color-optimization": ExperimentPreset(
        # Research Goal: Compare color quantization algorithms and dithering methods
        # Expected Results: Optimal color algorithm per palette size, dithering effectiveness rankings
        name="Color Optimization Study",
        description="Compare color reduction techniques and dithering methods across all variants",
        frame_slot=SlotConfiguration(
            type="locked", implementation="animately-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="variable",
            scope=["*"],  # All 17 color reduction variants
            parameters={"colors": [256, 128, 64, 32, 16, 8]},
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="representative",
        max_combinations=200,
        tags=["research", "color-optimization", "dithering"],
        author="GifLab Research Team",
    ),
    "lossy-quality-sweep": ExperimentPreset(
        name="Lossy Quality Sweep",
        description="Evaluate lossy compression effectiveness across different engines",
        frame_slot=SlotConfiguration(
            type="locked", implementation="none-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 64}
        ),
        lossy_slot=SlotConfiguration(
            type="variable",
            scope=["*"],  # All lossy compression tools
            parameters={"levels": [0, 20, 40, 60, 80, 100, 120, 140, 160]},
        ),
        custom_sampling="factorial",
        max_combinations=150,
        tags=["research", "lossy-optimization", "quality-analysis"],
        author="GifLab Research Team",
    ),
    "tool-comparison-baseline": ExperimentPreset(
        name="Tool Comparison Baseline",
        description="Fair engine comparison with conservative parameter sets across complete pipelines",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=[
                "animately-frame",
                "ffmpeg-frame",
                "gifsicle-frame",
                "imagemagick-frame",
            ],
            parameters={"ratios": [1.0, 0.8, 0.5]},
        ),
        color_slot=SlotConfiguration(
            type="variable",
            scope=[
                "animately-color",
                "ffmpeg-color",
                "gifsicle-color",
                "imagemagick-color",
            ],
            parameters={"colors": [64, 32, 16]},
        ),
        lossy_slot=SlotConfiguration(
            type="variable",
            scope=[
                "animately-advanced-lossy",
                "ffmpeg-lossy",
                "gifsicle-lossy",
                "imagemagick-lossy",
            ],
            parameters={"levels": [0, 40, 120]},
        ),
        custom_sampling="representative",
        max_combinations=200,
        tags=["baseline", "comparison", "comprehensive"],
        author="GifLab Research Team",
    ),
    "quick-test": ExperimentPreset(
        name="Quick Development Test",
        description="Fast preset for development and debugging with minimal pipeline combinations",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=["animately-frame", "gifsicle-frame"],
            parameters={"ratios": [1.0, 0.5]},
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        ),
        lossy_slot=SlotConfiguration(
            type="locked",
            implementation="animately-advanced-lossy",
            parameters={"level": 40},
        ),
        custom_sampling="quick",
        max_combinations=10,
        tags=["development", "testing", "debug"],
        author="GifLab Research Team",
    ),
    "dithering-focus": ExperimentPreset(
        name="Dithering Algorithm Focus",
        description="Compare dithering methods using FFmpeg and ImageMagick variants",
        frame_slot=SlotConfiguration(
            type="locked", implementation="none-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="variable",
            scope=[
                "ffmpeg-color-floyd",
                "ffmpeg-color-sierra2",
                "ffmpeg-color-bayer0",
                "ffmpeg-color-bayer2",
                "imagemagick-color-floyd",
                "imagemagick-color-riemersma",
            ],
            parameters={"colors": [64, 32, 16]},
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="factorial",
        max_combinations=80,
        tags=["research", "dithering", "specialized"],
        author="GifLab Research Team",
    ),
    "png-optimization": ExperimentPreset(
        name="PNG Sequence Optimization",
        description="Focus on gifski and animately-advanced PNG sequence workflows",
        frame_slot=SlotConfiguration(
            type="locked",
            implementation="ffmpeg-frame",  # Good for PNG extraction
            parameters={"ratio": 0.8},
        ),
        color_slot=SlotConfiguration(
            type="variable",
            scope=["ffmpeg-color", "imagemagick-color"],
            parameters={"colors": [128, 64, 32]},
        ),
        lossy_slot=SlotConfiguration(
            type="variable",
            scope=[
                "gifski-lossy",
                "animately-advanced-lossy",
            ],  # PNG sequence optimized
            parameters={"levels": [60, 80, 100]},
        ),
        custom_sampling="representative",
        max_combinations=60,
        tags=["specialized", "png-optimization", "advanced"],
        author="GifLab Research Team",
    ),
    "custom-gif-frame-study": ExperimentPreset(
        name="Custom GIF Frame Study",
        description="Compare frame reduction algorithms with custom GIF inputs - algorithm comparison focused",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=["*"],  # All frame reduction algorithms
            parameters={"ratios": [1.0, 0.8, 0.6, 0.4, 0.2]},
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="factorial",
        max_combinations=100,
        tags=["research", "frame-analysis", "custom-gifs"],
        author="GifLab Research Team",
    ),
    "frame-parameter-sweep": ExperimentPreset(
        name="Frame Parameter Sweep",
        description="Sweep frame reduction parameters with fixed algorithm - parameter sweep focused",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=["animately-frame"],  # Fixed to one algorithm
            parameters={
                "ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            },  # Parameter sweep
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="full",
        max_combinations=50,
        tags=["research", "parameter-sweep", "frame-analysis"],
        author="GifLab Research Team",
    ),
    "color-parameter-sweep": ExperimentPreset(
        name="Color Parameter Sweep",
        description="Sweep color reduction parameters with fixed algorithm - parameter sweep focused",
        frame_slot=SlotConfiguration(
            type="locked", implementation="none-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="variable",
            scope=["ffmpeg-color"],  # Fixed to one algorithm
            parameters={
                "colors": [256, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4]
            },  # Parameter sweep
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="full",
        max_combinations=50,
        tags=["research", "parameter-sweep", "color-analysis"],
        author="GifLab Research Team",
    ),
    "lossy-parameter-sweep": ExperimentPreset(
        name="Lossy Parameter Sweep",
        description="Sweep lossy compression parameters with fixed algorithm - parameter sweep focused",
        frame_slot=SlotConfiguration(
            type="locked", implementation="none-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 64}
        ),
        lossy_slot=SlotConfiguration(
            type="variable",
            scope=["ffmpeg-lossy"],  # Fixed to one algorithm
            parameters={
                "levels": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]
            },  # Parameter sweep
        ),
        custom_sampling="full",
        max_combinations=50,
        tags=["research", "parameter-sweep", "lossy-analysis"],
        author="GifLab Research Team",
    ),
    # Wrapper Validation Presets: Test individual tool wrappers in isolation
    #
    # Pattern for creating wrapper validation presets for any tool:
    # 1. Set all three slots (frame, color, lossy) as "variable" 
    # 2. Each slot scope includes both the tool-specific implementation and the "none-" equivalent
    #    - frame_slot: ["<tool>-frame", "none-frame"] 
    #    - color_slot: ["<tool>-color", "none-color"]
    #    - lossy_slot: ["<tool>-lossy", "none-lossy"]
    # 3. Use factorial sampling to generate all 8 combinations (2³ = 8 pipelines):
    #    - Full tool pipeline: tool-frame → tool-color → tool-lossy
    #    - Frame + Color only: tool-frame → tool-color → none-lossy
    #    - Frame + Lossy only: tool-frame → none-color → tool-lossy  
    #    - Frame only: tool-frame → none-color → none-lossy
    #    - Color + Lossy only: none-frame → tool-color → tool-lossy
    #    - Color only: none-frame → tool-color → none-lossy
    #    - Lossy only: none-frame → none-color → tool-lossy
    #    - Baseline (no processing): none-frame → none-color → none-lossy
    #
    # This pattern isolates each wrapper component for validation while testing
    # all meaningful combinations to ensure components work independently and together.
    #
    # Future tool validation presets:
    # - "ffmpeg-wrapper-validation"
    # - "imagemagick-wrapper-validation" 
    # - "animately-wrapper-validation"
    "gifsicle-frame-validation": ExperimentPreset(
        name="GIFsicle Frame Wrapper Validation",
        description="Test gifsicle-frame wrapper in pure isolation",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=["gifsicle-frame"],
            parameters={"ratios": [1.0, 0.8, 0.6, 0.4, 0.2]},
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="none-color", parameters={}
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="full",
        max_combinations=20,
        tags=["validation", "gifsicle", "frame-wrapper", "isolation"],
        author="GifLab Research Team",
    ),
    "gifsicle-color-validation": ExperimentPreset(
        name="GIFsicle Color Wrapper Validation",
        description="Test gifsicle-color wrapper in pure isolation",
        frame_slot=SlotConfiguration(
            type="locked", implementation="none-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="variable",
            scope=["gifsicle-color"],
            parameters={"colors": [64, 32, 16, 8]},
        ),
        lossy_slot=SlotConfiguration(
            type="locked", implementation="none-lossy", parameters={"level": 0}
        ),
        custom_sampling="full",
        max_combinations=20,
        tags=["validation", "gifsicle", "color-wrapper", "isolation"],
        author="GifLab Research Team",
    ),
    "gifsicle-lossy-validation": ExperimentPreset(
        name="GIFsicle Lossy Wrapper Validation", 
        description="Test gifsicle-lossy wrapper in pure isolation",
        frame_slot=SlotConfiguration(
            type="locked", implementation="none-frame", parameters={"ratio": 1.0}
        ),
        color_slot=SlotConfiguration(
            type="locked", implementation="none-color", parameters={}
        ),
        lossy_slot=SlotConfiguration(
            type="variable",
            scope=["gifsicle-lossy"],
            parameters={"levels": [0, 40, 80, 120]},
        ),
        custom_sampling="full",
        max_combinations=20,
        tags=["validation", "gifsicle", "lossy-wrapper", "isolation"],
        author="GifLab Research Team",
    ),
}


def register_builtin_presets() -> None:
    """Register all built-in presets with the global registry."""
    for preset_id, preset in TARGETED_PRESETS.items():
        PRESET_REGISTRY.register(preset_id, preset)


# Auto-register built-in presets when module is imported
register_builtin_presets()
