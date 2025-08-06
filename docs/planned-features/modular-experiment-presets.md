---
name: Modular Experiment Presets System
priority: medium
size: medium
status: planning
owner: @lachlants
issue: N/A - Feature request for targeted pipeline creation
---

# Modular Experiment Presets System

**Priority:** Medium  
**Estimated Effort:** 4-5 days  
**Target Release:** Next minor version

## Phase Progress Overview

### Phase 1: Planning & Requirements Analysis ⏳ PLANNED
**Progress:** 0% Complete  
**Current Focus:** Problem definition and solution architecture

### Phase 2: Architecture Design ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** Pipeline creation system design

### Phase 3: Core Implementation ⏳ PLANNED  
**Progress:** 0% Complete
**Current Focus:** Preset system and pipeline generation

### Phase 4: Interface Development ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** CLI integration and user experience

### Phase 5: Testing & Validation ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** Comprehensive testing across all presets

### Phase 6: Documentation & Deployment ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** User guides and production deployment

---

## Overview

This document outlines a targeted pipeline creation system for GIF compression experiments. Instead of generating all possible pipeline combinations then sampling subsets, this system creates only the specific pipelines needed by defining variable scopes and locked implementations for focused research studies.

## Problem Statement

The current experiment system uses `generate_all_pipelines()` which creates ALL possible combinations of frame × color × lossy reduction parameters across all available tools, then relies on sampling strategies to select subsets. For focused research studies, this approach is highly inefficient:

1. **Wasteful Generation**: `generate_all_pipelines()` creates hundreds of pipeline combinations that are immediately discarded by sampling
2. **Unclear Research Intent**: Cannot directly specify "test all frame algorithms with specific color/lossy implementations"  
3. **Sampling Overhead**: Must generate everything first, then intelligently sample, rather than creating only what's needed
4. **Resource Inefficiency**: Computational resources wasted on generating irrelevant pipeline combinations

## Proposed Solution: Targeted Pipeline Generation

### Core Concept: Replace `generate_all_pipelines()` with `generate_targeted_pipelines()`

Instead of generating ALL combinations then sampling, directly create only the pipeline combinations needed for specific research studies:

**Current Approach:**
```python
# In dynamic_pipeline.py
def generate_all_pipelines() -> list[Pipeline]:
    """Return *every* valid 3-slot pipeline (may be hundreds)."""
    frame_tools = tools_for("frame_reduction")
    color_tools = tools_for("color_reduction") 
    lossy_tools = tools_for("lossy_compression")
    
    for trio in itertools.product(frame_tools, color_tools, lossy_tools):
        # Creates ALL combinations, most discarded by sampling
```

**New Targeted Approach:**
```python
def generate_targeted_pipelines(preset: ExperimentPreset) -> list[Pipeline]:
    """Generate only the specific pipeline combinations needed for this experiment."""
    # Create combinations based on variable vs locked slots
    # Only generate what will actually be tested
```

### Example: Frame Removal Focus Study
```yaml
frame_slot: 
  type: variable
  scope: ["*"]  # Test all available frame algorithms
  parameters:
    ratios: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

color_slot:
  type: locked  
  implementation: ffmpeg-color  # Lock to specific color algorithm
  parameters:
    colors: 32

lossy_slot:
  type: locked
  implementation: animately-advanced  # Lock to specific lossy algorithm
  parameters:
    level: 40
```

This generates exactly N frame algorithms × 1 color × 1 lossy = N pipelines instead of hundreds.

**Available Frame Algorithms:** gifski, animately, ffmpeg, imagemagick (scope `["*"]` includes all)

---

## Phase 1: Planning & Requirements Analysis ⏳ PLANNED
**Progress:** 0% Complete  
**Current Focus:** Establishing requirements and technical approach

### Subtask: Problem Analysis ⏳ PLANNED
- [ ] Document current pipeline generation inefficiencies
- [ ] Analyze current generate_all_pipelines() + sampling inefficiencies  
- [ ] Define requirements for targeted pipeline creation
- [ ] Establish success criteria for focused experiments

### Subtask: Technical Requirements ⏳ PLANNED  
- [ ] Map current parameter flow in ExperimentalRunner
- [ ] Identify integration points for targeted generation
- [ ] Document compatibility requirements with existing sampling
- [ ] Define preset configuration format and validation needs

### Subtask: Research Use Cases ⏳ PLANNED
- [ ] Define frame removal study requirements  
- [ ] Specify color optimization study parameters
- [ ] Document lossy quality sweep configurations
- [ ] Identify tool comparison baseline needs
- [ ] Plan for future preset extensibility

---

## Phase 2: Architecture Design ⏳ PLANNED  
**Progress:** 0% Complete
**Current Focus:** System architecture and data structures

### Subtask: Preset Configuration Schema ⏳ PLANNED
- [ ] Design ExperimentPreset dataclass with variable scopes
- [ ] Define SlotConfiguration for frame/color/lossy slots  
- [ ] Create VariableScope and LockedImplementation types
- [ ] Implement preset validation and conflict detection

### Subtask: Pipeline Creation Engine ⏳ PLANNED
- [ ] Design TargetedPipelineGenerator class
- [ ] Plan variable scope expansion algorithms  
- [ ] Define implementation locking mechanisms
- [ ] Create pipeline combination generation logic

### Subtask: Integration Architecture ⏳ PLANNED  
- [ ] Plan ExperimentalRunner integration points
- [ ] Design CLI option processing workflow
- [ ] Define configuration override mechanisms  
- [ ] Plan backward compatibility preservation

---

## Phase 3: Core Implementation ⏳ PLANNED
**Progress:** 0% Complete  
**Current Focus:** Building targeted pipeline creation system

### Subtask: Preset System Foundation ⏳ PLANNED
**File:** `src/giflab/experimental/targeted_presets.py`

```python
@dataclass
class SlotConfiguration:
    """Configuration for a single algorithm slot (frame/color/lossy)."""
    type: Literal["variable", "locked"]
    implementation: Optional[str] = None  # For locked slots
    scope: Optional[List[str]] = None     # For variable slots  
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ExperimentPreset:
    """Targeted experiment configuration."""
    name: str
    description: str
    frame_slot: SlotConfiguration
    color_slot: SlotConfiguration  
    lossy_slot: SlotConfiguration
    custom_sampling: Optional[str] = None
    max_combinations: Optional[int] = None
```

- [ ] Implement SlotConfiguration with validation
- [ ] Create ExperimentPreset with slot management
- [ ] Add preset registry and conflict detection
- [ ] Implement parameter validation logic

### Subtask: Pipeline Generation Logic ⏳ PLANNED
**File:** `src/giflab/experimental/targeted_generator.py`

- [ ] Create `generate_targeted_pipelines(preset)` function
- [ ] Implement variable scope expansion (get all algorithms for variable slots)
- [ ] Add implementation locking mechanisms (use specific algorithms for locked slots)
- [ ] Create targeted combination generation (only create needed combinations)
- [ ] Integrate with existing `tools_for()` function from dynamic_pipeline.py

### Subtask: Built-in Preset Definitions ⏳ PLANNED

```python
TARGETED_PRESETS = {
    'frame-focus': ExperimentPreset(
        name="Frame Removal Focus Study",
        description="Test all frame reduction algorithms with locked color and lossy",
        frame_slot=SlotConfiguration(
            type="variable",
            scope=["*"],  # All available frame algorithms
            parameters={"ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]}
        ),
        color_slot=SlotConfiguration(
            type="locked",
            implementation="ffmpeg-color",
            parameters={"colors": 32}
        ),
        lossy_slot=SlotConfiguration(
            type="locked", 
            implementation="animately-advanced",
            parameters={"level": 40}
        ),
        custom_sampling="quick",
        max_combinations=100
    )
}
```

- [ ] Define frame-focus preset
- [ ] Create color-optimization-study preset
- [ ] Implement lossy-quality-sweep preset  
- [ ] Add tool-comparison-baseline preset
- [ ] Create advanced-research preset for complex studies

---

## Phase 4: Interface Development ⏳ PLANNED
**Progress:** 0% Complete  
**Current Focus:** CLI integration and user experience

### Subtask: CLI Integration ⏳ PLANNED
**File:** `src/giflab/cli/experiment_cmd.py`

- [ ] Add --targeted-preset option with preset choices
- [ ] Implement --variable-slot options for custom scopes  
- [ ] Add --lock-slot options for custom implementations
- [ ] Create --slot-params for parameter specification
- [ ] Integrate with existing sampling and output options

### Subtask: Configuration Processing ⏳ PLANNED  
- [ ] Implement preset option parsing and validation
- [ ] Add custom preset creation from CLI arguments
- [ ] Create slot configuration validation  
- [ ] Implement parameter type conversion and error handling

### Subtask: Pipeline Integration ⏳ PLANNED
- [ ] Integrate `generate_targeted_pipelines()` with ExperimentalRunner
- [ ] Replace `generate_all_pipelines()` + sampling workflow with targeted generation
- [ ] Update experiment_cmd.py to use targeted generation when presets are specified
- [ ] Maintain backward compatibility with existing full generation approach

---

## Phase 5: Testing & Validation ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** Comprehensive testing of targeted pipeline creation

### Subtask: Unit Testing ⏳ PLANNED
**File:** `tests/unit/test_targeted_presets.py`

- [ ] Test SlotConfiguration validation and creation
- [ ] Test ExperimentPreset initialization and validation
- [ ] Test TargetedPipelineGenerator algorithms
- [ ] Test preset registry and conflict detection  
- [ ] Validate error handling and edge cases

### Subtask: Integration Testing ⏳ PLANNED
**File:** `tests/integration/test_targeted_experiments.py`  

- [ ] Test end-to-end preset experiment execution
- [ ] Validate all preset types with real pipeline data
- [ ] Test CLI integration with various configurations
- [ ] Verify targeted generation vs generate_all_pipelines() + sampling performance
- [ ] Test sampling strategy integration

### Subtask: Performance Validation ⏳ PLANNED
**File:** `tests/performance/test_targeted_performance.py`

- [ ] Benchmark targeted generation vs current generate_all_pipelines() + sampling approach  
- [ ] Test memory usage with large combination sets
- [ ] Validate generation performance with complex presets
- [ ] Test concurrent pipeline creation performance
- [ ] Benchmark configuration override performance

---

## Phase 6: Documentation & Deployment ⏳ PLANNED
**Progress:** 0% Complete
**Current Focus:** User documentation and production deployment

### Subtask: User Documentation ⏳ PLANNED
- [ ] Create comprehensive user guide with examples
- [ ] Document all preset types with use cases  
- [ ] Add CLI reference with all options
- [ ] Create troubleshooting guide for common issues
- [ ] Write quick-start guide for new users

### Subtask: Technical Documentation ⏳ PLANNED
- [ ] Add inline code documentation and docstrings  
- [ ] Document preset extension guidelines
- [ ] Create architecture overview documentation
- [ ] Add performance tuning recommendations  
- [ ] Document integration patterns

### Subtask: Production Deployment ⏳ PLANNED
- [ ] Final validation across all preset types
- [ ] Performance benchmarking and acceptance testing  
- [ ] Code review and quality assurance
- [ ] Update project documentation with new capabilities
- [ ] Archive planning documents with final status

---

## Usage Examples

### Command Line Usage

```bash
# Use predefined frame removal study (generates only needed combinations)
poetry run python -m giflab experiment --targeted-preset frame-focus

# Custom experiment: test color algorithms with locked frame and lossy  
poetry run python -m giflab experiment --variable-slot color=all --lock-slot frame=animately-advanced --lock-slot lossy=animately-advanced

# Specify exact algorithms for variable testing (no sampling needed)
poetry run python -m giflab experiment --variable-slot frame=gifski,animately,ffmpeg --lock-slot color=ffmpeg-color:32 --lock-slot lossy=animately-advanced:40

# Targeted study creates exactly what's needed (no max-combinations limit required)
poetry run python -m giflab experiment --targeted-preset color-optimization-study
```

### Programmatic Usage  

```python
from src.giflab.experimental.runner import ExperimentalRunner
from src.giflab.experimental.targeted_presets import TARGETED_PRESETS
from src.giflab.experimental.targeted_generator import generate_targeted_pipelines

# Set up targeted experiment
runner = ExperimentalRunner(output_dir="frame_study_results", use_gpu=True)
preset = TARGETED_PRESETS['frame-focus']

# Generate only the specific pipelines needed (no sampling required)
targeted_pipelines = generate_targeted_pipelines(preset)
print(f"Generated {len(targeted_pipelines)} targeted pipelines (vs {len(generate_all_pipelines())} from generate_all_pipelines)")

# Run experiment with targeted pipelines
result = runner.run_experimental_analysis(test_pipelines=targeted_pipelines)

# Analyze results  
print(f"Tested {result.total_jobs_run} targeted pipeline combinations - all were relevant")
```

## Benefits of Targeted Pipeline Creation

### Efficiency Gains
- **No Wasted Generation**: Create only needed pipeline combinations
- **Faster Experiments**: Eliminate generation overhead and irrelevant pipeline creation
- **Clear Intent**: Experiment configuration directly expresses research goals  
- **Resource Optimization**: Focus computational resources on relevant comparisons

### Research Benefits  
- **Precise Control**: Specify exact algorithms for each slot with clear parameters
- **Flexible Scoping**: Easy to expand or narrow algorithm testing scope  
- **Reproducible Studies**: Preset configurations ensure consistent methodology
- **Future Extensibility**: Simple to add new algorithms to variable scopes

### Implementation Benefits
- **Intuitive Configuration**: Slot-based thinking matches research mental models
- **Non-invasive Integration**: Extends ExperimentalRunner without breaking changes  
- **Maintainable Code**: Clear separation between pipeline creation and execution
- **Comprehensive Testing**: Focused test coverage for targeted generation logic

## Success Metrics

1. **Generation Efficiency**: 80%+ reduction in pipeline generation (create only needed combinations vs all combinations)
2. **Memory Usage**: 70%+ reduction in memory footprint during pipeline generation phase
3. **Setup Speed**: 50%+ faster experiment startup (no sampling phase required)
4. **User Adoption**: Researchers regularly use targeted presets instead of generate_all_pipelines() + sampling
5. **Configuration Clarity**: New users can set up focused experiments in <10 minutes without understanding sampling strategies

