# Architecture Overview: Targeted Experiment Presets

This document provides a comprehensive technical overview of the targeted experiment presets system architecture, design patterns, and integration points within the GifLab framework.

## System Overview

The targeted experiment presets system replaces the inefficient `generate_all_pipelines()` + sampling approach with direct targeted pipeline generation, achieving 80-99% efficiency improvements for focused research studies.

### Core Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Preset System    │    │  Pipeline Generator  │    │ Experiment Runner   │
│                     │    │                      │    │                     │
│ ┌─────────────────┐ │    │ ┌──────────────────┐ │    │ ┌─────────────────┐ │
│ │ SlotConfiguration│ │    │ │TargetedPipeline  │ │    │ │ GifLabRunner│ │
│ │                 │ │    │ │    Generator     │ │    │ │                 │ │
│ │ • Variable Slot │ │───▶│ │                  │ │───▶│ │ • Preset Integration│
│ │ • Locked Slot   │ │    │ │ • Scope Expansion│ │    │ │ • Pipeline Execution│
│ │ • Parameters    │ │    │ │ • Tool Resolution│ │    │ │ • Result Analysis │ │
│ └─────────────────┘ │    │ │ • Combination    │ │    │ └─────────────────┘ │
│                     │    │ │   Generation     │ │    │                     │
│ ┌─────────────────┐ │    │ └──────────────────┘ │    └─────────────────────┘
│ │ExperimentPreset │ │    └──────────────────────┘              │
│ │                 │ │                                           │
│ │ • 3 Slot Config │ │    ┌──────────────────────┐              │
│ │ • Metadata      │ │    │   CLI Integration    │              │
│ │ • Validation    │ │    │                      │              │
│ └─────────────────┘ │    │ ┌──────────────────┐ │              │
│                     │    │ │ run_cmd.py│ │              │
│ ┌─────────────────┐ │    │ │                  │ │              │
│ │ PresetRegistry  │ │    │ │ • Preset Options │ │              │
│ │                 │ │    │ │ • Custom Slots   │ │──────────────┘
│ │ • Global Access │ │    │ │ • Parameter      │ │
│ │ • Conflict      │ │    │ │   Processing     │ │
│ │   Detection     │ │    │ └──────────────────┘ │
│ │ • Validation    │ │    └──────────────────────┘
│ └─────────────────┘ │
└─────────────────────┘
```

## Core Components

### 1. Slot Configuration System

**Purpose**: Define algorithm behavior for each compression stage
**Location**: `src/giflab/core/targeted_presets.py`

```python
@dataclass
class SlotConfiguration:
    """Configuration for frame/color/lossy algorithm slots."""
    type: Literal["variable", "locked"]           # Slot behavior
    implementation: Optional[str] = None          # For locked slots
    scope: Optional[List[str]] = None            # For variable slots  
    parameters: Dict[str, Any] = field(default_factory=dict)
```

**Design Patterns**:
- **Variable Slot**: Test multiple algorithms (`scope=["*"]` or specific tools)
- **Locked Slot**: Use single algorithm (`implementation="tool-name"`)
- **Parameter Binding**: Algorithm-specific parameters per slot

**Validation Logic**:
```python
def __post_init__(self):
    """Validate slot configuration consistency."""
    if self.type == "locked":
        if not self.implementation:
            raise ValueError("Locked slots must specify implementation")
        if self.scope:
            raise ValueError("Locked slots cannot have scope")
    elif self.type == "variable":
        if not self.scope:
            raise ValueError("Variable slots must specify scope")
        if self.implementation:
            raise ValueError("Variable slots cannot have implementation")
```

### 2. Experiment Preset System

**Purpose**: Combine slot configurations into complete experiment definitions
**Location**: `src/giflab/core/targeted_presets.py`

```python
@dataclass
class ExperimentPreset:
    """Complete experiment configuration with metadata."""
    name: str
    description: str
    
    # Required: Three algorithm slots
    frame_slot: SlotConfiguration     # Frame reduction algorithms
    color_slot: SlotConfiguration     # Color quantization algorithms  
    lossy_slot: SlotConfiguration     # Lossy compression algorithms
    
    # Optional: Enhancement parameters
    custom_sampling: Optional[str] = None      # Additional sampling strategy
    max_combinations: Optional[int] = None     # Hard limit on pipelines
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    version: str = "1.0"
```

**Key Methods**:
- `get_variable_slots()` - Identify which dimensions vary
- `get_locked_slots()` - Get fixed algorithm configurations
- `estimate_pipeline_count()` - Calculate expected pipeline combinations

### 3. Pipeline Generation Engine

**Purpose**: Convert presets into specific pipeline combinations
**Location**: `src/giflab/core/targeted_generator.py`

```python
class TargetedPipelineGenerator:
    """Core engine for targeted pipeline generation."""
    
    def generate_targeted_pipelines(self, preset: ExperimentPreset) -> List[Pipeline]:
        """Generate only needed pipeline combinations."""
        # 1. Resolve variable scopes to actual tools
        frame_tools = self._resolve_slot_tools("frame_reduction", preset.frame_slot)
        color_tools = self._resolve_slot_tools("color_reduction", preset.color_slot)
        lossy_tools = self._resolve_slot_tools("lossy_compression", preset.lossy_slot)
        
        # 2. Generate targeted combinations
        pipelines = []
        combinations = itertools.product(frame_tools, color_tools, lossy_tools)
        
        for frame_tool, color_tool, lossy_tool in combinations:
            pipeline = Pipeline([
                PipelineStep("frame_reduction", frame_tool),
                PipelineStep("color_reduction", color_tool),
                PipelineStep("lossy_compression", lossy_tool)
            ])
            pipelines.append(pipeline)
            
        return pipelines
```

**Algorithm Flow**:
1. **Slot Resolution**: Convert preset slots to tool lists
2. **Scope Expansion**: Expand wildcards (`*`) to available tools
3. **Tool Validation**: Verify all tools exist in capability registry
4. **Combination Generation**: Create cartesian product of resolved tools
5. **Pipeline Construction**: Build Pipeline objects with PipelineStep chains
6. **Limit Application**: Apply max_combinations if specified

### 4. Preset Registry System

**Purpose**: Global management of available presets
**Location**: `src/giflab/core/targeted_presets.py`

```python
class PresetRegistry:
    """Global registry for experiment presets."""
    
    def __init__(self):
        self.presets: Dict[str, ExperimentPreset] = {}
    
    def register(self, preset_id: str, preset: ExperimentPreset):
        """Register preset with validation and conflict detection."""
        self._validate_preset_configuration(preset)
        if preset_id in self.presets:
            logger.warning(f"Overwriting existing preset: {preset_id}")
        self.presets[preset_id] = preset
    
    def find_similar_presets(self, preset: ExperimentPreset) -> List[str]:
        """Detect similar configurations to avoid duplication."""
        similar = []
        for existing_id, existing_preset in self.presets.items():
            similarity = self._calculate_preset_similarity(preset, existing_preset)
            if similarity > 0.7:  # 70% similarity threshold
                similar.append(existing_id)
        return similar
```

**Registry Features**:
- **Global Access**: Single point of access for all presets
- **Conflict Detection**: Identify similar/duplicate presets
- **Validation**: Ensure all registered presets are valid
- **Introspection**: List, describe, and analyze presets

## Integration Architecture

### 1. GifLabRunner Integration

**Purpose**: Integrate targeted generation with existing experiment infrastructure
**Location**: `src/giflab/core/runner.py`

```python
class GifLabRunner:
    """Enhanced with targeted preset support."""
    
    def generate_targeted_pipelines(self, preset_id: str) -> List[Pipeline]:
        """Generate pipelines for specific preset."""
        preset = PRESET_REGISTRY.get(preset_id)
        generator = TargetedPipelineGenerator(self.logger)
        return generator.generate_targeted_pipelines(preset)
    
    def run_targeted_experiment(self, preset_id: str, **kwargs) -> ExperimentResult:
        """Complete targeted experiment workflow."""
        test_pipelines = self.generate_targeted_pipelines(preset_id)
        return self.run_analysis(test_pipelines=test_pipelines, **kwargs)
    
    def list_available_presets(self) -> Dict[str, str]:
        """List all available presets with descriptions."""
        return PRESET_REGISTRY.list_presets()
```

**Integration Points**:
- **Pipeline Generation**: `generate_targeted_pipelines()` replaces `generate_all_pipelines()` + sampling
- **Experiment Execution**: `run_targeted_experiment()` provides complete workflow
- **Preset Discovery**: `list_available_presets()` enables preset exploration

### 2. CLI Integration Architecture

**Purpose**: Expose targeted presets via command-line interface
**Location**: `src/giflab/cli/run_cmd.py`

```python
@click.command()
@click.option("--preset", "-p", type=str, help="Use targeted experiment preset")
@click.option("--list-presets", is_flag=True, help="List available presets")
@click.option("--variable-slot", multiple=True, help="Define variable slot")
@click.option("--lock-slot", multiple=True, help="Lock slot to specific implementation")
@click.option("--slot-params", multiple=True, help="Specify slot parameters")
def experiment(**kwargs):
    """Enhanced experiment command with preset support."""
    
    # Preset mode vs custom slot mode
    if kwargs['preset']:
        test_pipelines = runner.generate_targeted_pipelines(kwargs['preset'])
        approach = "targeted_preset"
    elif kwargs['variable_slot'] or kwargs['lock_slot']:
        custom_preset = create_custom_preset_from_cli(...)
        test_pipelines = generator.generate_targeted_pipelines(custom_preset)
        approach = "custom_slots"
    else:
        # Traditional fallback
        all_pipelines = generate_all_pipelines()
        test_pipelines = runner.select_pipelines_intelligently(...)
        approach = "traditional_sampling"
```

**CLI Architecture Features**:
- **Preset Mode**: `--preset frame-focus` uses built-in presets
- **Custom Mode**: `--variable-slot` and `--lock-slot` create dynamic presets
- **Parameter Override**: `--slot-params` specifies algorithm parameters
- **Discovery Mode**: `--list-presets` shows available options
- **Backward Compatibility**: Traditional workflows continue working

### 3. Built-in Presets Architecture

**Purpose**: Provide research-validated preset configurations
**Location**: `src/giflab/core/builtin_presets.py`

```python
# Auto-registration pattern
TARGETED_PRESETS = {
    'frame-focus': ExperimentPreset(...),          # Research presets
    'color-optimization': ExperimentPreset(...),   # Research presets
    'tool-comparison-baseline': ExperimentPreset(...), # Baseline presets
    'dithering-focus': ExperimentPreset(...),      # Specialized presets
    'quick-test': ExperimentPreset(...)            # Development presets
}

def register_builtin_presets():
    """Register all built-in presets with global registry."""
    for preset_id, preset in TARGETED_PRESETS.items():
        PRESET_REGISTRY.register(preset_id, preset)

# Auto-registration on module import
register_builtin_presets()
```

**Preset Categories**:
- **Research Presets**: Single-dimension algorithm comparisons (~5-17 pipelines)
- **Baseline Presets**: Multi-dimension comprehensive comparisons (~64 pipelines)
- **Specialized Presets**: Focused optimization scenarios (~4-6 pipelines)
- **Development Presets**: Minimal testing configurations (~2 pipelines)

## Data Flow Architecture

### 1. Preset-Based Workflow

```
User Input (--preset frame-focus)
         │
         ▼
    Preset Lookup (PRESET_REGISTRY.get())
         │
         ▼
    Preset Validation (validate_preset_feasibility())
         │
         ▼
    Tool Resolution (tools_for() + scope expansion)
         │
         ▼
    Pipeline Generation (itertools.product())
         │
         ▼
    Experiment Execution (run_analysis())
         │
         ▼
    Results Analysis (standard experiment pipeline)
```

### 2. Custom Slot Workflow

```
CLI Options (--variable-slot, --lock-slot, --slot-params)
         │
         ▼
    Parameter Parsing (create_custom_preset_from_cli())
         │
         ▼
    Dynamic Preset Creation (ExperimentPreset())
         │
         ▼
    Preset Validation (validate_preset_feasibility())
         │
         ▼
    [Same as preset-based workflow from Tool Resolution]
```

### 3. Traditional Fallback Workflow

```
No Preset Options
         │
         ▼
    Generate All Pipelines (generate_all_pipelines())
         │
         ▼
    Apply Sampling Strategy (select_pipelines_intelligently())
         │
         ▼
    Experiment Execution (run_analysis())
```

## Performance Architecture

### 1. Efficiency Optimizations

**Pipeline Count Reduction**:
```
Traditional:    generate_all_pipelines() → 935 pipelines → sampling → 46-233 used
Targeted:       generate_targeted_pipelines() → 5-64 pipelines → all used
Efficiency:     93-99% reduction in generated pipelines
```

**Memory Optimization**:
- **Tool Caching**: `_tool_cache` prevents repeated capability registry lookups
- **Lazy Evaluation**: Tools resolved only when needed
- **Scope Limiting**: Specific tool lists avoid wildcard expansion when possible

**Generation Speed**:
- **Direct Creation**: No intermediate pipeline generation and filtering
- **Validation Caching**: Preset validation cached during registration
- **Parameter Pre-binding**: Slot parameters resolved once during generation

### 2. Scalability Architecture

**Preset Scaling**:
- Registry supports unlimited preset definitions
- Memory usage scales with active presets, not total presets
- Registration validation prevents invalid configurations

**Pipeline Scaling**:
- `max_combinations` parameter prevents pipeline explosion
- Custom sampling integration provides additional limiting
- Resource estimation helps users choose appropriate presets

**Tool Scaling**:
- Dynamic tool discovery supports new algorithm additions
- Capability registry integration provides automatic tool availability
- Wildcard expansion scales with available tools

## Error Handling Architecture

### 1. Validation Layers

**Level 1: Slot Validation**
```python
# In SlotConfiguration.__post_init__()
- Type consistency (variable/locked requirements)
- Parameter format validation  
- Implementation existence checking
```

**Level 2: Preset Validation**
```python  
# In ExperimentPreset.__post_init__()
- At least one variable slot requirement
- Sampling strategy validation
- Max combinations bounds checking
```

**Level 3: Registry Validation**
```python
# In PresetRegistry.register()
- Preset configuration validation
- Conflict detection and warnings
- Registration constraint checking
```

**Level 4: Generation Validation**
```python
# In TargetedPipelineGenerator.generate_targeted_pipelines()
- Tool availability checking
- Scope resolution validation
- Pipeline construction verification
```

### 2. Error Recovery Strategies

**Graceful Degradation**:
- Invalid presets fall back to traditional generation
- Missing tools skip gracefully with warnings
- Parameter errors use defaults when possible

**User Guidance**:
- Specific error messages with suggested solutions
- Alternative preset recommendations for failed configurations
- Tool availability reporting for troubleshooting

## Testing Architecture

### 1. Test Structure

```
tests/
├── unit/
│   └── test_targeted_presets.py          # Component unit tests
└── integration/
    └── test_targeted_experiments.py      # End-to-end integration tests
```

**Unit Test Coverage**:
- SlotConfiguration validation and behavior
- ExperimentPreset creation and introspection
- TargetedPipelineGenerator algorithms
- PresetRegistry management and conflict detection
- Built-in preset validation and generation

**Integration Test Coverage**:
- Complete preset experiment execution
- CLI integration and parameter processing
- GifLabRunner integration
- Performance comparison validation
- Backward compatibility verification

### 2. Test Architecture Patterns

**Validation Testing**:
```python
def test_slot_validation():
    """Test slot configuration validation."""
    with pytest.raises(ValueError, match="Variable slots must specify scope"):
        SlotConfiguration(type="variable", implementation="should-not-have-this")
```

**Generation Testing**:
```python
def test_pipeline_generation():
    """Test targeted pipeline generation."""
    generator = TargetedPipelineGenerator()
    preset = PRESET_REGISTRY.get("frame-focus")
    pipelines = generator.generate_targeted_pipelines(preset)
    assert len(pipelines) > 0
    assert all(hasattr(p, 'steps') for p in pipelines)
```

**Integration Testing**:
```python
def test_end_to_end_execution():
    """Test complete experiment workflow."""
    runner = GifLabRunner(output_dir=Path(temp_dir), use_cache=False)
    result = runner.run_targeted_experiment("frame-focus", quality_threshold=0.1)
    assert result.total_jobs_run > 0
```

## Extension Points

### 1. Custom Preset Creation

**Programmatic API**:
```python
custom_preset = ExperimentPreset(...)
PRESET_REGISTRY.register("custom-id", custom_preset)
```

**CLI Dynamic Creation**:
```bash
--variable-slot frame=* --lock-slot color=ffmpeg-color --slot-params color=colors:32
```

### 2. Algorithm Integration

**Tool Registration**:
- New tools automatically available via capability registry
- Wildcard scopes (`*`) include new tools automatically
- Specific tool lists require manual updates

**Parameter Extension**:
- Slot parameters support arbitrary key-value pairs
- Parameter validation enforced by tool implementations
- Custom parameter ranges supported per preset

### 3. Sampling Integration

**Custom Sampling Strategies**:
- Presets can specify additional sampling via `custom_sampling`
- Sampling applied after targeted generation if specified
- Hybrid approach: targeted generation + intelligent sampling

This architecture provides a robust, extensible foundation for efficient experiment design while maintaining full compatibility with existing GifLab infrastructure.