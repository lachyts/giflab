# 🎯 Next Tools Implementation Plan

*Step-by-step guide for integrating additional compression tools into the **experimental framework** for **matrix-based testing** across compression variables.*  
Current focus: **ImageMagick**, **FFmpeg**, **gifski** + existing **Animately** and **Gifsicle**.

---

## 🚀 Implementation Goal

Enable **matrix-based experiments** testing different combinations of:
- **Color reduction** (palette optimization)
- **Frame reduction** (temporal sampling)
- **Lossy compression** (quality degradation)
- **Tool combinations** (multi-step pipelines)

**Target Strategy Model**: A central capability registry lists every tool that can perform each compression variable. The experiment runner builds all combinations dynamically; no static list is maintained in the documentation.

**Success Criteria**: A user can run matrix experiments testing how different tools perform on specific compression aspects, allowing us to identify which tools excel at which tasks and create optimal hybrid pipelines.

*(Non-GIF output formats such as WebP/AVIF remain out-of-scope for this phase and are listed under **Future Work**.)*

---

## 📊 Matrix Experimental Approach

### Core Compression Variables

1. **Color Reduction**
   - Palette size optimization (256 → 128 → 64 → 32 colors)
   - Dithering strategies (Floyd-Steinberg, ordered, none)
   - Color quantization algorithms

2. **Frame Reduction**
   - Temporal sampling (every 2nd, 3rd, 4th frame)
   - Intelligent frame selection
   - Motion-based frame dropping

3. **Lossy Compression**
   - Quality levels (100% → 90% → 80% → 70%)
   - Compression artifacts tolerance
   - Perceptual quality trade-offs

4. **Tool Combinations**
   - Single-tool approaches
   - Multi-step pipelines (Tool A → Tool B → Tool C)
   - Specialized task allocation

### Experimental Matrix Structure

The experimental framework operates on a *dynamic* matrix, where each axis represents one of the core compression variables (Frame, Color, Lossy). Instead of being statically defined, this matrix is **dynamically generated** by the experiment runner.

The runner populates the matrix by creating a pipeline for the Cartesian product of all registered tools for each slot. This ensures that every valid combination of tools and variables is tested systematically. For example, a simplified matrix might look like this conceptually:

- **Frame Reduction Slot**: [Animately, FFmpeg, Gifsicle, ImageMagick, None]
- **Color Reduction Slot**: [Animately, FFmpeg, Gifsicle, ImageMagick, None]
- **Lossy Compression Slot**: [Animately, Gifsicle, ImageMagick, FFmpeg, gifski, None]

The full set of experimental pipelines is the product of these lists, creating a comprehensive test suite automatically.

---

## 📋 Implementation Steps

### Step 1: Foundation Setup

**Goal**: Install tools and create the base infrastructure for matrix experiments.

**Tasks**:
1. Install system tools (ImageMagick, FFmpeg, gifski)
2. Create modular tool interfaces for each compression variable:
   - `ColorReductionTool` (palette optimization)
   - `FrameReductionTool` (temporal sampling)
   - `LossyCompressionTool` (quality degradation)
3. Update experiment configuration to support variable-based strategy definitions

**Completion Criteria**:
- [ ] Tools installed and responding to version checks
- [ ] Variable-specific tool interfaces created
- [ ] Experiment framework supports matrix experiment modes

---

### Step 2: Tool Capability Mapping

**Goal**: Implement specialized wrappers that expose each tool's specific compression capabilities.

**Tasks**:
1. **Animately Capabilities**:
   - Color Reduction: Advanced palette optimization
   - Frame Reduction: Intelligent frame selection
   - Lossy Compression: Quality-based optimization
   - Integration: Existing wrapper enhancement

2. **Gifsicle Capabilities**:
   - Color Reduction: Advanced palette optimization
   - Lossy Compression: Quality-based optimization
   - Frame Reduction: Layer optimization
   - Integration: Existing wrapper enhancement

3. **ImageMagick Capabilities**:
   - Color Reduction: Multiple quantization algorithms
   - Lossy Compression: Quality-based compression with artifact control
   - Frame Reduction: Layer optimization and frame manipulation

4. **FFmpeg Capabilities**:
   - Color Reduction: Sophisticated palette generation with customizable dithering
   - Frame Reduction: Intelligent frame selection and temporal filtering
   - Lossy Compression: Quality-based encoding with perceptual optimization

5. **gifski Capabilities**:
   - Input Requirement: PNG frames (requires frame extraction preprocessing)
   - Lossy Compression (High-Quality Encoding): Excellent for final compression step in pipelines
   - Integration: Existing wrapper enhancement

**Completion Criteria**:
- [ ] All 5 tools have variable-specific capability wrappers
- [ ] Each tool exposes its specialized compression functions
- [ ] Consistent input/output formats between tools

---

### Step 3: Single-Variable Strategy Implementation

**Goal**: Verify that each tool’s individual capabilities work when run in isolation.

**Tasks**:
1. For each compression variable (Frame, Color, Lossy) generate a pipeline containing **only** that slot and populate it with every tool that advertises the corresponding capability in the registry.
2. Run and validate the output to ensure correctness and consistent I/O.

**Completion Criteria**:
- [ ] 10+ single-variable strategies implemented
- [ ] Each strategy tests one compression aspect in isolation
- [ ] All strategies integrate with experiment framework

---

### Step 4: Multi-Variable Pipeline Strategies

**Goal**: Generate multi-step pipelines *dynamically* via a slot-based capability registry.

**Concept**:
Pipelines have three ordered slots (default execution order):
1. **Frame Reduction** – remove/sample frames first so later stages analyse the exact set of frames that will remain.
2. **Color Reduction** – build / optimise the palette from the already-reduced frame set for better color accuracy.
3. **Lossy Compression** – final size/quality trade-off once spatial & temporal content are fixed.

Each tool advertises which slot(s) it can fill in a central capability registry (YAML/JSON/Python dict).
The experiment runner builds pipelines by taking the Cartesian product of the selected slot options, with optional include/exclude filters.  Names are generated automatically (e.g. `ffmpegColor__noneFrame__gifsicleLossy`) or via a hashed ID.

**Optimization**: If consecutive slots are fulfilled by the same tool, the pipeline generator will check for a `combines: true` flag in that tool's capability entry. If present, the generator merges the flags for those steps into a single command, avoiding unnecessary intermediate file I/O and preserving efficiency. This logic also applies when intermediate slots are skipped; for example, if only Frame Reduction and Lossy Compression are selected for the same tool, they will be combined into a single step.

**Tasks**:
1. Design and populate the capability registry, including a `combines: true` flag for tools that can merge multiple operations into a single command.
2. Implement a pipeline generator that:
   - Enumerates valid slot combinations.
   - Groups consecutive, combinable operations from the same tool into a single execution step.
   - Applies user filters.
   - Emits an executable `Pipeline` object (Frame → Color → Lossy).
3. Add an automatic naming/ID scheme for each generated pipeline.
4. Integrate the generator into the experiment runner and retire the hard-coded lists.

**Completion Criteria**:
- [ ] Capability registry committed and loaded at runtime
- [ ] Generator produces valid pipelines across all slots and correctly groups combinable steps.
- [ ] Filtering and naming conventions documented and functional
- [ ] Example experiment run completes using generated pipelines

---

### Step 5: Testing & Validation

**Goal**: Ensure all strategies work correctly and provide reliable results.

**Tasks**:
1. **Variable Isolation Tests**:
   - Test each tool's specialized functions work in isolation
   - Verify consistent input/output formats

2. **Matrix Integration Tests**:
   - Test all variable combinations execute correctly
   - Verify pipeline strategies work end-to-end

3. **Performance Matrix Analysis**:
   - Run experiments across different content types
   - Measure performance of each variable combination
   - Identify optimal tool-to-task mappings

4. **Deprecate Legacy Strategy Aliases**: Once the dynamic generator is fully integrated and validated, remove the old hard-coded strategy names (`pure_gifsicle`, `animately_then_gifsicle`, etc.) from the codebase to simplify maintenance.

**Completion Criteria**:
- [ ] Variable isolation tests added and passing
- [ ] Matrix integration tests added and passing
- [ ] Pipeline validation tests added and passing
- [ ] Performance analysis tools working

---

### Step 6: Documentation & Analysis Tools

**Goal**: Create tools and documentation to analyze matrix experiment results.

**Tasks**:
1. **Matrix Experiment Documentation**:
   - Document which tools excel at which compression variables
   - Create performance matrices showing tool effectiveness per task

2. **Pipeline Visualization**:
   - Tools to visualize multi-step compression pipelines
   - Performance analysis across different variable combinations

3. **Optimization Recommendations**:
   - Based on matrix results, provide guidance on optimal tool selection
   - Content-aware recommendations for different GIF types

**Completion Criteria**:
- [ ] Matrix analysis tools created
- [ ] Pipeline visualization tools implemented
- [ ] Documentation updated with variable-focused approach
- [ ] Optimization recommendation system working

---

## 🎯 Current Step Focus

**When directing Cursor, reference specific steps:**
- "Work on Step 1" = Foundation Setup
- "Work on Step 2" = Tool Capability Mapping  
- "Work on Step 3" = Single-Variable Strategies
- "Work on Step 4" = Multi-Variable Pipelines
- "Work on Step 5" = Testing & Validation
- "Work on Step 6" = Documentation & Analysis

Each step has clear completion criteria that must be met before moving to the next step.

---

## 📈 Success Metrics

| Metric                          | Target                                   |
|---------------------------------|------------------------------------------|
| Compression variables tested    | 3 (color, frame, lossy)                 |
| Tools integrated               | 5 (Animately, Gifsicle, ImageMagick, FFmpeg, gifski) |
| Tool-variable combinations      | 15+ (5 tools × 3 variables minimum)     |
| Pipeline generation             | Dynamic slot-based generator active     |
| Matrix experiment success       | All variable combinations execute       |
| Performance insights            | Clear tool-to-task effectiveness mapping |

---

## 🔮 Future Work (Out-Of-Scope for This Phase)

| Candidate            | Notes                                   |
|----------------------|-----------------------------------------|
| **ML-driven selection** | Automatic tool selection based on content analysis |
| **WebP pipelines**   | GIF → WebP conversion with quality optimization |
| **AVIF pipelines**   | GIF → AVIF conversion for next-gen formats |
| **Neural compression** | AI-based compression techniques         |
| **Real-time optimization** | Dynamic pipeline adjustment based on results |

---

*This matrix-based approach will provide comprehensive insights into which tools excel at specific compression tasks, enabling the creation of optimal hybrid pipelines that leverage each tool's strengths.* 