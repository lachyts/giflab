# GifLab Pipeline Consolidation Refactor

**Status:** 🚧 In Progress  
**Started:** 2025-08-22  
**Priority:** High  
**Impact:** Major refactor - breaking changes  

## Executive Summary

This refactor consolidates GifLab's dual pipeline architecture by promoting the experimental pipeline to become the main pipeline, eliminating redundant code and providing users with the full suite of advanced features that were previously experimental.

## Problem Statement

GifLab currently maintains two separate pipelines:

1. **Main Pipeline** (`CompressionPipeline`) - Simple, production-focused
2. **Experimental Pipeline** (`ExperimentalRunner`) - Feature-rich with advanced capabilities

This creates:
- Code duplication and maintenance overhead
- User confusion about which pipeline to use
- Advanced features trapped behind "experimental" terminology
- Inconsistent CSV output formats between pipelines

## Solution Overview

**Promote the experimental pipeline to become the main pipeline** while preserving valuable metadata fields from the current main pipeline.

## Current Architecture Analysis

### Main Pipeline (`src/giflab/pipeline.py`)
- **Class:** `CompressionPipeline`
- **CLI Command:** `giflab run`
- **CSV Fields (22 total):**
  ```
  gif_sha, orig_filename, engine, engine_version, lossy, frame_keep_ratio, 
  color_keep_count, kilobytes, ssim, render_ms, orig_kilobytes, orig_width, 
  orig_height, orig_frames, orig_fps, orig_n_colors, entropy, source_platform, 
  source_metadata, timestamp, giflab_version, code_commit, dataset_version
  ```
- **Capabilities:** File discovery, job generation, multiprocessing, basic metrics

### Experimental Pipeline (`src/giflab/experimental/runner.py`)
- **Class:** `ExperimentalRunner`  
- **CLI Command:** `giflab experiment`
- **CSV Fields (42 total):** Comprehensive quality metrics, error handling, pipeline analysis
- **Advanced Capabilities:**
  - GPU acceleration for quality metrics
  - Comprehensive quality analysis (12+ metrics vs just SSIM)
  - Targeted presets and intelligent sampling
  - Pipeline validation and error handling
  - Result caching system
  - Progress tracking and resume functionality
  - Pareto frontier analysis
  - ML-ready data structure
  - Streaming CSV output for large datasets
  - Advanced signal handling

## Missing Fields Analysis

**Fields in Main Pipeline but NOT in Experimental:**
- `gif_sha` - SHA hash for deduplication (CRITICAL)
- `orig_filename` - Original filename preservation (IMPORTANT)
- `engine_version` - Tool version tracking (IMPORTANT)
- `orig_fps` - Original FPS metadata (USEFUL)
- `orig_n_colors` - Original color count (USEFUL)
- `entropy` - Image entropy measure (RESEARCH VALUE)
- `source_platform` - Platform detection (Tenor, Animately, etc.) (IMPORTANT)
- `source_metadata` - Extracted platform metadata (IMPORTANT)
- `timestamp` - Processing timestamp (IMPORTANT)
- `giflab_version` - Version tracking (CRITICAL)
- `code_commit` - Git commit hash (CRITICAL)
- `dataset_version` - Dataset version tracking (IMPORTANT)

## Refactor Implementation Plan

### Phase 1: CSV Field Enhancement ✅
**Goal:** Add missing main pipeline fields to experimental pipeline

**Changes Required:**
1. **Update `csv_fieldnames` in `ExperimentalRunner._run_comprehensive_testing()`**
   - Add the 12 missing fields from main pipeline
   - Ensure proper ordering for readability
   - Update CSV writer initialization

2. **Modify result collection logic**
   - Add field population in `_execute_pipeline_with_metrics()`
   - Implement SHA hash calculation for `gif_sha`
   - Add filename extraction for `orig_filename`
   - Integrate engine version detection for `engine_version`
   - Add FPS and color count extraction
   - Implement entropy calculation
   - Integrate source platform detection
   - Add timestamp, version, and commit tracking

### Phase 2: Terminology Cleanup 📋
**Goal:** Remove "experimental" terminology from user-facing components

**Changes Required:**
1. **Directory Structure:**
   - `src/giflab/experimental/` → `src/giflab/core/`
   - Update all import statements across codebase

2. **Class Renaming:**
   - `ExperimentalRunner` → `GifLabRunner`
   - Update all references and imports

3. **Method and Variable Renaming:**
   - Remove "experimental" from method names, docstrings, log messages
   - Update CLI help text and command descriptions

4. **Documentation Updates:**
   - Update all README references
   - Remove experimental warnings from documentation

### Phase 3: CLI Restructuring 🔧
**Goal:** Make promoted pipeline the default while maintaining backward compatibility

**Changes Required:**
1. **Update `giflab run` command (`src/giflab/cli/run_cmd.py`)**
   - Replace `CompressionPipeline` usage with `GifLabRunner`
   - Maintain CLI argument compatibility where possible
   - Add migration warnings for deprecated arguments

2. **Deprecate `giflab experiment` command**
   - Keep as alias pointing to `giflab run` 
   - Add deprecation warnings
   - Update help text to redirect users

3. **CLI Interface Harmonization**
   - Ensure consistent argument naming
   - Merge useful options from both commands
   - Add migration guide in CLI help

### Phase 4: Legacy Code Removal 🗑️
**Goal:** Remove outdated main pipeline code

**Changes Required:**
1. **Archive `src/giflab/pipeline.py`**
   - Move to `src/giflab/legacy/pipeline.py` (temporary)
   - Add deprecation warnings
   - Eventually remove after validation period

2. **Update Import Dependencies**
   - Find all imports of `CompressionPipeline`
   - Replace with `GifLabRunner` usage
   - Update test files

3. **Configuration Updates**
   - Update any config files referencing old pipeline
   - Update pyproject.toml if needed

### Phase 5: Testing and Validation ✅
**Goal:** Ensure refactor maintains functionality

**Tasks:**
1. **Unit Test Updates**
   - Update test imports and class names
   - Add tests for new CSV fields
   - Validate field compatibility

2. **Integration Testing**
   - Run full pipeline with existing datasets
   - Compare CSV outputs before/after refactor
   - Validate performance characteristics

3. **Regression Testing**
   - Test CLI command compatibility
   - Validate resume functionality works
   - Test caching system integration

## Implementation Progress Tracking

### Completed ✅
- [ ] Architecture analysis and planning
- [ ] Documentation creation

### In Progress 🚧
- [ ] CSV field enhancement implementation

### Pending 📋
- [ ] Terminology cleanup
- [ ] CLI restructuring  
- [ ] Legacy code removal
- [ ] Testing and validation

## Risk Assessment and Mitigation

### High Risk Items
1. **Breaking Changes for Existing Users**
   - **Mitigation:** Maintain backward compatibility aliases, provide migration guide
   
2. **CSV Output Changes**
   - **Mitigation:** Add new fields without removing existing ones, provide field mapping documentation
   
3. **Performance Regression**
   - **Mitigation:** Benchmark before/after, optimize field extraction code

### Medium Risk Items
1. **Import Path Changes**
   - **Mitigation:** Gradual migration with deprecation warnings
   
2. **Configuration Compatibility**
   - **Mitigation:** Test with existing config files, provide migration scripts

## Success Criteria

### Functional Requirements ✅
- [ ] Single unified pipeline with all advanced features
- [ ] All CSV fields from both pipelines available
- [ ] CLI commands work without breaking existing scripts
- [ ] Performance matches or exceeds current experimental pipeline

### Quality Requirements ✅
- [ ] No "experimental" terminology in user-facing components
- [ ] Comprehensive test coverage (>90%)
- [ ] Documentation completely updated
- [ ] Migration guide available for users

### Technical Requirements ✅
- [ ] Clean code structure without duplication
- [ ] Proper deprecation warnings for legacy features
- [ ] Backward compatibility for essential workflows
- [ ] All existing functionality preserved

## Timeline Estimate

- **Phase 1:** 2-3 days (CSV field enhancement)
- **Phase 2:** 1-2 days (terminology cleanup)  
- **Phase 3:** 2-3 days (CLI restructuring)
- **Phase 4:** 1 day (legacy removal)
- **Phase 5:** 2-3 days (testing/validation)

**Total Estimated Duration:** 8-12 days

## Post-Refactor Maintenance

1. **Monitor for Issues**
   - Track user feedback on CLI changes
   - Monitor performance metrics
   - Watch for regression reports

2. **Documentation Maintenance**
   - Keep migration guide updated
   - Update examples and tutorials
   - Maintain changelog

3. **Future Considerations**
   - Complete removal of legacy code after 1-2 release cycles
   - Consider additional pipeline optimizations
   - Plan for next major feature additions

---

## Notes and Decisions

### Decision Log
- **2025-08-22:** Decided to preserve all CSV fields from both pipelines rather than removing any
- **2025-08-22:** Chose `GifLabRunner` over `CompressionRunner` for clarity
- **2025-08-22:** Decided to keep `giflab experiment` as deprecated alias rather than immediate removal

### Technical Considerations
- New CSV fields will be populated on best-effort basis to maintain compatibility
- Resume functionality must work across the refactor
- Caching system should be preserved and enhanced where possible

### Dependencies
- No external dependencies expected to change
- Poetry configuration should not require updates
- Existing test infrastructure should largely remain compatible