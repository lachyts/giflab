î# GifLab – Refactor TODO Roadmap

> Last updated: August 2025  
> Author: development team (updated via code analysis)

This document captures the medium-sized refactor items that surfaced during the
code-review on `main` (post-commit).  None of the tasks change current
behavior, but they will greatly improve maintainability, performance and test
coverage.

---
## 1.  Structural splits

### 1.1  `src/giflab/cli.py` (1,014 lines → 13 lines) 
**Status: ✅ COMPLETED** - All 7 commands successfully extracted

* **✅ Completed**: All CLI commands extracted into separate modules:
  * `debug_failures_cmd.py` (203 lines) - Full implementation with filtering & caching
  * `experiment_cmd.py` (160 lines) - Complex experimental pipeline testing
  * `run_cmd.py` (243 lines) - Core compression pipeline with dry-run support  
  * `tag_cmd.py` (146 lines) - Comprehensive tagging with validation
  * `view_failures_cmd.py` (147 lines) - Failure analysis with detailed reporting
  * `organize_cmd.py` (30 lines) - Directory structure creation
  * `select_pipelines_cmd.py` (35 lines) - Pipeline selection from experiments
* **✅ Completed**: Utility functions extracted to `giflab/cli/utils.py` (95 lines)
  * Error handling, worker validation, GPU detection, time estimation
  * Consistent CLI formatting and path management utilities
* **✅ Completed**: Modular CLI structure with `giflab/cli/__init__.py` re-exports
  * `cli.main()` remains the entry-point for backward compatibility
  * All commands properly registered and tested

### 1.2  `src/giflab/experimental.py` (3,755 lines → 506 lines, 182KB → ~28KB)
**Status: ✅ COMPLETED** - All 6 modules extracted + fully modular structure

* **✅ Completed extractions**:
  * `elimination_cache.py` – `PipelineResultsCache`, DB schema & batching (456 lines)
  * `elimination_errors.py` – Error classification and handling (136 lines)  
  * `synthetic_gifs.py` – Synthetic GIF specs & frame generation (754 lines)
  * `experimental/pareto.py` – `ParetoAnalyzer` class (~270 lines) ✅ **Priority 1 Complete**
  * `experimental/sampling.py` – All sampling strategies & `PipelineSampler` (~350 lines)
  * `experimental/runner.py` – Core `ExperimentalRunner` logic (~2983 lines) ✅ **Priority 2 Complete**
* **✅ Fully modular structure created**:
  * `experimental/__init__.py` – Exports all modular components including `ExperimentalRunner`
  * Multiple import paths supported for backwards compatibility
  * Original `experimental.py` reduced to 506 lines (imports + utility functions)
* **✅ Import conflicts resolved**:
  * `ExperimentalRunner` can be imported from `giflab.experimental` (backwards compatible)
  * `ExperimentalRunner` can be imported from `giflab.experimental.runner` (direct access)
  * All modular components accessible via `giflab.experimental` package
* **📊 Final metrics**: 6/6 extractions complete, ~3249 lines extracted (86% reduction), fully modular API

---
## 2.  Bug & cleanup follow-ups  
**Status: ✅ COMPLETED** - All cleanup tasks resolved

* **✅ Completed**: Hard-coded lossy levels now properly configurable via `config.py` (line 31)
* **✅ Completed**: Deduplicated imports: Optimized 8 source files with `import shutil` statements:
  * **Single-function imports**: 5 files now use `from shutil import copy/which/move/copy2`
  * **Multi-function imports**: 3 files use specific imports `from shutil import which, rmtree`
  * **Function-level imports**: 1 file moved to module-level for better performance
  * **Performance gain**: Cleaner imports, reduced namespace pollution, better IDE support
* **✅ Completed**: CSV writers now have `atexit` and signal handlers to prevent data loss:
  * **Signal handling**: SIGTERM, SIGINT, SIGHUP handlers for graceful cleanup
  * **Atexit handlers**: Automatic CSV flush and close on normal/abnormal termination
  * **Cross-platform**: Signal availability detection for Windows/Unix compatibility
  * **Logging**: Cleanup events are logged for debugging and monitoring

---
## 3.  Performance improvements
* **🎯 Critical**: Vectorize synthetic GIF generation in `synthetic_gifs.py`; current nested Python loops dominate
  generation time (esp. ≥ 500 × 500 frames). Located around lines 1650-1680 in current `experimental.py`.
  Possible approaches: NumPy array image generation or Pillow's `Image.effect_noise` where applicable.
* Offer multiprocessing for frame generation & pipeline execution, guarding DB
  writes with a process-safe queue (cache system already supports batching).

---
## 4.  Test coverage extensions
* Add tests for:
  * `get_*_version` helpers (stub binaries).
  * CLI commands via `click.testing.CliRunner`.
  * GPU metric fall-backs when OpenCV-CUDA absent.
* Convert remaining long-running integration tests to use the new *fast* set
  (fixtures & `fast_compress`).

---
## 5.  Documentation
* Promote this TODO into the public developer guide once items start moving.

---
## 6.  Implementation Priority & Guidance

### **Phase 1: High Impact, Low Risk** 
1. **Extract `ParetoAnalyzer`** → `experimental/pareto.py` (~270 lines, clean class boundary)
2. **Extract `experiment` command** → `cli/experiment_cmd.py` (~110 lines, most complex CLI command)
3. **Deduplicate shutil imports** (quick cleanup across 12 files)

### **Phase 2: Major Structural Changes**
4. **Split `ExperimentalRunner`**: Extract sampling strategies → `experimental/sampling.py` 
5. **Complete CLI extraction**: Remaining 5 commands → individual `cli/*_cmd.py` files
6. **Complete `ExperimentalRunner` split** → `experimental/runner.py` (largest remaining work)

### **Phase 3: Performance & Polish**
7. **Vectorize GIF generation** (performance critical)
8. **Enhanced test coverage** for new modular structure
9. **Add multiprocessing support** for pipeline execution

### **Risk Assessment**: ✅ **Low Risk**
- All proposed splits follow natural class/responsibility boundaries  
- Existing modular imports demonstrate the pattern works well
- Significant progress already made proves approach is sound
