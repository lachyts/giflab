î# GifLab – Refactor TODO Roadmap

> Last updated: January 2025  
> Author: development team (updated via code analysis)

This document captures the medium-sized refactor items that surfaced during the
code-review on `main` (post-commit).  None of the tasks change current
behavior, but they will greatly improve maintainability, performance and test
coverage.

---
## 1.  Structural splits

### 1.1  `src/giflab/cli.py` (1,014 lines)
**Status: Partially started** - 1 of 7 commands already extracted

* **✅ Completed**: `debug_failures_cmd.py` (25 lines)
* **🔄 Remaining**: Extract remaining `@main.command()` functions into their own modules:
  * `run_cmd.py` (lines 70-269, ~200 lines) 
  * `experiment_cmd.py` (lines 505-616, ~110 lines) - **Priority 1: Most complex**
  * `tag_cmd.py` (lines 296-418, ~120 lines)
  * `view_failures_cmd.py` (lines 677-813, ~135 lines)
  * `organize_cmd.py` (lines 420-451, ~30 lines)
  * `select_pipelines_cmd.py` (lines 623-655, ~30 lines)
* Re-export commands via a lightweight `giflab/cli/__init__.py` so
  `cli.main()` remains the entry-point.
* Keep only option parsing in the command modules; move reusable logic (e.g.
  time-estimates, GPU detection) into `giflab/cli/utils.py` or service layers.

### 1.2  `src/giflab/experimental.py` (3,755 lines, 182KB)
**Status: Significantly progressed** - 3 of 6 modules already extracted

* **✅ Completed extractions**:
  * `elimination_cache.py` – `PipelineResultsCache`, DB schema & batching (456 lines)
  * `elimination_errors.py` – Error classification and handling (136 lines)  
  * `synthetic_gifs.py` – Synthetic GIF specs & frame generation (754 lines)
* **🔄 Remaining splits** (from current `experimental.py`):
  * `pareto.py` – `ParetoAnalyzer` class (lines 508-780, ~270 lines) - **Priority 1**
  * `sampling.py` – Sampling strategies from `ExperimentalRunner` (extract `SAMPLING_STRATEGIES` dict and related methods)
  * `runner.py` – Core `ExperimentalRunner` logic after sampling extraction (~2500+ lines) - **Priority 2**
* Update imports in remaining modules to use new structure
* Consider creating `experimental/__init__.py` façade for backward compatibility

---
## 2.  Bug & cleanup follow-ups
* **✅ Completed**: Hard-coded lossy levels now properly configurable via `config.py` (line 31)
* **🔄 Deduplicate imports**: Found 12 files with duplicate `import shutil` statements:
  * `combiner_registry.py`, `tool_wrappers.py`, `io.py`, `lossy.py`, `system_tools.py`
  * `experimental.py`, `external_engines/imagemagick*.py`, plus test files
  * Consider centralizing shutil usage or using qualified imports
* **⚠️ Verify**: CSV writers flush behavior on `atexit` to prevent data loss on signals

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
