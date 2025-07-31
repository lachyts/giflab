# GifLab – Refactor TODO Roadmap

> Last updated: <!--DATE_PLACEHOLDER-->  
> Author: development team (generated via Cursor assistant)

This document captures the medium-sized refactor items that surfaced during the
code-review on `main` (post-commit).  None of the tasks change current
behaviour, but they will greatly improve maintainability, performance and test
coverage.

---
## 1.  Structural splits

### 1.1  `src/giflab/cli.py` (≈ 1 kLoC)
* Extract each `@main.command()` into its own module below `giflab/cli/`:
  * `run_cmd.py`, `experiment_cmd.py`, `tag_cmd.py`, `view_failures_cmd.py`,
    `debug_failures_cmd.py`, `organise_cmd.py`, `select_pipelines_cmd.py`.
* Re-export commands via a lightweight `giflab/cli/__init__.py` so
  `cli.main()` remains the entry-point.
* Keep only option parsing in the command modules; move reusable logic (e.g.
  time-estimates, GPU detection) into `giflab/cli/utils.py` or service layers.

### 1.2  `src/giflab/pipeline_elimination.py` (≈ 3.7 kLoC)
* Split into sub-packages:
  * `analysis.py` – Pareto & elimination  
  * `sampling.py` – sampling strategies  
  * `cache.py` – `PipelineResultsCache`, DB schema & batching  
  * `gif_generator.py` – synthetic gif specs & frame generation  
  * `runner.py` – comprehensive testing loop & streaming CSV  
  * `pareto.py` – advanced frontier maths (optional)
* Provide a thin façade (`pipeline_elimination/__init__.py`) that re-exports
  the public classes `PipelineEliminator`, `EliminationResult`, etc.

---
## 2.  Bug & cleanup follow-ups
* Replace hard-coded lossy-level list `[0,40,120]` with values from config.
* Deduplicate any remaining double imports (e.g. `import shutil`).
* Ensure CSV writers always flush on `atexit` to prevent data loss on signals.

---
## 3.  Performance improvements
* Vectorise synthetic GIF generation; current nested Python loops dominate
  generation time (esp. ≥ 500 × 500 frames).  Possible approaches: NumPy array
  image generation or Pillow’s `Image.effect_noise` where applicable.
* Offer multiprocessing for frame generation & pipeline execution, guarding DB
  writes with a process-safe queue.

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
