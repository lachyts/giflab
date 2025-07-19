# Fast Test-Suite Checklist (≤ 20 s wall-time)

## 1. Flatten the parameter grid
- [x] Add an **autouse** fixture in `conftest.py` that patches  
      `ExperimentalConfig` to return a **single-value** list for
      • `FRAME_KEEP_RATIOS` → `[1.0]`
      • `COLOR_KEEP_COUNTS` → `[256]`
      • `LOSSY_LEVELS`      → `[0]`
- [x] Inside the same fixture, `assert` that each list’s length is 1  
      (skip if `GIFLAB_FULL_MATRIX=1` is in env).
- [x] Also cap the *number of generated pipelines* in
      `dynamic_pipeline.generate_all_pipelines()` by honouring an env var  
      `GIFLAB_MAX_PIPES` (default **50**) to avoid huge cartesian products
      when new tool wrappers land.

## 2. External-binary integration tests
- [x] Keep input GIF tiny (≤ 50×50, ≤ 10 frames).  
- [x] Ensure every `subprocess.run()` has `timeout=<10` s.  
- [x] If any test grows, mark it `@pytest.mark.slow`.

## 3. Dynamic-pipeline unit tests
- [x] Assert in `test_dynamic_pipeline.py` that `len(generate_all_pipelines()) <= GIFLAB_MAX_PIPES` (default 50) instead of slicing.

## 4. Pytest configuration
- [x] Add / edit `pytest.ini`  
  ```ini
  [pytest]
  addopts = --durations=10
  markers =
      slow: long-running or external-binary tests
  ```
- [x] CI command runs `pytest -m "not slow"`.
- [x] Add a *secondary* marker `fast` so helpers like `fast_compress` can
      monkey-patch binaries only when desired (see §5).

## 5. Documentation & tooling helpers
- [x] Document this 20-second rule in `docs/testing-strategy.md`.
- [x] Provide helper fixture `fast_compress(monkeypatch)` that stubs
      `compress_with_gifsicle` / `compress_with_animately` with a no-op copy.

## 6. Continuous monitoring
- [x] Watch `pytest --durations=10` output; investigate any test > 2 s. 

## 7. Next-Stage Roadmap

### 7.1 Parallel execution stability
- [x] Refactor: move `execute_job()` to a *top-level* helper (or switch to
      `ThreadPoolExecutor`) so the fast suite doesn’t hit pickle errors on
      macOS / Windows.

### 7.2 Type-safety
- [x] Update `compress_with_gifsicle_extended` and callers to accept `color_keep_count: int | None` and mark accordingly.
- [x] Run mypy to confirm wrappers (`_BaseGifsicleLossyOptim` etc.) are clean.

### 7.3 Pipeline combinatorics guard-rails
- [x] Skip `NoOp*` wrappers in `generate_all_pipelines()` **or** rely on the
      `GIFLAB_MAX_PIPES` cap (see §1) to keep CI fast.
- [x] Update `test_dynamic_pipeline.py` to assert the cap via
      `len(pipelines) <= 50` instead of hard-coded slice.

### 7.4 Placeholder combiners
- [x] Extend `_noop_copy` to add basic size/ssim metrics so downstream analysis isn’t skewed.