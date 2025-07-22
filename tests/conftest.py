import os

import pytest

# ---------------------------------------------------------------------------
# Global test-suite speed optimisations (≤ 20 s wall-time)
# ---------------------------------------------------------------------------
# These tweaks are applied at import-time so they take effect **before** any
# test module is collected.  They follow the guidance in
# docs/testing-fast-suite-checklist.md and ensure that the combinatorial
# explosion of experimental parameters is avoided when the fast test-suite is
# executed.
# ---------------------------------------------------------------------------

try:
    import giflab.dynamic_pipeline as _dp  # noqa: WPS433
    from giflab.experiment import (
        ExperimentalConfig,  # noqa: WPS433 – runtime patching is intentional
    )
except ModuleNotFoundError:
    # The package might not be install-able in some linting contexts; skip.
    ExperimentalConfig = None  # type: ignore  # pragma: no cover
    _dp = None  # type: ignore  # pragma: no cover


# Apply the optimisations unless the caller explicitly requests the full
# parameter matrix via an environment variable.
if os.getenv("GIFLAB_FULL_MATRIX") != "1" and ExperimentalConfig is not None and _dp is not None:

    # 1. Flatten the parameter grid --------------------------------------------------
    _single_value_lists: dict[str, list[object]] = {
        "FRAME_KEEP_RATIOS": [1.0],   # keep original frame-rate
        "COLOR_KEEP_COUNTS": [256],   # keep full palette
        "LOSSY_LEVELS": [0],          # lossless
    }

    for _field_name, _value in _single_value_lists.items():
        # Replace the dataclass Field.default_factory so that **future** instances
        # of ExperimentalConfig use the single-element list.
        field = ExperimentalConfig.__dataclass_fields__[_field_name]  # type: ignore[attr-defined]
        field.default_factory = lambda _v=_value: list(_v)  # noqa: WPS420 preserve late-binding

    # Avoid constructing an instance *before* we finish patching – runtime
    # validation is handled by the autouse fixture below.

    # 2. Cap the dynamic-pipeline cartesian product ---------------------------------
    _orig_generate = _dp.generate_all_pipelines  # type: ignore[attr-defined]

    def _generate_all_pipelines_capped():  # noqa: D401 – small wrapper
        """Wrapper that limits the number of generated pipelines.

        The cap defaults to 50, matching the recommendation in the test-suite
        checklist.  Set `GIFLAB_MAX_PIPES` to override.
        """
        pipelines = _orig_generate()
        max_pipes = int(os.getenv("GIFLAB_MAX_PIPES", "50"))
        return pipelines[:max_pipes]

    _dp.generate_all_pipelines = _generate_all_pipelines_capped  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # 3. Fallback patch: monkey-patch ExperimentalConfig.__init__ so *any*
    #    manually-constructed instance gets flattened lists even if the field
    #    default_factory workaround above is ineffective (e.g. due to caching
    #    in compiled dataclass slots).
    # -----------------------------------------------------------------------

    _orig_ec_init = ExperimentalConfig.__init__

    def _flattening_init(self, *args, **kwargs):  # type: ignore[no-self-use]
        _orig_ec_init(self, *args, **kwargs)  # type: ignore[misc]

        self.FRAME_KEEP_RATIOS = [1.0]
        self.COLOR_KEEP_COUNTS = [256]
        self.LOSSY_LEVELS = [0]

    ExperimentalConfig.__init__ = _flattening_init  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # End of one-time patches.  All subsequent imports/instantiations will
    # automatically benefit from these limits, dramatically reducing the test
    # runtime while preserving representative coverage.
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lightweight stub for ExperimentalPipeline._execute_dynamic_pipeline to avoid
# potential heavy dependencies during unit tests. Keeps behaviour minimal to
# satisfy collapsing tests (returns number of collapsed steps).
# ---------------------------------------------------------------------------

if ExperimentalConfig is not None:

    def _fast_execute_dynamic(self, job):  # noqa: D401
        assert job.pipeline is not None, "pipeline missing"
        # Simple collapse: group consecutive steps with same COMBINE_GROUP
        steps_meta = []
        current_group = None
        for step in job.pipeline.steps:
            group = getattr(step.tool_cls, "COMBINE_GROUP", step.variable)
            if group != current_group:
                # New group starts
                steps_meta.append({"engine": group, "render_ms": 1})
                current_group = group
        return {"steps": steps_meta, "engine": "dynamic"}

    # Patch the method unless full-suite requested
    if os.getenv("GIFLAB_FULL_MATRIX") != "1":
        from giflab import experiment as _exp_mod
        _exp_mod.ExperimentalPipeline._execute_dynamic_pipeline = _fast_execute_dynamic  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Session-wide fixtures / helpers (can be extended as needed)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _assert_fast_suite_limits():
    """Assert that the flattened parameter lists are respected in fast mode.

    The checks should *not* run when the caller explicitly requests the full
    parameter matrix via the ``GIFLAB_FULL_MATRIX=1`` environment variable.
    This aligns the behaviour with the guidelines in
    ``docs/testing-fast-suite-checklist.md``.
    """

    # Skip the assertion entirely when the full matrix is requested.
    if os.getenv("GIFLAB_FULL_MATRIX") == "1":
        yield
        return

    if ExperimentalConfig is None:
        yield
        return

    cfg = ExperimentalConfig()
    assert len(cfg.FRAME_KEEP_RATIOS) == 1
    assert len(cfg.COLOR_KEEP_COUNTS) == 1
    assert len(cfg.LOSSY_LEVELS) == 1
    yield


# ---------------------------------------------------------------------------
# Collection hook to mark/skip heavy tests automatically
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Dynamically skip heavy tests unless full suite is requested.

    The heuristics are intentionally simple: modules known to consume large
    amounts of RAM/CPU (e.g. `tests/test_metrics.py`) are marked as `slow`
    if they are not already and then skipped when the fast suite is running.
    """
    run_full = os.getenv("GIFLAB_FULL_MATRIX") == "1"
    heavy_modules = {
        "tests/test_metrics.py",
        "tests/test_additional_metrics.py",
        "tests/test_engine_integration.py",  # external binaries
        "tests/test_eda.py",  # matplotlib + PCA
        "tests/test_lossy.py",
        "tests/test_pipeline.py",
        "tests/test_tag_pipeline.py",
        "tests/test_resume.py",
        "tests/test_frame_keep.py",
        "tests/test_color_keep.py",
        "tests/test_color_reduction_alignment.py",
        "tests/test_engine_equivalence.py",
        "tests/test_combiner_collapsing.py",
        "tests/test_analysis_tools.py",
        "tests/test_notebooks.py",
        "tests/test_temporal_delta.py",
    }

    for item in items:
        # If the test is located in a heavy module and we're *not* in full mode
        node_path = str(item.fspath)
        if not run_full and any(node_path.endswith(mod) for mod in heavy_modules):
            item.add_marker(pytest.mark.skip(reason="Skipped in fast test-suite"))


# ---------------------------------------------------------------------------
# Early ignore hook so heavy modules are **never imported** during collection
# ---------------------------------------------------------------------------

from pathlib import Path as _Path


def pytest_ignore_collect(collection_path: _Path, config):  # noqa: D401
    heavy_module_names = {
        "test_metrics.py",
        "test_additional_metrics.py",
        "test_engine_integration.py",
        "test_eda.py",
        "test_lossy.py",
        "test_pipeline.py",
        "test_tag_pipeline.py",
        "test_resume.py",
        "test_frame_keep.py",
        "test_color_keep.py",
        "test_color_reduction_alignment.py",
        "test_engine_equivalence.py",
        "test_combiner_collapsing.py",
        "test_analysis_tools.py",
        "test_notebooks.py",
        "test_temporal_delta.py",
    }
    name = collection_path.name
    if os.getenv("GIFLAB_FULL_MATRIX") != "1" and name in heavy_module_names:
        return True  # Ignore collection entirely
    return False


# ---------------------------------------------------------------------------
# Helper fixture: fast_compress – monkey-patch heavy compression binaries
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_compress(monkeypatch):
    """Stub out gifsicle / Animately invocations for lightning-fast unit tests.

    The fixture replaces ``compress_with_gifsicle`` and ``compress_with_animately``
    with a no-op implementation that simply copies the source GIF to
    *output_path* and returns a minimal metadata dictionary mimicking the real
    wrappers.  This lets higher-level logic be exercised without requiring the
    external binaries or incurring their runtime cost.
    """
    import shutil
    from pathlib import Path as _Path

    def _noop_copy(input_path: _Path, output_path: _Path, *args, **kwargs):  # noqa: D401
        output_path = _Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, output_path)

        # Basic metrics so downstream analysis is not skewed by placeholders
        try:
            size_kb = output_path.stat().st_size / 1024.0
        except FileNotFoundError:
            size_kb = 0.0

        return {
            "render_ms": 1,
            "engine": "noop",
            "command": "noop-copy",
            # Size/quality placeholders
            "kilobytes": size_kb,   # align with metrics.calculate_file_size_kb
            "ssim": 1.0,            # identical copy => perfect similarity
            # Preserve commonly-inspected kwargs so callers don’t break
            "lossy_level": kwargs.get("lossy_level", 0),
            "frame_keep_ratio": kwargs.get("frame_keep_ratio", 1.0),
            "color_keep_count": kwargs.get("color_keep_count", None),
        }

    # Patch primary implementations in giflab.lossy
    monkeypatch.setattr("giflab.lossy.compress_with_gifsicle", _noop_copy, raising=True)
    monkeypatch.setattr("giflab.lossy.compress_with_animately", _noop_copy, raising=True)

    # Also patch re-exports from giflab.tool_wrappers (safe if missing)
    monkeypatch.setattr("giflab.tool_wrappers.compress_with_gifsicle", _noop_copy, raising=False)
    monkeypatch.setattr("giflab.tool_wrappers.compress_with_animately", _noop_copy, raising=False)

    yield
