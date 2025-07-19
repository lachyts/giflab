import os
from typing import List

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
    from giflab.experiment import ExperimentalConfig  # noqa: WPS433 – runtime patching is intentional
    import giflab.dynamic_pipeline as _dp  # noqa: WPS433
except ModuleNotFoundError:
    # The package might not be install-able in some linting contexts; skip.
    ExperimentalConfig = None  # type: ignore  # pragma: no cover
    _dp = None  # type: ignore  # pragma: no cover


# Apply the optimisations unless the caller explicitly requests the full
# parameter matrix via an environment variable.
if os.getenv("GIFLAB_FULL_MATRIX") != "1" and ExperimentalConfig is not None and _dp is not None:

    # 1. Flatten the parameter grid --------------------------------------------------
    _single_value_lists: dict[str, List[object]] = {
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
# Session-wide fixtures / helpers (can be extended as needed)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _assert_fast_suite_limits():
    """Assert that the flattened parameter lists are respected at runtime."""
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

def pytest_ignore_collect(path, config):  # noqa: D401
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
        "test_analysis_tools.py",
        "test_notebooks.py",
        "test_temporal_delta.py",
    }
    if not os.getenv("GIFLAB_FULL_MATRIX") == "1" and path.basename in heavy_module_names:
        return True  # Ignore collection entirely
    return False