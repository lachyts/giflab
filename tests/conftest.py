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

    # Safety check: constructing an instance now should respect the flattened lists
    _cfg = ExperimentalConfig()  # noqa: WPS122  – local sanity object
    assert len(_cfg.FRAME_KEEP_RATIOS) == 1, "FRAME_KEEP_RATIOS patch failed"
    assert len(_cfg.COLOR_KEEP_COUNTS) == 1, "COLOR_KEEP_COUNTS patch failed"
    assert len(_cfg.LOSSY_LEVELS) == 1, "LOSSY_LEVELS patch failed"

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