"""Data-preparation helpers for metric post-processing.

Stage-5 of the Quality-Metrics expansion introduces lightweight utilities that
can be reused by notebooks, pipelines, and ML feature-engineering scripts.
The helpers are *pure* – they have no external dependencies beyond NumPy and
pandas (optional) and never mutate their inputs.

All functions operate on NumPy arrays for performance and accept Python lists
transparently via ``np.asarray``. Each helper raises ``ValueError`` on invalid
input shapes or parameters so callers receive immediate feedback.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import numpy as np

__all__ = [
    "minmax_scale",
    "zscore_scale",
    "normalise_metrics",
    "apply_confidence_weights",
    "clip_outliers",
]


# --------------------------------------------------------------------------- #
# Scaling helpers
# --------------------------------------------------------------------------- #


def _to_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert *values* to a 1-D ``np.ndarray`` of type ``float64``."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be a 1-D array or iterable")
    return arr


def minmax_scale(
    values: Iterable[float] | np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Scale *values* to the provided *feature_range* using min-max scaling.

    Identical values (zero range) return an array of mid-range constants.
    """
    arr = _to_array(values)
    if arr.size == 0:
        return arr.copy()

    data_min = float(np.min(arr))
    data_max = float(np.max(arr))
    min_val, max_val = feature_range

    if max_val <= min_val:
        raise ValueError("feature_range max must be greater than min")

    data_range = data_max - data_min
    if data_range == 0:
        # All values identical – map to centre of range
        return np.full_like(arr, (min_val + max_val) / 2.0)

    scaled = (arr - data_min) / data_range  # 0-1
    return scaled * (max_val - min_val) + min_val


def zscore_scale(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Standard-score (z-score) scaling: (x - μ) / σ.

    When the standard deviation is zero, returns zeros.
    """
    arr = _to_array(values)
    if arr.size == 0:
        return arr.copy()
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def normalise_metrics(
    metrics: Mapping[str, float],
    method: str = "zscore",
    inplace: bool = False,
) -> dict[str, float]:
    """Return a *new* dict with scaled metric values.

    Only numeric values are scaled; non-float entries are copied untouched.
    Supported *method*s: ``"zscore"`` (default) or ``"minmax"``.
    """
    if method not in {"zscore", "minmax"}:
        raise ValueError("method must be 'zscore' or 'minmax'")

    out: dict[str, float] = cast(
        dict[str, float], metrics if inplace else dict(metrics)
    )

    numeric_keys = [k for k, v in metrics.items() if isinstance(v, int | float)]
    values = np.array([metrics[k] for k in numeric_keys], dtype=np.float64)

    if method == "zscore":
        scaled = zscore_scale(values)
    else:
        scaled = minmax_scale(values)

    for key, val in zip(numeric_keys, scaled, strict=False):
        out[key] = float(val)

    return out


# --------------------------------------------------------------------------- #
# Confidence weighting
# --------------------------------------------------------------------------- #


def apply_confidence_weights(
    metrics: Mapping[str, float],
    confidences: Mapping[str, float],
    inplace: bool = False,
) -> dict[str, float]:
    """Multiply each metric by its confidence score.

    Missing confidence defaults to **1.0** (no change). Negative confidences
    raise ``ValueError``.
    """
    out: dict[str, float] = cast(
        dict[str, float], metrics if inplace else dict(metrics)
    )

    for key, value in metrics.items():
        conf = float(confidences.get(key, 1.0))
        if conf < 0:
            raise ValueError(f"confidence for {key} must be non-negative")
        if isinstance(value, int | float):
            out[key] = float(value) * conf
    return out


# --------------------------------------------------------------------------- #
# Outlier clipping
# --------------------------------------------------------------------------- #


def clip_outliers(
    values: Iterable[float] | np.ndarray,
    method: str = "iqr",
    factor: float = 1.5,
) -> np.ndarray:
    """Clip outliers in *values* using the specified *method*.

    Supported *method*s:
      • ``"iqr"`` – Inter-quartile range clipping with multiplier *factor* (Tukey).
      • ``"sigma"`` – Standard-deviation clipping (mean ± factor·σ).
    """
    arr = _to_array(values)
    if arr.size == 0:
        return arr.copy()

    if method == "iqr":
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
    elif method == "sigma":
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        lower = mean - factor * std
        upper = mean + factor * std
    else:
        raise ValueError("method must be 'iqr' or 'sigma'")

    return np.clip(arr, lower, upper)
