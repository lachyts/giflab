from __future__ import annotations

"""Schemas for GifLab data exports."""


from pydantic import BaseModel, ConfigDict, Field, ValidationError

# --------------------------------------------------------------------------- #
# Helper constants
# --------------------------------------------------------------------------- #

_BASE_METRICS: list[str] = [
    "ssim",
    "ms_ssim",
    "psnr",
    "mse",
    "rmse",
    "fsim",
    "gmsd",
    "chist",
    "edge_similarity",
    "texture_similarity",
    "sharpness_similarity",
    "temporal_consistency",
]


# --------------------------------------------------------------------------- #
# Metric record schema
# --------------------------------------------------------------------------- #

class MetricRecordV1(BaseModel):
    """Validated record for a single GIF comparison / metric extraction.

    Only a *minimal* set of core keys is declared – the model accepts arbitrary
    additional numeric keys (e.g. per-metric stats, positional samples, raw
    metrics) thanks to *extra = "allow"*. This keeps the schema flexible while
    still enforcing basic sanity constraints for the most critical fields.
    """

    render_ms: int = Field(ge=0, description="Time taken to compute metrics (ms)")
    kilobytes: float = Field(ge=0, description="Size of compressed GIF in KB")
    composite_quality: float = Field(
        ge=0.0, le=1.0, description="Weighted composite quality score (0-1)"
    )

    model_config = ConfigDict(extra="allow")

# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #

def validate_metric_record(data: dict) -> MetricRecordV1:
    """Validate *data* against :class:`MetricRecordV1`.

    Raises ``pydantic.ValidationError`` if the record is invalid.
    Returns the parsed model instance otherwise.
    """
    return MetricRecordV1.model_validate(data)


def is_valid_record(data: dict) -> bool:
    """Return *True* if *data* passes :class:`MetricRecordV1` validation."""
    try:
        validate_metric_record(data)
        return True
    except ValidationError:
        return False
