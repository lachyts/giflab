from __future__ import annotations

"""Analysis & visualisation helpers (Stage-6).

These utilities live outside the critical runtime path so they favour
readability and pandas for data wrangling.  They can be imported from notebooks
or used in CLI utilities later.
"""

from pathlib import Path

import pandas as pd

from .dynamic_pipeline import Pipeline

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> pd.DataFrame:
    """Load a GifLab results CSV into a pandas *DataFrame*.

    Raises *FileNotFoundError* if the path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)

# ---------------------------------------------------------------------------
# Performance matrices
# ---------------------------------------------------------------------------

def _detect_variable(step_name: str) -> str:
    if step_name.endswith("Frame"):
        return "frame_reduction"
    if step_name.endswith("Color"):
        return "color_reduction"
    if step_name.endswith("Lossy"):
        return "lossy_compression"
    return "unknown"


def performance_matrix(df: pd.DataFrame, metric: str = "ssim") -> dict[str, pd.Series]:
    """Return performance *Series* (indexed by tool NAME) for each variable.

    Example output keys: ``frame_reduction``, ``color_reduction``.
    Values: pandas Series with index=tool NAME, value=mean(metric).
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in DataFrame")

    # Expect engine/tool names encoded in the 'engine' column OR fallback to wrapper names in 'strategy'
    name_col = "engine" if "engine" in df.columns else "strategy"

    out: dict[str, pd.Series] = {}
    for variable in ["frame_reduction", "color_reduction", "lossy_compression"]:
        mask = df["variable"] == variable if "variable" in df.columns else pd.Series(True, index=df.index)
        if mask.any():
            grouped = df[mask].groupby(name_col)[metric].mean().sort_values(ascending=False)
            out[variable] = grouped
    return out

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend_tools(df: pd.DataFrame, metric: str = "ssim", top_n: int = 1) -> dict[str, list[str]]:
    """Return *top_n* tool names per variable based on *metric* (higher is better)."""
    matrices = performance_matrix(df, metric)
    return {
        var: list(series.head(top_n).index)
        for var, series in matrices.items()
    }

# ---------------------------------------------------------------------------
# Pipeline visualisation
# ---------------------------------------------------------------------------

def pipeline_to_mermaid(pipeline: Pipeline) -> str:
    """Return Mermaid DSL representing *pipeline* as a flow chart."""
    lines = ["flowchart LR"]
    prev_id: str | None = None
    for i, step in enumerate(pipeline.steps):
        node_id = f"S{i}"
        label = f"{step.tool_cls.NAME}\n({step.variable})"
        lines.append(f"    {node_id}[\"{label}\"]")
        if prev_id is not None:
            lines.append(f"    {prev_id} --> {node_id}")
        prev_id = node_id
    return "\n".join(lines)

# Convenience ----------------------------------------------------------------

def save_pipeline_diagram(pipeline: Pipeline, out_path: Path) -> None:
    """Save a pipeline diagram as a `.mmd` Mermaid file for later rendering."""
    out_path.write_text(pipeline_to_mermaid(pipeline))
