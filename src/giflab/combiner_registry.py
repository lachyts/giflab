from __future__ import annotations

"""Registry of *command combiners* for external tools.

A *combiner* receives:
    • input_path (Path) – source GIF
    • output_path (Path) – destination GIF
    • params (dict)       – may include
          "ratio"         float | None   – frame_keep_ratio
          "colors"        int | None     – palette size
          "lossy_level"   int | None     – lossy compression level
          additional tool-specific hints (e.g. "_opt_level" for gifsicle)

It returns the metadata dict produced by the underlying compression helper.

Adding support for a new tool family only requires registering one function
with the ``@register(group)`` decorator.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Protocol

CombinerFn = Callable[[Path, Path, Dict[str, Any]], Dict[str, Any]]

_COMBINERS: Dict[str, CombinerFn] = {}


def register(group: str) -> Callable[[CombinerFn], CombinerFn]:
    """Decorator to register a combiner for *group* (e.g. "gifsicle")."""

    def _decorator(fn: CombinerFn) -> CombinerFn:  # type: ignore[misc]
        _COMBINERS[group] = fn
        return fn

    return _decorator


def combiner_for(group: str | None) -> CombinerFn | None:  # noqa: D401
    """Return the combiner for *group* (or ``None`` if not registered)."""

    return _COMBINERS.get(group)


# ---------------------------------------------------------------------------
# Default combiners for built-in tools
# ---------------------------------------------------------------------------

# GIFSICLE -------------------------------------------------------------------

from .lossy_extended import (
    compress_with_gifsicle_extended,
    GifsicleDitheringMode,
    GifsicleOptimizationLevel,
)


@register("gifsicle")
def _combine_gifsicle(input_path: Path, output_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    lossy = int(params.get("lossy_level", 0) or 0)
    ratio = float(params.get("ratio", 1.0) or 1.0)
    colors = params.get("colors")  # may be None
    opt_level: GifsicleOptimizationLevel = params.get("_opt_level", GifsicleOptimizationLevel.BASIC)

    return compress_with_gifsicle_extended(
        input_path=input_path,
        output_path=output_path,
        lossy_level=lossy,
        frame_keep_ratio=ratio,
        color_keep_count=colors,
        optimization_level=opt_level,
        dithering_mode=GifsicleDitheringMode.NONE,
    )


# ANIMATELY ------------------------------------------------------------------

from .lossy import apply_compression_with_all_params, LossyEngine


@register("animately")
def _combine_animately(input_path: Path, output_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    lossy = int(params.get("lossy_level", 0) or 0)
    ratio = float(params.get("ratio", 1.0) or 1.0)
    colors = params.get("colors")

    return apply_compression_with_all_params(
        input_path=input_path,
        output_path=output_path,
        lossy_level=lossy,
        frame_keep_ratio=ratio,
        color_keep_count=colors,
        engine=LossyEngine.ANIMATELY,
    )

# ---------------------------------------------------------------------------
# Generic placeholder combiners for other tool families (copy pass-through)
# ---------------------------------------------------------------------------

import shutil, time


def _noop_copy(input_path: Path, output_path: Path, engine: str) -> Dict[str, Any]:
    start = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_path, output_path)
    return {"render_ms": int((time.time() - start) * 1000), "engine": engine}


@register("imagemagick")
def _combine_imagemagick(input_path: Path, output_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    return _noop_copy(input_path, output_path, "imagemagick")


@register("ffmpeg")
def _combine_ffmpeg(input_path: Path, output_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    return _noop_copy(input_path, output_path, "ffmpeg")


@register("gifski")
def _combine_gifski(input_path: Path, output_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    return _noop_copy(input_path, output_path, "gifski") 