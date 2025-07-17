from __future__ import annotations

"""Concrete wrappers that expose compression *variables* for each external tool.

Stage-2 of the matrix experimentation plan (see docs/technical/next-tools-priority.md).
The wrappers inherit from the abstract interfaces in ``tool_interfaces.py`` and
provide variable-specific ``apply`` methods.  For now, full implementations are
only provided for *gifsicle* and *animately* because the project already ships
robust helpers for these binaries.  ImageMagick, FFmpeg and gifski wrappers are
stubbed with clear *NotImplementedError* so that the capability registry can be
populated without breaking existing functionality.

Key design choices
------------------
1. **Single-variable contract** – each wrapper focuses on *one* compression
   variable, keeping logic simple and allowing the pipeline generator to mix
   them freely.
2. **Uniform params** – the ``params`` dict accepted by ``apply`` is small and
   self-documenting:

   • Color Reduction → ``{"colors": int}``
   • Frame Reduction → ``{"ratio": float}``
   • Lossy Compression → ``{"lossy_level": int}``

   Additional, tool-specific options (e.g. dithering) can be added later via
   optional keys (the wrappers ignore unknown keys for forward compatibility).
3. **Metadata passthrough** – the returned dict always contains ``render_ms``
   so that higher-level code does not need to special-case different tools.
"""

from pathlib import Path
from typing import Any, Dict

from .tool_interfaces import (
    ColorReductionTool,
    FrameReductionTool,
    LossyCompressionTool,
)
from .lossy import (
    compress_with_gifsicle,
    compress_with_animately,
    get_gifsicle_version,
    get_animately_version,
    _is_executable,  # pyright: ignore [private-use]
)
from .config import DEFAULT_ENGINE_CONFIG
from .system_tools import discover_tool, ToolInfo

# ---------------------------------------------------------------------------
# GIFSICLE
# ---------------------------------------------------------------------------

class GifsicleColorReducer(ColorReductionTool):
    NAME = "gifsicle-color"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        colors = int(params["colors"])
        return compress_with_gifsicle(
            input_path,
            output_path,
            lossy_level=0,
            frame_keep_ratio=1.0,
            color_keep_count=colors,
        )


class GifsicleFrameReducer(FrameReductionTool):
    NAME = "gifsicle-frame"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if params is None or "ratio" not in params:
            raise ValueError("params must include 'ratio' for frame reduction")
        ratio = float(params["ratio"])
        return compress_with_gifsicle(
            input_path,
            output_path,
            lossy_level=0,
            frame_keep_ratio=ratio,
            color_keep_count=None,
        )


class GifsicleLossyCompressor(LossyCompressionTool):
    NAME = "gifsicle-lossy"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if params is None or "lossy_level" not in params:
            raise ValueError("params must include 'lossy_level' for lossy compression")
        level = int(params["lossy_level"])
        return compress_with_gifsicle(
            input_path,
            output_path,
            lossy_level=level,
            frame_keep_ratio=1.0,
            color_keep_count=None,
        )

# ---------------------------------------------------------------------------
# ANIMATELY
# ---------------------------------------------------------------------------

class AnimatelyColorReducer(ColorReductionTool):
    NAME = "animately-color"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_animately_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        colors = int(params["colors"])
        return compress_with_animately(
            input_path,
            output_path,
            lossy_level=0,
            frame_keep_ratio=1.0,
            color_keep_count=colors,
        )


class AnimatelyFrameReducer(FrameReductionTool):
    NAME = "animately-frame"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_animately_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if params is None or "ratio" not in params:
            raise ValueError("params must include 'ratio' for frame reduction")
        ratio = float(params["ratio"])
        return compress_with_animately(
            input_path,
            output_path,
            lossy_level=0,
            frame_keep_ratio=ratio,
            color_keep_count=None,
        )


class AnimatelyLossyCompressor(LossyCompressionTool):
    NAME = "animately-lossy"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_animately_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if params is None or "lossy_level" not in params:
            raise ValueError("params must include 'lossy_level' for lossy compression")
        level = int(params["lossy_level"])
        return compress_with_animately(
            input_path,
            output_path,
            lossy_level=level,
            frame_keep_ratio=1.0,
            color_keep_count=None,
        )

# ---------------------------------------------------------------------------
# ImageMagick (stub)
# ---------------------------------------------------------------------------

class ImageMagickColorReducer(ColorReductionTool):
    NAME = "imagemagick-color"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("imagemagick").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("imagemagick").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("ImageMagick color reduction not implemented yet")


class ImageMagickFrameReducer(FrameReductionTool):
    NAME = "imagemagick-frame"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("imagemagick").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("imagemagick").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("ImageMagick frame reduction not implemented yet")


class ImageMagickLossyCompressor(LossyCompressionTool):
    NAME = "imagemagick-lossy"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("imagemagick").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("imagemagick").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("ImageMagick lossy compression not implemented yet")

# ---------------------------------------------------------------------------
# FFmpeg (stub)
# ---------------------------------------------------------------------------

class FFmpegColorReducer(ColorReductionTool):
    NAME = "ffmpeg-color"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("FFmpeg color reduction not implemented yet")


class FFmpegFrameReducer(FrameReductionTool):
    NAME = "ffmpeg-frame"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("FFmpeg frame reduction not implemented yet")


class FFmpegLossyCompressor(LossyCompressionTool):
    NAME = "ffmpeg-lossy"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("FFmpeg lossy compression not implemented yet")

# ---------------------------------------------------------------------------
# gifski (stub)
# ---------------------------------------------------------------------------

class GifskiLossyCompressor(LossyCompressionTool):
    NAME = "gifski-lossy"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("gifski").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("gifski").version or "unknown"

    def apply(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("gifski lossy compression not implemented yet")

# ---------------------------------------------------------------------------
# No-Operation fallbacks (always available)
# ---------------------------------------------------------------------------

import shutil
import time


class _BaseNoOpTool:
    """Mixin shared by no-operation tool wrappers."""

    @classmethod
    def available(cls) -> bool:  # noqa: D401
        return True  # always available

    @classmethod
    def version(cls) -> str:  # noqa: D401
        return "none"

    def _copy_file(self, src: Path, dst: Path) -> Dict[str, Any]:
        start = time.time()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return {"render_ms": int((time.time() - start) * 1000), "engine": self.NAME}


class NoOpColorReducer(_BaseNoOpTool, ColorReductionTool):
    """Placeholder that performs no color reduction (identity copy)."""

    NAME = "none-color"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._copy_file(input_path, output_path)


class NoOpFrameReducer(_BaseNoOpTool, FrameReductionTool):
    """Placeholder that performs no frame reduction (identity copy)."""

    NAME = "none-frame"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._copy_file(input_path, output_path)


class NoOpLossyCompressor(_BaseNoOpTool, LossyCompressionTool):
    """Placeholder that performs no lossy compression (identity copy)."""

    NAME = "none-lossy"

    def apply(self, input_path: Path, output_path: Path, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._copy_file(input_path, output_path) 