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
from typing import Any

from .config import DEFAULT_ENGINE_CONFIG
from .lossy import (
    _is_executable,  # pyright: ignore [private-use]
    compress_with_animately,
    compress_with_gifsicle,
    get_animately_version,
    get_gifsicle_version,
)

# Extended gifsicle helper for optimisation variants
from .lossy_extended import (
    GifsicleDitheringMode,
    GifsicleOptimizationLevel,
    compress_with_gifsicle_extended,
)
from .system_tools import discover_tool
from .tool_interfaces import (
    ColorReductionTool,
    ExternalTool,
    FrameReductionTool,
    LossyCompressionTool,
)

# ---------------------------------------------------------------------------
# GIFSICLE
# ---------------------------------------------------------------------------

class GifsicleColorReducer(ColorReductionTool):
    NAME = "gifsicle-color"
    COMBINE_GROUP = "gifsicle"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP


class GifsicleFrameReducer(FrameReductionTool):
    NAME = "gifsicle-frame"
    COMBINE_GROUP = "gifsicle"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP


class GifsicleLossyCompressor(LossyCompressionTool):
    NAME = "gifsicle-lossy"
    COMBINE_GROUP = "gifsicle"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP

# ---------------------------------------------------------------------------
# Gifsicle lossy wrappers per optimisation level
# ---------------------------------------------------------------------------


class _BaseGifsicleLossyOptim(LossyCompressionTool):
    """Shared logic for specific gifsicle optimisation-level lossy compressors."""

    COMBINE_GROUP = "gifsicle"

    _OPT_LEVEL: GifsicleOptimizationLevel = GifsicleOptimizationLevel.BASIC  # override

    @classmethod
    def available(cls) -> bool:  # noqa: D401
        return _is_executable(DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH)

    @classmethod
    def version(cls) -> str:  # noqa: D401
        try:
            return get_gifsicle_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # lossy_level required even though optimisation level is fixed per class
        if params is None or "lossy_level" not in params:
            raise ValueError("params must include 'lossy_level' for lossy compression")
        level = int(params["lossy_level"])

        # Preserve any prior palette reduction; if a 'colors' param is supplied we
        # forward it, otherwise pass *None* so gifsicle keeps the existing palette
        # size instead of restoring to 256 colours (avoids "palette bounce").
        color_keep = params.get("colors") if params else None

        return compress_with_gifsicle_extended(
            input_path=input_path,
            output_path=output_path,
            lossy_level=level,
            frame_keep_ratio=1.0,
            color_keep_count=color_keep,
            optimization_level=self._OPT_LEVEL,
            dithering_mode=GifsicleDitheringMode.NONE,
        )

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP


class GifsicleLossyBasic(_BaseGifsicleLossyOptim):
    NAME = "gifsicle-lossy-basic"
    _OPT_LEVEL = GifsicleOptimizationLevel.BASIC


class GifsicleLossyO1(_BaseGifsicleLossyOptim):
    NAME = "gifsicle-lossy-O1"
    _OPT_LEVEL = GifsicleOptimizationLevel.LEVEL1


class GifsicleLossyO2(_BaseGifsicleLossyOptim):
    NAME = "gifsicle-lossy-O2"
    _OPT_LEVEL = GifsicleOptimizationLevel.LEVEL2


class GifsicleLossyO3(_BaseGifsicleLossyOptim):
    NAME = "gifsicle-lossy-O3"
    _OPT_LEVEL = GifsicleOptimizationLevel.LEVEL3

# ---------------------------------------------------------------------------
# ANIMATELY
# ---------------------------------------------------------------------------

class AnimatelyColorReducer(ColorReductionTool):
    NAME = "animately-color"
    COMBINE_GROUP = "animately"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_animately_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP


class AnimatelyFrameReducer(FrameReductionTool):
    NAME = "animately-frame"
    COMBINE_GROUP = "animately"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_animately_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP


class AnimatelyLossyCompressor(LossyCompressionTool):
    NAME = "animately-lossy"
    COMBINE_GROUP = "animately"

    @classmethod
    def available(cls) -> bool:
        return _is_executable(DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH)

    @classmethod
    def version(cls) -> str:
        try:
            return get_animately_version()
        except Exception:
            return "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def combines_with(self, other: ExternalTool) -> bool:
        return getattr(other, "COMBINE_GROUP", None) == self.COMBINE_GROUP

# ---------------------------------------------------------------------------
# ImageMagick (stub)
# ---------------------------------------------------------------------------

class ImageMagickColorReducer(ColorReductionTool):
    NAME = "imagemagick-color"
    COMBINE_GROUP = "imagemagick"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("imagemagick").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("imagemagick").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        from .external_engines.imagemagick import color_reduce
        
        colors = int(params["colors"])
        dither = params.get("dither", False)
        
        return color_reduce(
            input_path,
            output_path,
            colors=colors,
            dither=dither,
        )


class ImageMagickFrameReducer(FrameReductionTool):
    NAME = "imagemagick-frame"
    COMBINE_GROUP = "imagemagick"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("imagemagick").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("imagemagick").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "ratio" not in params:
            raise ValueError("params must include 'ratio' for frame reduction")
        
        from .external_engines.imagemagick import frame_reduce
        
        keep_ratio = float(params["ratio"])
        
        return frame_reduce(
            input_path,
            output_path,
            keep_ratio=keep_ratio,
        )


class ImageMagickLossyCompressor(LossyCompressionTool):
    NAME = "imagemagick-lossy"
    COMBINE_GROUP = "imagemagick"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("imagemagick").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("imagemagick").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .external_engines.imagemagick import lossy_compress
        
        quality = 85  # default quality
        if params and "quality" in params:
            quality = int(params["quality"])
        
        return lossy_compress(
            input_path,
            output_path,
            quality=quality,
        )

# ---------------------------------------------------------------------------
# FFmpeg (stub)
# ---------------------------------------------------------------------------

class FFmpegColorReducer(ColorReductionTool):
    NAME = "ffmpeg-color"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .external_engines.ffmpeg import color_reduce
        
        fps = 15.0  # default fps for palette generation
        if params and "fps" in params:
            fps = float(params["fps"])
        
        return color_reduce(
            input_path,
            output_path,
            fps=fps,
        )


class FFmpegFrameReducer(FrameReductionTool):
    NAME = "ffmpeg-frame"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "fps" not in params:
            raise ValueError("params must include 'fps' for frame reduction")
        
        from .external_engines.ffmpeg import frame_reduce
        
        fps = float(params["fps"])
        
        return frame_reduce(
            input_path,
            output_path,
            fps=fps,
        )


class FFmpegLossyCompressor(LossyCompressionTool):
    NAME = "ffmpeg-lossy"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .external_engines.ffmpeg import lossy_compress
        
        qv = 30  # default quantiser value
        fps = 15.0  # default fps
        
        if params:
            if "qv" in params:
                qv = int(params["qv"])
            if "fps" in params:
                fps = float(params["fps"])
        
        return lossy_compress(
            input_path,
            output_path,
            qv=qv,
            fps=fps,
        )

# ---------------------------------------------------------------------------
# gifski (stub)
# ---------------------------------------------------------------------------

class GifskiLossyCompressor(LossyCompressionTool):
    NAME = "gifski-lossy"
    COMBINE_GROUP = "gifski"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("gifski").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("gifski").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .external_engines.gifski import lossy_compress
        
        quality = 60  # default quality
        if params and "quality" in params:
            quality = int(params["quality"])
        
        return lossy_compress(
            input_path,
            output_path,
            quality=quality,
        )

# ---------------------------------------------------------------------------
# No-Operation fallbacks (always available)
# ---------------------------------------------------------------------------

import shutil
import time


class _BaseNoOpTool:
    """Mixin shared by no-operation tool wrappers."""

    NAME: str = "noop-tool"

    @classmethod
    def available(cls) -> bool:  # noqa: D401
        return True  # always available

    @classmethod
    def version(cls) -> str:  # noqa: D401
        return "none"

    def _copy_file(self, src: Path, dst: Path) -> dict[str, Any]:
        start = time.time()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return {"render_ms": int((time.time() - start) * 1000), "engine": self.NAME}


class NoOpColorReducer(_BaseNoOpTool, ColorReductionTool):
    """Placeholder that performs no color reduction (identity copy)."""

    NAME = "none-color"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._copy_file(input_path, output_path)


class NoOpFrameReducer(_BaseNoOpTool, FrameReductionTool):
    """Placeholder that performs no frame reduction (identity copy)."""

    NAME = "none-frame"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._copy_file(input_path, output_path)


class NoOpLossyCompressor(_BaseNoOpTool, LossyCompressionTool):
    """Placeholder that performs no lossy compression (identity copy)."""

    NAME = "none-lossy"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._copy_file(input_path, output_path)
