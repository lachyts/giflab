from __future__ import annotations

"""Concrete wrappers that expose compression *variables* for each external tool.

Stage-2 of the matrix experimentation plan (see docs/technical/next-tools-priority.md).
The wrappers inherit from the abstract interfaces in ``tool_interfaces.py`` and
provide variable-specific ``apply`` methods. All external engines (gifsicle,
Animately, ImageMagick, FFmpeg, and gifski) are now fully implemented with
real functionality calling their respective helper modules.

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

# External engine imports - centralized at top level for consistency
from .external_engines.ffmpeg import frame_reduce as ffmpeg_frame_reduce
from .external_engines.ffmpeg import lossy_compress as ffmpeg_lossy_compress
from .external_engines.ffmpeg_enhanced import color_reduce_with_dithering as ffmpeg_color_reduce_with_dithering
from .external_engines.gifski import lossy_compress as gifski_lossy_compress
from .external_engines.imagemagick import frame_reduce as imagemagick_frame_reduce
from .external_engines.imagemagick import lossy_compress as imagemagick_lossy_compress
from .external_engines.imagemagick_enhanced import color_reduce_with_dithering as imagemagick_color_reduce_with_dithering
from .meta import extract_gif_metadata

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


class AnimatelyAdvancedLossyCompressor(LossyCompressionTool):
    NAME = "animately-advanced-lossy"
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
        """Apply Animately's advanced lossy compression with PNG sequence input.
        
        This uses the PNG sequence pipeline for superior compression quality,
        particularly effective for gradients and mixed content types.
        
        Args:
            input_path: Path to input GIF file
            output_path: Path to save compressed GIF
            params: Dict containing 'lossy_level', optionally 'colors' and 'png_sequence_dir'
            
        Returns:
            Dictionary with compression metadata
        """
        from .lossy import compress_with_animately_advanced_lossy
        
        if params is None or "lossy_level" not in params:
            raise ValueError("params must include 'lossy_level' for lossy compression")
        
        lossy_level = int(params["lossy_level"])
        color_keep_count = params.get("colors", None)
        if color_keep_count is not None:
            color_keep_count = int(color_keep_count)
        
        # Check if PNG sequence directory was provided by previous pipeline step
        png_sequence_dir = params.get("png_sequence_dir", None)
        if png_sequence_dir is not None:
            png_sequence_dir = Path(png_sequence_dir)
        
        return compress_with_animately_advanced_lossy(
            input_path,
            output_path,
            lossy_level=lossy_level,
            color_keep_count=color_keep_count,
            png_sequence_dir=png_sequence_dir,
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
        
        colors = int(params["colors"])
        dithering_method = params.get("dithering_method", "None")
        
        return imagemagick_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method=dithering_method,
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
        
        keep_ratio = float(params["ratio"])
        
        return imagemagick_frame_reduce(
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
        quality = 85  # default quality
        if params and "quality" in params:
            quality = int(params["quality"])
        
        return imagemagick_lossy_compress(
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
        colors = 256  # default color count
        fps = 15.0    # default fps for palette generation
        dithering_method = "none"  # default no dithering
        
        if params:
            if "colors" in params:
                colors = int(params["colors"])
            if "fps" in params:
                fps = float(params["fps"])
            if "dithering_method" in params:
                dithering_method = params["dithering_method"]
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method=dithering_method,
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
        if params is None or "ratio" not in params:
            raise ValueError("params must include 'ratio' for frame reduction")
        
        ratio = float(params["ratio"])
        
        # Get original FPS to calculate target FPS
        try:
            metadata = extract_gif_metadata(input_path)
            original_fps = metadata.orig_fps
        except Exception:
            # Fallback to default FPS if metadata extraction fails
            original_fps = 10.0
        
        # Calculate target FPS based on ratio
        target_fps = original_fps * ratio
        
        # Ensure minimum FPS to avoid issues
        target_fps = max(target_fps, 0.1)
        
        return ffmpeg_frame_reduce(
            input_path,
            output_path,
            fps=target_fps,
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
        q_scale = 4  # default quantiser value (1-31 range)
        
        if params:
            # Map lossy_level (0-100) to q_scale (1-31)
            if "lossy_level" in params:
                lossy_level = int(params["lossy_level"])
                # Map 0-100 to 31-1 (lower q_scale = higher quality)
                q_scale = max(1, min(31, 31 - int(lossy_level * 30 / 100)))
            elif "q_scale" in params:
                q_scale = int(params["q_scale"])
        
        return ffmpeg_lossy_compress(
            input_path,
            output_path,
            q_scale=q_scale,
        )

# ---------------------------------------------------------------------------
# ImageMagick Dithering-Specific Wrappers (Research-Based)
# ---------------------------------------------------------------------------

class ImageMagickColorReducerRiemersma(ColorReductionTool):
    """ImageMagick color reducer with Riemersma dithering (best all-around performer from research)."""
    NAME = "imagemagick-color-riemersma"
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
        
        colors = int(params["colors"])
        
        return imagemagick_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="Riemersma",
        )


class ImageMagickColorReducerFloydSteinberg(ColorReductionTool):
    """ImageMagick color reducer with Floyd-Steinberg dithering (standard high-quality baseline)."""
    NAME = "imagemagick-color-floyd"
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
        
        colors = int(params["colors"])
        
        return imagemagick_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="FloydSteinberg",
        )


class ImageMagickColorReducerNone(ColorReductionTool):
    """ImageMagick color reducer with no dithering (size priority baseline)."""
    NAME = "imagemagick-color-none"
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
        
        colors = int(params["colors"])
        
        return imagemagick_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="None",
        )


# ---------------------------------------------------------------------------
# FFmpeg Dithering-Specific Wrappers (Research-Based)
# ---------------------------------------------------------------------------

class FFmpegColorReducerSierra2(ColorReductionTool):
    """FFmpeg color reducer with Sierra2 dithering (excellent quality/size balance from research)."""
    NAME = "ffmpeg-color-sierra2"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="sierra2",
            fps=fps,
        )


class FFmpegColorReducerFloydSteinberg(ColorReductionTool):
    """FFmpeg color reducer with Floyd-Steinberg dithering (quality baseline)."""
    NAME = "ffmpeg-color-floyd"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="floyd_steinberg",
            fps=fps,
        )


class FFmpegColorReducerBayerScale0(ColorReductionTool):
    """FFmpeg color reducer with Bayer Scale 0 (2×2 matrix, poor quality from research - elimination candidate)."""
    NAME = "ffmpeg-color-bayer0"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="bayer:bayer_scale=0",
            fps=fps,
        )


class FFmpegColorReducerBayerScale1(ColorReductionTool):
    """FFmpeg color reducer with Bayer Scale 1 (4×4 matrix, higher quality Bayer variant from research)."""
    NAME = "ffmpeg-color-bayer1"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="bayer:bayer_scale=1",
            fps=fps,
        )


class FFmpegColorReducerBayerScale2(ColorReductionTool):
    """FFmpeg color reducer with Bayer Scale 2 (8×8 matrix, medium pattern from research)."""
    NAME = "ffmpeg-color-bayer2"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="bayer:bayer_scale=2",
            fps=fps,
        )


class FFmpegColorReducerBayerScale3(ColorReductionTool):
    """FFmpeg color reducer with Bayer Scale 3 (16×16 matrix, good balance from research)."""
    NAME = "ffmpeg-color-bayer3"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="bayer:bayer_scale=3",
            fps=fps,
        )


class FFmpegColorReducerBayerScale4(ColorReductionTool):
    """FFmpeg color reducer with Bayer Scale 4 (32×32 matrix, best compression for noisy content from research)."""
    NAME = "ffmpeg-color-bayer4"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="bayer:bayer_scale=4",
            fps=fps,
        )


class FFmpegColorReducerBayerScale5(ColorReductionTool):
    """FFmpeg color reducer with Bayer Scale 5 (64×64 matrix, maximum compression for noisy content from research)."""
    NAME = "ffmpeg-color-bayer5"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="bayer:bayer_scale=5",
            fps=fps,
        )


class FFmpegColorReducerNone(ColorReductionTool):
    """FFmpeg color reducer with no dithering (size priority baseline)."""
    NAME = "ffmpeg-color-none"
    COMBINE_GROUP = "ffmpeg"

    @classmethod
    def available(cls) -> bool:
        return discover_tool("ffmpeg").available

    @classmethod
    def version(cls) -> str:
        return discover_tool("ffmpeg").version or "unknown"

    def apply(self, input_path: Path, output_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params is None or "colors" not in params:
            raise ValueError("params must include 'colors' for color reduction")
        
        colors = int(params["colors"])
        fps = params.get("fps", 15.0)
        
        return ffmpeg_color_reduce_with_dithering(
            input_path,
            output_path,
            colors=colors,
            dithering_method="none",
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
        quality = 60  # default quality
        if params and "quality" in params:
            quality = int(params["quality"])
        
        # Check if PNG sequence was provided by previous pipeline step
        png_sequence_dir = None
        if params and "png_sequence_dir" in params:
            png_sequence_dir = Path(params["png_sequence_dir"])
        
        return gifski_lossy_compress(
            input_path,
            output_path,
            quality=quality,
            png_sequence_dir=png_sequence_dir,
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
