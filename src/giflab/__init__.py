"""GifLab - GIF compression and analysis laboratory."""

__version__ = "0.1.0"
__author__ = "GifLab Team"
__email__ = "team@giflab.example"

# Public re-exports for convenience ---------------------------------------------------

# NOTE: keep imports lightweight to avoid slow import-time side-effects.  Only import
# small, dependency-free symbols.

from .tool_interfaces import (
    ExternalTool,
    ColorReductionTool,
    FrameReductionTool,
    LossyCompressionTool,
)

from .system_tools import verify_required_tools, ToolInfo

# Stage-2: capability wrappers ------------------------------------------------

from .tool_wrappers import (
    GifsicleColorReducer,
    GifsicleFrameReducer,
    GifsicleLossyCompressor,
    AnimatelyColorReducer,
    AnimatelyFrameReducer,
    AnimatelyLossyCompressor,
    ImageMagickColorReducer,
    ImageMagickFrameReducer,
    ImageMagickLossyCompressor,
    FFmpegColorReducer,
    FFmpegFrameReducer,
    FFmpegLossyCompressor,
    GifskiLossyCompressor,
)

# Capability registry --------------------------------------------------------

from .capability_registry import tools_for as tools_for_variable, all_single_variable_strategies
