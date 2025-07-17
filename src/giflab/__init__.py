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
