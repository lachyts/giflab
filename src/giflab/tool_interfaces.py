from __future__ import annotations

"""Abstract interfaces for external compression tools used in GifLab.

Stage-1 foundation for the matrix-based experimentation framework described in
`docs/technical/next-tools-priority.md`.

Each compression *variable* (color reduction, frame reduction, lossy
compression) gets its own specialised interface so that capabilities can be
mixed-and-matched dynamically at runtime.

Concrete wrappers (e.g. `GifsicleColorReducer`, `FFmpegFrameReducer`) will be
implemented in later stages and registered with the experiment runner.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExternalTool(ABC):
    """Common behaviour for any external CLI-based tool.

    Sub-classes should *not* execute anything in the constructor – keep them
    lightweight so they can be instantiated freely by the pipeline generator.
    """

    #: Human-readable name, e.g. ``"gifsicle"`` or ``"ImageMagick"``
    NAME: str = "external-tool"

    @classmethod
    @abstractmethod
    def available(cls) -> bool:  # pragma: no cover – concrete impl later
        """Return ``True`` iff the binary can be found on the *current* system."""

    @classmethod
    @abstractmethod
    def version(cls) -> str:  # pragma: no cover – concrete impl later
        """Return the tool's version string ("unknown" if it can't be determined)."""

    # ---------------------------------------------------------------------
    # Public API to run the tool
    # ---------------------------------------------------------------------
    @abstractmethod
    def apply(
        self,
        input_path: Path,
        output_path: Path,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:  # pragma: no cover – concrete impl later
        """Run the tool on *input_path* and write results to *output_path*.

        ``params`` is a free-form dictionary interpreted by the concrete
        implementation.  Implementations *must* return a metadata dict (e.g.
        execution time, command used, additional statistics) so that the
        experiment runner can record provenance.
        """

    # ------------------------------------------------------------------
    # Optional helper for multi-step pipelines
    # ------------------------------------------------------------------
    def combines_with(self, other: ExternalTool) -> bool:
        """Return ``True`` if *self* can be merged with *other* into one step.

        The default implementation is conservative and returns ``False``.  Tool
        wrappers may override this (see docs/technical/next-tools-priority.md
        `combines: true` flag).
        """
        return False


# ---------------------------------------------------------------------------
# Specialised variable interfaces
# ---------------------------------------------------------------------------
class ColorReductionTool(ExternalTool):
    """Palette optimisation tools (e.g. *gifsicle --colors*, ImageMagick)."""

    VARIABLE = "color_reduction"


class FrameReductionTool(ExternalTool):
    """Temporal sampling tools (drop / merge frames)."""

    VARIABLE = "frame_reduction"


class LossyCompressionTool(ExternalTool):
    """Spatial quality degradation tools (lossy re-encoding)."""

    VARIABLE = "lossy_compression"
