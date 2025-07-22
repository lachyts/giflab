from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import subprocess
import time

__all__ = [
    "run_command",
]


def run_command(cmd: list[str], *, engine: str, output_path: Path, timeout: int | None = 60) -> dict[str, Any]:
    """Execute *cmd* and return GifLab-style metadata.

    The helper blocks until *cmd* completes, raises *RuntimeError* on non-zero
    exit status and captures the elapsed wall-clock time in milliseconds.

    Parameters
    ----------
    cmd
        Full command as a list of strings (preferred over shell=True).
    engine
        Human-readable engine key, e.g. "imagemagick", "ffmpeg", "gifski".
    output_path
        Path expected to be produced by the command – used to calculate the
        final file size in kilobytes.
    timeout
        Optional hard timeout (seconds) – *None* disables the limit.

    Returns
    -------
    dict
        Minimal metadata dict with the mandatory keys:
        ``render_ms``, ``engine``, ``command``, ``kilobytes``.
    """
    start = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    duration_ms = int((time.perf_counter() - start) * 1000)

    if completed.returncode != 0:
        raise RuntimeError(
            f"{engine} command failed (exit {completed.returncode}).\n\n"
            f"STDERR:\n{completed.stderr.strip()}"
        )

    try:
        size_kb = int(os.path.getsize(output_path) / 1024)
    except OSError:
        # File might not exist (e.g., intermediate palette). Return 0 and let
        # caller decide if that is acceptable.
        size_kb = 0

    return {
        "render_ms": duration_ms,
        "engine": engine,
        "command": " ".join(cmd),
        "kilobytes": size_kb,
    } 