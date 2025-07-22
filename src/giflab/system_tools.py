from __future__ import annotations

"""Utility helpers for verifying external system tools.

These lightweight checks ensure that ImageMagick, FFmpeg, gifski and other
binaries are present *before* the compression pipelines run.  They purposefully
avoid heavy dependencies and keep failures explicit so that CI and users get
fast feedback if the environment is mis-configured.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """Metadata for an external binary discovered on the system."""

    name: str
    available: bool
    version: str | None = None

    def require(self) -> None:
        """Raise *RuntimeError* if the tool isn’t available."""
        if not self.available:
            raise RuntimeError(
                f"Required tool '{self.name}' not found in PATH.\n"
                "📖 See docs/technical/next-tools-priority.md → Step 1 for setup instructions."
            )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _which(cmd: str) -> str | None:
    """Return full path if *cmd* is executable in $PATH, else *None*."""
    return shutil.which(cmd)


def _extract_version(output: str, pattern: str) -> str | None:
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def _run_version_cmd(cmd: list[str], regex: str) -> str | None:
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )
    except FileNotFoundError:
        return None  # Not installed
    except Exception:
        return None  # Fallback to generic "unknown" later

    return _extract_version(completed.stdout, regex) or _extract_version(completed.stderr, regex)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_REQUIRED_TOOLS: dict[str, list[str]] = {
    "imagemagick": ["magick", "convert"],  # try "magick" first (newer) then fallback
    "ffmpeg": ["ffmpeg"],
    "gifski": ["gifski"],
}


_VERSION_PATTERNS: dict[str, str] = {
    "imagemagick": r"ImageMagick (\S+)",
    "ffmpeg": r"ffmpeg version (\S+)",
    "gifski": r"gifski (\S+)",
}


def discover_tool(tool_key: str) -> ToolInfo:
    """Return *ToolInfo* for *tool_key* (imagemagick / ffmpeg / gifski)."""
    if tool_key not in _REQUIRED_TOOLS:
        raise ValueError(f"Unknown tool: {tool_key}")

    candidates = _REQUIRED_TOOLS[tool_key]
    version_regex = _VERSION_PATTERNS.get(tool_key, r"(\d+\.\d+\.\d+)")

    for candidate in candidates:
        path = _which(candidate)
        if path:
            version = _run_version_cmd([candidate, "-version" if candidate != "ffmpeg" else "-version"], version_regex)
            return ToolInfo(name=candidate, available=True, version=version)

    return ToolInfo(name=candidates[0], available=False, version=None)


def verify_required_tools() -> dict[str, ToolInfo]:
    """Ensure all *Step 1* binaries are available – raise on failure.

    Returns a mapping so callers can log versions.
    """
    results: dict[str, ToolInfo] = {}
    for key in _REQUIRED_TOOLS:
        info = discover_tool(key)
        info.require()  # Will raise if missing
        results[key] = info
    return results
