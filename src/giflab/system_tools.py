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

# Legacy tool discovery (fallback to PATH search)
_FALLBACK_TOOLS: dict[str, list[str]] = {
    "imagemagick": ["magick", "convert"],  # try "magick" first (newer) then fallback
    "ffmpeg": ["ffmpeg"],
    "ffprobe": ["ffprobe"],
    "gifski": ["gifski"],
    "gifsicle": ["gifsicle"],
    "animately": ["animately"],
}

_VERSION_PATTERNS: dict[str, str] = {
    "imagemagick": r"ImageMagick (\S+)",
    "ffmpeg": r"ffmpeg version (\S+)",
    "ffprobe": r"ffprobe version (\S+)",
    "gifski": r"gifski (\S+)",
    "gifsicle": r"LCDF Gifsicle (\S+)",
    "animately": r"animately (\S+)",
}

# Map tool keys to configuration attributes
_CONFIG_MAPPING: dict[str, str] = {
    "imagemagick": "IMAGEMAGICK_PATH",
    "ffmpeg": "FFMPEG_PATH",
    "ffprobe": "FFPROBE_PATH",
    "gifski": "GIFSKI_PATH",
    "gifsicle": "GIFSICLE_PATH",
    "animately": "ANIMATELY_PATH",
}


def discover_tool(tool_key: str, engine_config = None) -> ToolInfo:
    """Return *ToolInfo* for *tool_key* using configuration and fallback discovery.
    
    Args:
        tool_key: Tool identifier (imagemagick, ffmpeg, gifski, gifsicle, animately)
        engine_config: EngineConfig instance (uses DEFAULT_ENGINE_CONFIG if None)
        
    Returns:
        ToolInfo with availability and version information
    """
    if tool_key not in _FALLBACK_TOOLS and tool_key not in _CONFIG_MAPPING:
        raise ValueError(f"Unknown tool: {tool_key}")

    # Use provided config or import default
    if engine_config is None:
        from .config import DEFAULT_ENGINE_CONFIG
        engine_config = DEFAULT_ENGINE_CONFIG

    # Try configured path first if tool has config mapping
    if tool_key in _CONFIG_MAPPING:
        config_attr = _CONFIG_MAPPING[tool_key]
        configured_path = getattr(engine_config, config_attr, None)
        
        if configured_path:
            # Test the configured path directly
            if _which(configured_path):
                version_regex = _VERSION_PATTERNS.get(tool_key, r"(\d+\.\d+\.\d+)")
                version_cmd = [configured_path, "-version"]
                version = _run_version_cmd(version_cmd, version_regex)
                return ToolInfo(name=configured_path, available=True, version=version)
    
    # Fallback to PATH discovery for backward compatibility
    if tool_key in _FALLBACK_TOOLS:
        candidates = _FALLBACK_TOOLS[tool_key]
        version_regex = _VERSION_PATTERNS.get(tool_key, r"(\d+\.\d+\.\d+)")

        for candidate in candidates:
            path = _which(candidate)
            if path:
                version_cmd = [candidate, "-version"]
                version = _run_version_cmd(version_cmd, version_regex)
                return ToolInfo(name=candidate, available=True, version=version)

    # Not found via config or PATH
    fallback_name = (_FALLBACK_TOOLS.get(tool_key, [tool_key]))[0]
    return ToolInfo(name=fallback_name, available=False, version=None)


def verify_required_tools(engine_config = None) -> dict[str, ToolInfo]:
    """Ensure all configured binaries are available – raise on failure.

    Args:
        engine_config: EngineConfig instance (uses DEFAULT_ENGINE_CONFIG if None)

    Returns:
        Mapping of tool keys to ToolInfo instances
    """
    results: dict[str, ToolInfo] = {}
    
    # Verify all tools that have configuration mappings
    for key in _CONFIG_MAPPING:
        info = discover_tool(key, engine_config)
        info.require()  # Will raise if missing
        results[key] = info
    return results


def get_available_tools(engine_config = None) -> dict[str, ToolInfo]:
    """Get availability status for all supported tools without requiring them.
    
    Args:
        engine_config: EngineConfig instance (uses DEFAULT_ENGINE_CONFIG if None)
        
    Returns:
        Mapping of tool keys to ToolInfo instances (available=False if not found)
    """
    results: dict[str, ToolInfo] = {}
    
    # Check all tools that have configuration mappings
    for key in _CONFIG_MAPPING:
        info = discover_tool(key, engine_config)
        results[key] = info
    return results
