"""Compat entry point for GifLab CLI.

Exposes the same `main()` function but avoids the module/package name
collision that existed with the previous `giflab/cli.py` shim.
"""

from .cli import main  # re-export from the real CLI package

__all__ = ["main"]

if __name__ == "__main__":
    main()
