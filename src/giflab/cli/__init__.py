"""CLI module for GifLab commands.

This module re-exports all command functions to maintain compatibility
with the main CLI entry point while keeping commands organized in separate modules.
"""

import click

from .debug_failures_cmd import debug_failures
from .experiment_cmd import experiment
from .organize_cmd import organize_directories
from .run_cmd import run
from .select_pipelines_cmd import select_pipelines
from .tag_cmd import tag
from .view_failures_cmd import view_failures


@click.group()
@click.version_option(version="0.1.0", prog_name="giflab")
def main() -> None:
    """🎞️ GifLab — GIF compression and analysis laboratory."""
    pass


# Register all commands from the modular CLI structure
main.add_command(run)
main.add_command(tag)
main.add_command(organize_directories)
main.add_command(experiment)
main.add_command(select_pipelines)
main.add_command(view_failures)
main.add_command(debug_failures)

__all__ = [
    "debug_failures",
    "experiment",
    "main",
    "organize_directories",
    "run",
    "select_pipelines",
    "tag",
    "view_failures",
]
