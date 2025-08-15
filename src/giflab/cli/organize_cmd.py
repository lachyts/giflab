"""Organize directories command for creating source-based directory structure."""

from pathlib import Path

import click

from .utils import handle_generic_error


@click.command("organize-directories")
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
def organize_directories(raw_dir: Path) -> None:
    """Create organized directory structure for source-based GIF collection.

    Creates subdirectories in RAW_DIR for different GIF sources:
    - tenor/      - GIFs from Tenor
    - animately/  - GIFs from Animately platform
    - tgif_dataset/ - GIFs from TGIF dataset
    - unknown/    - Ungrouped GIFs

    Each directory includes a README with organization guidelines.
    """
    try:
        from ..directory_source_detection import (
            create_directory_structure,
            get_directory_organization_help,
        )

        click.echo("🗂️  Creating directory structure for source organization...")
        create_directory_structure(raw_dir)

        click.echo("✅ Directory structure created successfully!")
        click.echo(f"📁 Organized directories in: {raw_dir}")
        click.echo("\n" + get_directory_organization_help())

    except Exception as e:
        handle_generic_error("Directory organization", e)
