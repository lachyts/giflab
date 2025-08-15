"""Tagging command for adding comprehensive scores to compression results."""

from pathlib import Path

import click

from .utils import (
    display_common_header,
    display_path_info,
    handle_generic_error,
    handle_keyboard_interrupt,
    validate_and_get_raw_dir,
    validate_and_get_worker_count,
)


@click.command()
@click.argument(
    "csv_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path (default: auto-timestamped in same directory)",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=1,
    help="Number of worker processes (default: 1, parallel tagging not yet implemented)",
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate CSV structure, don't run tagging",
)
def tag(
    csv_file: Path,
    raw_dir: Path,
    output: Path | None,
    workers: int,
    validate_only: bool,
) -> None:
    """Add comprehensive tagging scores to existing compression results.

    Analyzes original GIF files and adds 25 continuous scores (0.0-1.0) to compression results:
    - 6 content classification scores (CLIP)
    - 4 quality/artifact assessment scores (Classical CV)
    - 5 technical characteristic scores (Classical CV)
    - 10 temporal motion analysis scores (Classical CV)

    CRITICAL: Tagging runs ONCE on original GIFs only, scores inherited by all variants.

    CSV_FILE: Path to existing compression results CSV file
    RAW_DIR: Directory containing original GIF files
    """
    try:
        from ..tag_pipeline import TaggingPipeline, validate_tagged_csv

        # Validate RAW_DIR input
        validated_raw_dir = validate_and_get_raw_dir(raw_dir, require_gifs=True)

        # Validate worker count
        validated_workers = validate_and_get_worker_count(workers)

        display_common_header("GifLab Comprehensive Tagging Pipeline")
        display_path_info("Input CSV", csv_file, "📊")
        display_path_info("Raw GIFs directory", validated_raw_dir)

        if validate_only:
            click.echo("🔍 Validation mode - checking CSV structure...")
            validation_report = validate_tagged_csv(csv_file)

            if validation_report["valid"]:
                click.echo("✅ CSV structure is valid")
                click.echo(
                    f"   • {validation_report['tagging_columns_present']}/25 tagging columns present"
                )
            else:
                click.echo("❌ CSV validation failed")
                if "error" in validation_report:
                    click.echo(f"   • Error: {validation_report['error']}")
                else:
                    click.echo(
                        f"   • Missing {validation_report['tagging_columns_missing']} tagging columns"
                    )
                    if validation_report["missing_columns"]:
                        click.echo(
                            f"   • Missing: {', '.join(validation_report['missing_columns'][:5])}..."
                        )
            return

        if output:
            display_path_info("Output CSV", output, "📄")
        else:
            click.echo("📄 Output CSV: auto-timestamped in same directory")

        click.echo(
            f"👥 Workers: {validated_workers} (parallel processing not yet implemented)"
        )
        click.echo("🎯 Will add 25 comprehensive tagging scores")

        # Initialize tagging pipeline
        click.echo("\n🔧 Initializing hybrid tagging system...")
        pipeline = TaggingPipeline(workers=validated_workers)

        # Run comprehensive tagging
        click.echo("🚀 Starting comprehensive tagging analysis...")
        result = pipeline.run(csv_file, validated_raw_dir, output)

        # Report results
        status = result["status"]

        click.echo("\n📊 Tagging Results:")
        click.echo(f"   • Status: {status}")

        if "total_results" in result:
            click.echo(f"   • Total compression results: {result['total_results']}")
        if "original_gifs" in result:
            click.echo(f"   • Original GIFs found: {result['original_gifs']}")
        if "tagged_successfully" in result:
            click.echo(f"   • Successfully tagged: {result['tagged_successfully']}")
        if "tagging_failures" in result:
            click.echo(f"   • Tagging failures: {result['tagging_failures']}")
        if "tagging_columns_added" in result:
            click.echo(f"   • Tagging columns added: {result['tagging_columns_added']}")
        if "output_path" in result:
            click.echo(f"   • Results saved to: {result['output_path']}")

        if status == "completed":
            click.echo("✅ Comprehensive tagging completed successfully!")
            click.echo(
                "\n🎯 Added 25 continuous scores for ML-ready compression optimization:"
            )
            click.echo("   • Content classification (CLIP): 6 scores")
            click.echo("   • Quality assessment (Classical CV): 4 scores")
            click.echo("   • Technical characteristics (Classical CV): 5 scores")
            click.echo("   • Temporal motion analysis (Classical CV): 10 scores")
        elif status == "no_results":
            click.echo("⚠️  No compression results found in CSV")
        elif status == "no_original_gifs":
            click.echo("⚠️  No original GIFs found (engine='original')")
            click.echo(
                "   💡 Tagging requires original records from compression pipeline"
            )
        elif status == "no_successful_tags":
            click.echo("❌ No GIFs could be successfully tagged")
        else:
            click.echo(f"⚠️  Tagging completed with status: {status}")

    except KeyboardInterrupt:
        handle_keyboard_interrupt("Tagging")
    except ImportError as e:
        click.echo(f"❌ Missing dependencies for tagging: {e}", err=True)
        click.echo(
            "💡 Run: poetry install (to install torch and open-clip-torch)", err=True
        )
        raise SystemExit(1)
    except Exception as e:
        handle_generic_error("Tagging", e)
