"""Run compression analysis command."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import click

from ..config import DEFAULT_COMPRESSION_CONFIG, PathConfig
from ..pipeline import CompressionPipeline
from ..utils_pipeline_yaml import read_pipelines_yaml
from .utils import (
    display_common_header,
    display_path_info,
    display_worker_info,
    ensure_csv_parent_exists,
    estimate_pipeline_time,
    generate_timestamped_csv_path,
    get_cpu_count,
    handle_generic_error,
    handle_keyboard_interrupt,
    validate_and_get_raw_dir,
    validate_and_get_worker_count,
)


@click.command()
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=0,
    help=f"Number of worker processes (default: {multiprocessing.cpu_count()} = CPU count)",
)
@click.option(
    "--resume/--no-resume", default=True, help="Skip existing renders (default: true)"
)
@click.option(
    "--fail-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Folder for bad GIFs (default: data/bad_gifs)",
)
@click.option(
    "--csv",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path (default: auto-date in data/csv/)",
)
@click.option("--dry-run", is_flag=True, help="List work only, don't execute")
@click.option(
    "--renders-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for rendered variants (default: data/renders)",
)
@click.option(
    "--detect-source-from-directory/--no-detect-source-from-directory",
    default=True,
    help="Detect source platform from directory structure (default: true)",
)
@click.option(
    "--pipelines",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML file listing pipeline identifiers to run (overrides engine grid)",
)
def run(
    raw_dir: Path,
    workers: int,
    resume: bool,
    fail_dir: Path | None,
    csv: Path | None,
    dry_run: bool,
    renders_dir: Path | None,
    detect_source_from_directory: bool,
    pipelines: Path | None,
):
    """Run compression analysis on GIFs in RAW_DIR.

    Generates a grid of compression variants for every GIF and writes
    one CSV row per variant with quality metrics and metadata.

    RAW_DIR: Directory containing original GIF files to analyze
    """
    try:
        # Validate RAW_DIR input
        validated_raw_dir = validate_and_get_raw_dir(raw_dir, require_gifs=not dry_run)

        # Validate worker count
        validated_workers = validate_and_get_worker_count(workers)

        # Create path configuration
        path_config = PathConfig()

        # Override paths if provided
        if fail_dir:
            path_config.BAD_GIFS_DIR = fail_dir
        if renders_dir:
            path_config.RENDERS_DIR = renders_dir

        selected_pipes = None
        if pipelines is not None:
            selected_pipes = read_pipelines_yaml(pipelines)

        pipeline = CompressionPipeline(
            compression_config=DEFAULT_COMPRESSION_CONFIG,
            path_config=path_config,
            workers=validated_workers,
            resume=resume,
            detect_source_from_directory=detect_source_from_directory,
            selected_pipelines=selected_pipes,
        )

        # Generate CSV path if not provided
        if csv is None:
            csv = generate_timestamped_csv_path(path_config.CSV_DIR)

        # Ensure CSV parent directory exists
        ensure_csv_parent_exists(csv)

        display_common_header("GifLab Compression Pipeline")
        display_path_info("Input directory", validated_raw_dir)
        display_path_info("Output CSV", csv, "📊")
        display_path_info("Renders directory", path_config.RENDERS_DIR, "🎬")
        display_path_info("Bad GIFs directory", path_config.BAD_GIFS_DIR, "❌")
        display_worker_info(validated_workers)
        click.echo(f"🔄 Resume: {'Yes' if resume else 'No'}")
        if pipelines:
            click.echo(f"🎛️  Selected pipelines: {len(selected_pipes)} from {pipelines}")
        click.echo(
            f"🗂️  Directory source detection: {'Yes' if detect_source_from_directory else 'No'}"
        )

        if dry_run:
            click.echo("🔍 DRY RUN MODE - Analysis only")
            _run_dry_run(pipeline, validated_raw_dir, csv)
        else:
            click.echo("🚀 Starting compression pipeline...")
            _run_pipeline(pipeline, validated_raw_dir, csv)

    except KeyboardInterrupt:
        handle_keyboard_interrupt("Pipeline")
    except Exception as e:
        handle_generic_error("Pipeline", e)


def _run_dry_run(pipeline: CompressionPipeline, raw_dir: Path, csv_path: Path):
    """Run dry-run analysis showing what work would be done."""

    # Discover GIFs
    click.echo("\n📋 Discovering GIF files...")
    gif_paths = pipeline.discover_gifs(raw_dir)

    if not gif_paths:
        click.echo(f"⚠️  No GIF files found in {raw_dir}")
        return

    click.echo(f"✅ Found {len(gif_paths)} GIF files")

    # Generate jobs
    click.echo("\n🔧 Generating compression jobs...")
    all_jobs = pipeline.generate_jobs(gif_paths)

    if not all_jobs:
        click.echo("⚠️  No valid compression jobs could be generated")
        return

    # Filter existing jobs if resume is enabled
    jobs_to_run = pipeline.filter_existing_jobs(all_jobs, csv_path)

    # Show summary
    engines = DEFAULT_COMPRESSION_CONFIG.ENGINES
    frame_ratios = DEFAULT_COMPRESSION_CONFIG.FRAME_KEEP_RATIOS
    color_counts = DEFAULT_COMPRESSION_CONFIG.COLOR_KEEP_COUNTS
    lossy_levels = DEFAULT_COMPRESSION_CONFIG.LOSSY_LEVELS

    variants_per_gif = (
        len(engines) * len(frame_ratios) * len(color_counts) * len(lossy_levels)
    )

    click.echo("\n📊 Compression Matrix:")
    click.echo(f"   • Engines: {', '.join(engines)}")
    click.echo(f"   • Frame ratios: {', '.join(f'{r:.2f}' for r in frame_ratios)}")
    click.echo(f"   • Color counts: {', '.join(str(c) for c in color_counts)}")
    click.echo(f"   • Lossy levels: {', '.join(str(level) for level in lossy_levels)}")
    click.echo(f"   • Variants per GIF: {variants_per_gif}")

    click.echo("\n📈 Job Summary:")
    click.echo(f"   • Total jobs: {len(all_jobs)}")
    click.echo(f"   • Jobs to run: {len(jobs_to_run)}")
    click.echo(f"   • Jobs to skip: {len(all_jobs) - len(jobs_to_run)}")

    if not jobs_to_run:
        click.echo("✅ All jobs already completed")
    else:
        estimated_time = estimate_pipeline_time(len(jobs_to_run))
        click.echo(f"⏱️  Estimated runtime: {estimated_time}")

    # Show sample jobs
    if jobs_to_run:
        click.echo("\n📝 Sample jobs to execute:")
        for i, job in enumerate(jobs_to_run[:5]):  # Show first 5 jobs
            click.echo(f"   {i+1}. {job.metadata.orig_filename}")
            click.echo(
                f"      • {job.engine}, lossy={job.lossy}, frames={job.frame_keep_ratio:.2f}, colors={job.color_keep_count}"
            )
            click.echo(f"      • Output: {job.output_path}")

        if len(jobs_to_run) > 5:
            click.echo(f"   ... and {len(jobs_to_run) - 5} more jobs")


def _run_pipeline(pipeline: CompressionPipeline, raw_dir: Path, csv_path: Path):
    """Execute the compression pipeline."""

    result = pipeline.run(raw_dir, csv_path)

    # Report results
    status = result["status"]
    processed = result["processed"]
    failed = result["failed"]
    skipped = result["skipped"]

    click.echo("\n📊 Pipeline Results:")
    click.echo(f"   • Status: {status}")
    click.echo(f"   • Processed: {processed}")
    click.echo(f"   • Failed: {failed}")
    click.echo(f"   • Skipped: {skipped}")

    if "total_jobs" in result:
        click.echo(f"   • Total jobs: {result['total_jobs']}")

    if "csv_path" in result:
        click.echo(f"   • Results saved to: {result['csv_path']}")

    if status == "completed":
        click.echo("✅ Pipeline completed successfully!")
    elif status == "no_files":
        click.echo("⚠️  No GIF files found to process")
    elif status == "no_jobs":
        click.echo("⚠️  No valid compression jobs could be generated")
    elif status == "all_complete":
        click.echo("✅ All jobs were already completed")
    elif status == "error":
        error_msg = result.get("error", "Unknown error")
        click.echo(f"❌ Pipeline failed: {error_msg}")
        sys.exit(1)
    else:
        click.echo(f"⚠️  Pipeline completed with status: {status}")
