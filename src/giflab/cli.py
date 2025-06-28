"""Command-line interface for GifLab."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from .config import (
    CompressionConfig,
    PathConfig,
    DEFAULT_COMPRESSION_CONFIG,
    DEFAULT_PATH_CONFIG,
)
from .pipeline import CompressionPipeline


@click.group()
@click.version_option(version="0.1.0", prog_name="giflab")
def main():
    """🎞️ GifLab — GIF compression and analysis laboratory."""
    pass


@main.command()
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
def run(
    raw_dir: Path,
    workers: int,
    resume: bool,
    fail_dir: Optional[Path],
    csv: Optional[Path],
    dry_run: bool,
    renders_dir: Optional[Path],
):
    """Run compression analysis on GIFs in RAW_DIR.

    Generates a grid of compression variants for every GIF and writes
    one CSV row per variant with quality metrics and metadata.

    RAW_DIR: Directory containing original GIF files to analyze
    """
    try:
        # Create path configuration
        path_config = PathConfig()

        # Override paths if provided
        if fail_dir:
            path_config.BAD_GIFS_DIR = fail_dir
        if renders_dir:
            path_config.RENDERS_DIR = renders_dir

        # Create compression pipeline
        pipeline = CompressionPipeline(
            compression_config=DEFAULT_COMPRESSION_CONFIG,
            path_config=path_config,
            workers=workers,
            resume=resume,
        )

        # Generate CSV path if not provided
        if csv is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            csv = path_config.CSV_DIR / f"results_{timestamp}.csv"

        # Ensure CSV parent directory exists
        csv.parent.mkdir(parents=True, exist_ok=True)

        click.echo(f"🎞️  GifLab Compression Pipeline")
        click.echo(f"📁 Input directory: {raw_dir}")
        click.echo(f"📊 Output CSV: {csv}")
        click.echo(f"🎬 Renders directory: {path_config.RENDERS_DIR}")
        click.echo(f"❌ Bad GIFs directory: {path_config.BAD_GIFS_DIR}")
        click.echo(
            f"👥 Workers: {workers if workers > 0 else multiprocessing.cpu_count()}"
        )
        click.echo(f"🔄 Resume: {'Yes' if resume else 'No'}")

        if dry_run:
            click.echo(f"🔍 DRY RUN MODE - Analysis only")
            _run_dry_run(pipeline, raw_dir, csv)
        else:
            click.echo(f"🚀 Starting compression pipeline...")
            _run_pipeline(pipeline, raw_dir, csv)

    except KeyboardInterrupt:
        click.echo(f"\n⏹️  Pipeline interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        sys.exit(1)


def _run_dry_run(pipeline: CompressionPipeline, raw_dir: Path, csv_path: Path):
    """Run dry-run analysis showing what work would be done."""

    # Discover GIFs
    click.echo(f"\n📋 Discovering GIF files...")
    gif_paths = pipeline.discover_gifs(raw_dir)

    if not gif_paths:
        click.echo(f"⚠️  No GIF files found in {raw_dir}")
        return

    click.echo(f"✅ Found {len(gif_paths)} GIF files")

    # Generate jobs
    click.echo(f"\n🔧 Generating compression jobs...")
    all_jobs = pipeline.generate_jobs(gif_paths)

    if not all_jobs:
        click.echo(f"⚠️  No valid compression jobs could be generated")
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

    click.echo(f"\n📊 Compression Matrix:")
    click.echo(f"   • Engines: {', '.join(engines)}")
    click.echo(f"   • Frame ratios: {', '.join(f'{r:.2f}' for r in frame_ratios)}")
    click.echo(f"   • Color counts: {', '.join(str(c) for c in color_counts)}")
    click.echo(f"   • Lossy levels: {', '.join(str(l) for l in lossy_levels)}")
    click.echo(f"   • Variants per GIF: {variants_per_gif}")

    click.echo(f"\n📈 Job Summary:")
    click.echo(f"   • Total jobs: {len(all_jobs)}")
    click.echo(f"   • Jobs to run: {len(jobs_to_run)}")
    click.echo(f"   • Jobs to skip: {len(all_jobs) - len(jobs_to_run)}")

    if len(jobs_to_run) == 0:
        click.echo(f"✅ All jobs already completed")
    else:
        estimated_time = len(jobs_to_run) * 2  # Rough estimate: 2 seconds per job
        estimated_hours = estimated_time / 3600
        click.echo(
            f"⏱️  Estimated runtime: ~{estimated_time}s (~{estimated_hours:.1f}h)"
        )

    # Show sample jobs
    if jobs_to_run:
        click.echo(f"\n📝 Sample jobs to execute:")
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

    click.echo(f"\n📊 Pipeline Results:")
    click.echo(f"   • Status: {status}")
    click.echo(f"   • Processed: {processed}")
    click.echo(f"   • Failed: {failed}")
    click.echo(f"   • Skipped: {skipped}")

    if "total_jobs" in result:
        click.echo(f"   • Total jobs: {result['total_jobs']}")

    if "csv_path" in result:
        click.echo(f"   • Results saved to: {result['csv_path']}")

    if status == "completed":
        click.echo(f"✅ Pipeline completed successfully!")
    elif status == "no_files":
        click.echo(f"⚠️  No GIF files found to process")
    elif status == "no_jobs":
        click.echo(f"⚠️  No valid compression jobs could be generated")
    elif status == "all_complete":
        click.echo(f"✅ All jobs were already completed")
    elif status == "error":
        error_msg = result.get("error", "Unknown error")
        click.echo(f"❌ Pipeline failed: {error_msg}")
        sys.exit(1)
    else:
        click.echo(f"⚠️  Pipeline completed with status: {status}")


@main.command()
@click.argument(
    "csv_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "raw_dir", 
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--output", 
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path (default: auto-timestamped in same directory)"
)
@click.option(
    "--workers", 
    "-j",
    type=int, 
    default=1,
    help="Number of worker processes (default: 1, parallel tagging not yet implemented)"
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate CSV structure, don't run tagging"
)
def tag(
    csv_file: Path, 
    raw_dir: Path, 
    output: Optional[Path], 
    workers: int,
    validate_only: bool
):
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
        from .tag_pipeline import TaggingPipeline, validate_tagged_csv
        
        click.echo(f"🏷️  GifLab Comprehensive Tagging Pipeline")
        click.echo(f"📊 Input CSV: {csv_file}")
        click.echo(f"📁 Raw GIFs directory: {raw_dir}")
        
        if validate_only:
            click.echo(f"🔍 Validation mode - checking CSV structure...")
            validation_report = validate_tagged_csv(csv_file)
            
            if validation_report["valid"]:
                click.echo(f"✅ CSV structure is valid")
                click.echo(f"   • {validation_report['tagging_columns_present']}/25 tagging columns present")
            else:
                click.echo(f"❌ CSV validation failed")
                if "error" in validation_report:
                    click.echo(f"   • Error: {validation_report['error']}")
                else:
                    click.echo(f"   • Missing {validation_report['tagging_columns_missing']} tagging columns")
                    if validation_report['missing_columns']:
                        click.echo(f"   • Missing: {', '.join(validation_report['missing_columns'][:5])}...")
            return
        
        if output:
            click.echo(f"📄 Output CSV: {output}")
        else:
            click.echo(f"📄 Output CSV: auto-timestamped in same directory")
        
        click.echo(f"👥 Workers: {workers} (parallel processing not yet implemented)")
        click.echo(f"🎯 Will add 25 comprehensive tagging scores")
        
        # Initialize tagging pipeline
        click.echo(f"\n🔧 Initializing hybrid tagging system...")
        pipeline = TaggingPipeline(workers=workers)
        
        # Run comprehensive tagging
        click.echo(f"🚀 Starting comprehensive tagging analysis...")
        result = pipeline.run(csv_file, raw_dir, output)
        
        # Report results
        status = result["status"]
        
        click.echo(f"\n📊 Tagging Results:")
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
            click.echo(f"✅ Comprehensive tagging completed successfully!")
            click.echo(f"\n🎯 Added 25 continuous scores for ML-ready compression optimization:")
            click.echo(f"   • Content classification (CLIP): 6 scores")
            click.echo(f"   • Quality assessment (Classical CV): 4 scores")
            click.echo(f"   • Technical characteristics (Classical CV): 5 scores")
            click.echo(f"   • Temporal motion analysis (Classical CV): 10 scores")
        elif status == "no_results":
            click.echo(f"⚠️  No compression results found in CSV")
        elif status == "no_original_gifs":
            click.echo(f"⚠️  No original GIFs found (engine='original')")
            click.echo(f"   💡 Tagging requires original records from compression pipeline")
        elif status == "no_successful_tags":
            click.echo(f"❌ No GIFs could be successfully tagged")
        else:
            click.echo(f"⚠️  Tagging completed with status: {status}")
            
    except KeyboardInterrupt:
        click.echo(f"\n⏹️  Tagging interrupted by user", err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(f"❌ Missing dependencies for tagging: {e}", err=True)
        click.echo(f"💡 Run: poetry install (to install torch and clip-by-openai)")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Tagging failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
