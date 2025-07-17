"""Command-line interface for GifLab."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import click

from .config import (
    DEFAULT_COMPRESSION_CONFIG,
    PathConfig,
)
from .experiment import ExperimentalConfig, ExperimentalPipeline, create_experimental_pipeline
from .pipeline import CompressionPipeline
from .validation import validate_raw_dir, validate_worker_count, ValidationError
from .utils_pipeline_yaml import read_pipelines_yaml, write_pipelines_yaml
from .analysis_tools import performance_matrix
import pandas as pd


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
        try:
            validated_raw_dir = validate_raw_dir(raw_dir, require_gifs=not dry_run)
        except ValidationError as e:
            click.echo(f"❌ Invalid RAW_DIR: {e}", err=True)
            click.echo("💡 Please provide a valid directory containing GIF files", err=True)
            sys.exit(1)
        
        # Validate worker count
        try:
            validated_workers = validate_worker_count(workers)
        except ValidationError as e:
            click.echo(f"❌ Invalid worker count: {e}", err=True)
            sys.exit(1)
        
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
            timestamp = datetime.now().strftime("%Y%m%d")
            csv = path_config.CSV_DIR / f"results_{timestamp}.csv"

        # Ensure CSV parent directory exists
        csv.parent.mkdir(parents=True, exist_ok=True)

        click.echo("🎞️  GifLab Compression Pipeline")
        click.echo(f"📁 Input directory: {validated_raw_dir}")
        click.echo(f"📊 Output CSV: {csv}")
        click.echo(f"🎬 Renders directory: {path_config.RENDERS_DIR}")
        click.echo(f"❌ Bad GIFs directory: {path_config.BAD_GIFS_DIR}")
        click.echo(
            f"👥 Workers: {validated_workers if validated_workers > 0 else multiprocessing.cpu_count()}"
        )
        click.echo(f"🔄 Resume: {'Yes' if resume else 'No'}")
        if pipelines:
            click.echo(f"🎛️  Selected pipelines: {len(selected_pipes)} from {pipelines}")
        click.echo(f"🗂️  Directory source detection: {'Yes' if detect_source_from_directory else 'No'}")

        if dry_run:
            click.echo("🔍 DRY RUN MODE - Analysis only")
            _run_dry_run(pipeline, validated_raw_dir, csv)
        else:
            click.echo("🚀 Starting compression pipeline...")
            _run_pipeline(pipeline, validated_raw_dir, csv)

    except KeyboardInterrupt:
        click.echo("\n⏹️  Pipeline interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        sys.exit(1)


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
        estimated_time = len(jobs_to_run) * 2  # Rough estimate: 2 seconds per job
        estimated_hours = estimated_time / 3600
        click.echo(
            f"⏱️  Estimated runtime: ~{estimated_time}s (~{estimated_hours:.1f}h)"
        )

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
    output: Path | None,
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

        # Validate RAW_DIR input
        try:
            validated_raw_dir = validate_raw_dir(raw_dir, require_gifs=True)
        except ValidationError as e:
            click.echo(f"❌ Invalid RAW_DIR: {e}", err=True)
            click.echo("💡 Please provide a valid directory containing GIF files", err=True)
            sys.exit(1)
        
        # Validate worker count
        try:
            validated_workers = validate_worker_count(workers)
        except ValidationError as e:
            click.echo(f"❌ Invalid worker count: {e}", err=True)
            sys.exit(1)

        click.echo("🏷️  GifLab Comprehensive Tagging Pipeline")
        click.echo(f"📊 Input CSV: {csv_file}")
        click.echo(f"📁 Raw GIFs directory: {validated_raw_dir}")

        if validate_only:
            click.echo("🔍 Validation mode - checking CSV structure...")
            validation_report = validate_tagged_csv(csv_file)

            if validation_report["valid"]:
                click.echo("✅ CSV structure is valid")
                click.echo(f"   • {validation_report['tagging_columns_present']}/25 tagging columns present")
            else:
                click.echo("❌ CSV validation failed")
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
            click.echo("📄 Output CSV: auto-timestamped in same directory")

        click.echo(f"👥 Workers: {validated_workers} (parallel processing not yet implemented)")
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
            click.echo("\n🎯 Added 25 continuous scores for ML-ready compression optimization:")
            click.echo("   • Content classification (CLIP): 6 scores")
            click.echo("   • Quality assessment (Classical CV): 4 scores")
            click.echo("   • Technical characteristics (Classical CV): 5 scores")
            click.echo("   • Temporal motion analysis (Classical CV): 10 scores")
        elif status == "no_results":
            click.echo("⚠️  No compression results found in CSV")
        elif status == "no_original_gifs":
            click.echo("⚠️  No original GIFs found (engine='original')")
            click.echo("   💡 Tagging requires original records from compression pipeline")
        elif status == "no_successful_tags":
            click.echo("❌ No GIFs could be successfully tagged")
        else:
            click.echo(f"⚠️  Tagging completed with status: {status}")

    except KeyboardInterrupt:
        click.echo("\n⏹️  Tagging interrupted by user", err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(f"❌ Missing dependencies for tagging: {e}", err=True)
        click.echo("💡 Run: poetry install (to install torch and open-clip-torch)")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Tagging failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
def organize_directories(raw_dir: Path):
    """Create organized directory structure for source-based GIF collection.
    
    Creates subdirectories in RAW_DIR for different GIF sources:
    - tenor/      - GIFs from Tenor
    - animately/  - GIFs from Animately platform
    - tgif_dataset/ - GIFs from TGIF dataset
    - unknown/    - Ungrouped GIFs
    
    Each directory includes a README with organization guidelines.
    """
    from .directory_source_detection import create_directory_structure, get_directory_organization_help
    
    try:
        click.echo("🗂️  Creating directory structure for source organization...")
        create_directory_structure(raw_dir)
        
        click.echo("✅ Directory structure created successfully!")
        click.echo(f"📁 Organized directories in: {raw_dir}")
        click.echo("\n" + get_directory_organization_help())
        
    except Exception as e:
        click.echo(f"❌ Failed to create directory structure: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--gifs",
    "-g",
    type=int,
    default=10,
    help="Number of test GIFs to generate (default: 10)",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=0,
    help=f"Number of worker processes (default: {multiprocessing.cpu_count()} = CPU count)",
)
@click.option(
    "--sample-gifs-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing sample GIFs to use instead of generating new ones",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save experiment results (default: data/experimental/results)",
)
@click.option(
    "--strategies",
    type=click.Choice([
        "pure_gifsicle",
        "pure_animately",
        "animately_then_gifsicle",
        "gifsicle_dithered",
        "gifsicle_optimized",
        "all"
    ]),
    multiple=True,
    default=["all"],
    help="Compression strategies to test (default: all)",
)
@click.option(
    "--matrix/--no-matrix",
    default=False,
    help="Enable dynamic matrix mode (ignores --strategies)",
)
@click.option(
    "--no-analysis",
    is_flag=True,
    help="Disable detailed analysis report generation",
)
def experiment(
    gifs: int,
    workers: int,
    sample_gifs_dir: Path | None,
    output_dir: Path | None,
    strategies: tuple[str, ...],
    no_analysis: bool,
    matrix: bool,
):
    """Run experimental compression testing with diverse sample GIFs.
    
    This command tests different compression strategies on a small set of
    diverse GIFs to validate workflows and identify optimal parameters
    before running on large datasets.
    """
    try:
        # Validate worker count
        try:
            validated_workers = validate_worker_count(workers)
        except ValidationError as e:
            click.echo(f"❌ Invalid worker count: {e}", err=True)
            sys.exit(1)
        
        # Expand strategy selection
        all_strategies = [
            "pure_gifsicle",
            "pure_animately",
            "animately_then_gifsicle",
            "gifsicle_dithered",
            "gifsicle_optimized"
        ]
        
        if "all" in strategies:
            selected_strategies = all_strategies
        else:
            selected_strategies = list(strategies)
        
        # Create experimental configuration
        cfg = ExperimentalConfig(
            TEST_GIFS_COUNT=gifs,
            STRATEGIES=selected_strategies,
            ENABLE_DETAILED_ANALYSIS=not no_analysis,
            ENABLE_MATRIX_MODE=matrix,
        )
        
        # Override paths if provided
        if sample_gifs_dir:
            cfg.SAMPLE_GIFS_PATH = sample_gifs_dir
        if output_dir:
            cfg.RESULTS_PATH = output_dir
        
        # Create experimental pipeline
        pipeline = ExperimentalPipeline(cfg, validated_workers)
        
        click.echo("🧪 GifLab Experimental Testing")
        click.echo(f"📊 Test GIFs: {gifs}")
        click.echo(f"🛠️ Strategies: {', '.join(selected_strategies)}")
        click.echo(f"📁 Sample GIFs: {cfg.SAMPLE_GIFS_PATH}")
        click.echo(f"📈 Results: {cfg.RESULTS_PATH}")
        click.echo(f"👥 Workers: {validated_workers}")
        click.echo(f"📊 Analysis: {'Enabled' if not no_analysis else 'Disabled'}")
        
        # Load sample GIFs
        sample_gifs = None
        if sample_gifs_dir and sample_gifs_dir.exists():
            sample_gifs = list(sample_gifs_dir.glob("*.gif"))
            if not sample_gifs:
                click.echo(f"⚠️ No GIF files found in {sample_gifs_dir}")
                click.echo("Will generate test GIFs instead")
                sample_gifs = None
            else:
                click.echo(f"📂 Found {len(sample_gifs)} sample GIFs")
        
        # Run experiment
        click.echo("\n🚀 Starting experimental pipeline...")
        results_path = pipeline.run_experiment(sample_gifs)
        
        if results_path.exists():
            click.echo(f"\n✅ Experiment completed successfully!")
            click.echo(f"📊 Results saved to: {results_path}")
            
            # Show quick summary
            if not no_analysis:
                analysis_path = results_path.parent / "analysis_report.json"
                if analysis_path.exists():
                    click.echo(f"📈 Analysis report: {analysis_path}")
                    
        else:
            click.echo(f"\n❌ Experiment failed - no results generated")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Experiment failed: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# select-pipelines command
# ---------------------------------------------------------------------------


@main.command("select-pipelines")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--metric", default="ssim", help="Quality metric to optimise (default: ssim)")
@click.option("--top", default=1, help="Top-N pipelines to pick (per variable)")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=Path("winners.yaml"))
def select_pipelines(csv_file: Path, metric: str, top: int, output: Path):
    """Pick the best pipelines from an experiment CSV and write a YAML list."""

    click.echo("📊 Loading experiment results…")
    df = pd.read_csv(csv_file)

    if metric not in df.columns:
        click.echo(f"❌ Metric '{metric}' not found in CSV", err=True)
        raise SystemExit(1)

    click.echo(f"🔎 Selecting top {top} pipelines by {metric}…")
    grouped = df.groupby("strategy")[metric].mean().sort_values(ascending=False)
    winners = list(grouped.head(top).index)

    write_pipelines_yaml(output, winners)
    click.echo(f"✅ Wrote {len(winners)} pipelines to {output}")


if __name__ == "__main__":
    main()
