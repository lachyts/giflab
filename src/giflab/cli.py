"""Command-line interface for GifLab."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from .config import (
    DEFAULT_COMPRESSION_CONFIG,
    PathConfig,
)

from .pipeline import CompressionPipeline
from .utils_pipeline_yaml import read_pipelines_yaml, write_pipelines_yaml
from .validation import ValidationError, validate_raw_dir, validate_worker_count
from giflab.combiner_registry import combiner_for
from giflab.experimental import ExperimentalRunner


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
    from .directory_source_detection import (
        create_directory_structure,
        get_directory_organization_help,
    )

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
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("experiment_results"),
    help="Base directory for timestamped experiment results (default: experiment_results)",
)
@click.option(
    "--sampling",
    type=click.Choice(['full', 'representative', 'factorial', 'progressive', 'targeted', 'quick']),
    default='representative',
    help="Sampling strategy to reduce testing time (default: representative)",
)
@click.option(
    "--threshold",
    "-t", 
    type=float,
    default=0.3,
    help="Quality threshold for pipeline elimination (default: 0.3, lower = stricter)",
)
@click.option(
    "--max-pipelines",
    type=int,
    default=0,
    help="Limit number of pipelines to test (0 = no limit, useful for quick tests)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from previous incomplete run (uses progress tracking)",
)
@click.option(
    "--estimate-time",
    is_flag=True,
    help="Show time estimate and exit (no actual testing)",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Enable GPU acceleration for quality metrics calculation (requires OpenCV with CUDA)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable cache and force re-running all pipeline tests (slower but fresh results)",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear the pipeline results cache before running (forces fresh start)",
)
def experiment(
    output_dir: Path,
    sampling: str,
    threshold: float,
    max_pipelines: int,
    resume: bool,
    estimate_time: bool,
    use_gpu: bool,
    no_cache: bool,
    clear_cache: bool,
):
    """Run comprehensive experimental pipeline testing with intelligent sampling.

    This command tests pipeline combinations on synthetic GIFs with diverse
    characteristics and eliminates underperforming pipelines based on quality
    metrics like SSIM, compression ratio, and processing speed.

    Results are saved in timestamped directories to preserve historical data.
    Smart caching avoids re-running identical pipeline tests.
    """
    from giflab.experimental import ExperimentalRunner
    from giflab.dynamic_pipeline import generate_all_pipelines

    # Create pipeline runner with cache settings
    use_cache = not no_cache  # Invert the no_cache flag
    runner = ExperimentalRunner(output_dir, use_gpu=use_gpu, use_cache=use_cache)
    
    # Clear cache if requested
    if clear_cache and runner.cache:
        click.echo("🗑️ Clearing pipeline results cache...")
        runner.cache.clear_cache()
    
    # Get pipeline count for estimation
    all_pipelines = generate_all_pipelines()
    
    # Apply intelligent sampling strategy
    if sampling != 'full':
        test_pipelines = runner.select_pipelines_intelligently(all_pipelines, sampling)
        strategy_info = runner.SAMPLING_STRATEGIES[sampling]
        click.echo(f"🧠 Sampling strategy: {strategy_info.name}")
        click.echo(f"📋 {strategy_info.description}")
    elif max_pipelines > 0 and max_pipelines < len(all_pipelines):
        test_pipelines = all_pipelines[:max_pipelines]
        click.echo(f"⚠️  Limited testing: Using {max_pipelines} of {len(all_pipelines)} available pipelines")
    else:
        test_pipelines = all_pipelines
        click.echo("🔬 Full comprehensive testing: Using all available pipelines")
    
    # Calculate total job estimates
    if sampling == 'targeted':
        synthetic_gifs = runner.get_targeted_synthetic_gifs()
    else:
        synthetic_gifs = runner.generate_synthetic_gifs()
    total_jobs = len(synthetic_gifs) * len(test_pipelines) * len(runner.test_params)
    estimated_time = runner._estimate_execution_time(total_jobs)
    
    click.echo("🧪 GifLab Experimental Pipeline Testing")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"🎯 Quality threshold: {threshold}")
    click.echo(f"📊 Total jobs: {total_jobs:,}")
    click.echo(f"⏱️  Estimated time: {estimated_time}")
    click.echo(f"🔄 Resume enabled: {resume}")
    
    # Display GPU status
    if use_gpu:
        try:
            import cv2
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                click.echo(f"🚀 GPU acceleration: Enabled ({cuda_devices} CUDA device(s) available)")
            else:
                click.echo(f"🔄 GPU acceleration: Requested but no CUDA devices found - will use CPU")
        except ImportError:
            click.echo(f"🔄 GPU acceleration: Requested but OpenCV CUDA not available - will use CPU")
        except Exception as e:
            click.echo(f"🔄 GPU acceleration: Requested but initialization failed - will use CPU")
    else:
        click.echo(f"📊 GPU acceleration: Disabled (using CPU processing)")
    
    if estimate_time:
        click.echo("✅ Time estimation complete. Use without --estimate-time to run actual analysis.")
        return
    
    click.echo("\n🚀 Running comprehensive experimental pipeline testing...")
    
    # Run the experimental analysis
    use_targeted_gifs = (sampling == 'targeted')
    elimination_result = runner.run_experimental_analysis(
            test_pipelines=test_pipelines,
            elimination_threshold=threshold,
            use_targeted_gifs=use_targeted_gifs
    )

    # Display results
    click.echo(f"\n📊 Experimental Results Summary:")
    click.echo(f"   📉 Eliminated pipelines: {len(elimination_result.eliminated_pipelines)}")
    click.echo(f"   ✅ Retained pipelines: {len(elimination_result.retained_pipelines)}")
    total_pipelines = len(elimination_result.eliminated_pipelines) + len(elimination_result.retained_pipelines)
    if total_pipelines > 0:
        elimination_rate = len(elimination_result.eliminated_pipelines) / total_pipelines * 100
        click.echo(f"   📈 Elimination rate: {elimination_rate:.1f}%")
    
    # Show top performers
    if elimination_result.retained_pipelines:
        click.echo(f"\n🏆 Top performing pipelines:")
        for i, pipeline in enumerate(list(elimination_result.retained_pipelines)[:5], 1):
            click.echo(f"   {i}. {pipeline}")
    
    click.echo(f"\n✅ Experimental analysis complete!")
    click.echo(f"📁 Results saved to: {output_dir}")
    click.echo(f"💡 Use 'giflab select-pipelines {output_dir}/latest/results.csv --top 3' to get production configs")


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




# ---------------------------------------------------------------------------
# failure analysis commands
# ---------------------------------------------------------------------------


@main.command()
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--error-type",
    type=click.Choice(['all', 'gifski', 'ffmpeg', 'imagemagick', 'gifsicle', 'animately', 'command', 'timeout', 'other']),
    default='all',
    help="Filter failures by error type (default: all)",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Maximum number of failures to show (default: 10, 0 = show all)",
)
@click.option(
    "--detailed",
    "-v",
    is_flag=True,
    help="Show detailed error information including stack traces",
)
def view_failures(results_dir: Path, error_type: str, limit: int, detailed: bool):
    """
    View detailed information about failed pipelines from elimination testing.
    
    This command analyzes the failed_pipelines.json file from elimination runs
    and provides a human-readable summary of what went wrong.
    
    Examples:
    
        # View top 10 failures from latest run
        giflab view-failures elimination_results/

        # View all gifski failures with details
        giflab view-failures elimination_results/ --error-type gifski --limit 0 --detailed

        # Quick overview of command execution failures
        giflab view-failures elimination_results/ --error-type command --limit 5
    """
    import json
    from collections import Counter
    
    failed_pipelines_file = results_dir / "failed_pipelines.json"
    if not failed_pipelines_file.exists():
        click.echo(f"❌ No failed pipelines file found at: {failed_pipelines_file}")
        click.echo("   Make sure you're pointing to a directory with elimination results.")
        return
    
    try:
        with open(failed_pipelines_file, 'r') as f:
            failed_pipelines = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"❌ Error reading failed pipelines file: {e}")
        return
    
    if not failed_pipelines:
        click.echo("✅ No failed pipelines found!")
        return
    
    # Filter by error type if specified
    if error_type != 'all':
        filtered_failures = []
        error_keywords = {
            'gifski': 'gifski',
            'ffmpeg': 'ffmpeg', 
            'imagemagick': 'imagemagick',
            'gifsicle': 'gifsicle',
            'animately': 'animately',
            'command': 'command failed',
            'timeout': 'timeout',
            'other': None  # Will be handled separately
        }

        keyword = error_keywords.get(error_type)
        for failure in failed_pipelines:
            error_msg = failure.get('error_message', '').lower()
            if error_type == 'other':
                # Show failures that don't match any specific tool
                if not any(tool in error_msg for tool in ['gifski', 'ffmpeg', 'imagemagick', 'gifsicle', 'animately', 'command failed', 'timeout']):
                    filtered_failures.append(failure)
            elif keyword and keyword in error_msg:
                filtered_failures.append(failure)

        failed_pipelines = filtered_failures
    
    # Apply limit
    if limit > 0:
        failed_pipelines = failed_pipelines[:limit]
    
    # Show summary statistics
    all_failures_file = results_dir / "failed_pipelines.json"
    with open(all_failures_file, 'r') as f:
        all_failures = json.load(f)
    
    click.echo(f"📊 Failure Analysis for {results_dir}")
    click.echo(f"   Total failures: {len(all_failures)}")
    if error_type != 'all':
        click.echo(f"   Showing {error_type} failures: {len(failed_pipelines)}")
    
    # Error type breakdown
    error_types = Counter()
    for failure in all_failures:
        error_msg = failure.get('error_message', '').lower()
        if 'gifski' in error_msg:
            error_types['gifski'] += 1
        elif 'ffmpeg' in error_msg:
            error_types['ffmpeg'] += 1
        elif 'imagemagick' in error_msg:
            error_types['imagemagick'] += 1
        elif 'gifsicle' in error_msg:
            error_types['gifsicle'] += 1
        elif 'animately' in error_msg:
            error_types['animately'] += 1
        elif 'command failed' in error_msg:
            error_types['command'] += 1
        elif 'timeout' in error_msg:
            error_types['timeout'] += 1
        else:
            error_types['other'] += 1
    
    click.echo(f"   Error type breakdown:")
    for error_type_name, count in error_types.most_common():
        click.echo(f"     {error_type_name}: {count}")
    
    if not failed_pipelines:
        click.echo(f"\n❌ No failures found matching filter: {error_type}")
        return
    
    # Show individual failures
    click.echo(f"\n🔍 Failed Pipeline Details:")
    for i, failure in enumerate(failed_pipelines, 1):
        pipeline_id = failure.get('pipeline_id', 'unknown')
        gif_name = failure.get('gif_name', 'unknown')
        error_msg = failure.get('error_message', 'No error message')
        tools = failure.get('tools_used', [])

        click.echo(f"\n{i:2d}. {pipeline_id}")
        click.echo(f"    GIF: {gif_name} ({failure.get('content_type', 'unknown')})")
        click.echo(f"    Tools: {', '.join(tools) if tools else 'unknown'}")
        click.echo(f"    Error: {error_msg}")

        if detailed:
            traceback_info = failure.get('error_traceback', '')
            if traceback_info:
                click.echo(f"    Traceback: {traceback_info}")
            
            test_params = failure.get('test_parameters', {})
            if test_params:
                click.echo(f"    Parameters: colors={test_params.get('colors')}, lossy={test_params.get('lossy')}, frame_ratio={test_params.get('frame_ratio')}")
            
            timestamp = failure.get('error_timestamp', failure.get('timestamp'))
            if timestamp:
                click.echo(f"    Time: {timestamp}")
    
    click.echo(f"\n💡 To see more details, use --detailed flag")
    click.echo(f"💡 To filter by error type, use --error-type <type>")


@main.command()
@click.option(
    "--cache-dir",
    "-c",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("elimination_results"),
    help="Directory containing the cache database (default: elimination_results)",
)
@click.option(
    "--error-type",
    "-e",
    type=str,
    help="Filter by specific error type (e.g., 'gifski', 'ffmpeg', 'other')",
)
@click.option(
    "--pipeline",
    "-p",
    type=str,
    help="Filter by specific pipeline ID",
)
@click.option(
    "--recent-hours",
    "-r",
    type=int,
    help="Only show failures from last N hours",
)
@click.option(
    "--summary",
    "-s",
    is_flag=True,
    help="Show summary statistics instead of detailed failures",
)
@click.option(
    "--clear-fixed",
    is_flag=True,
    help="Clear failures that should be fixed by recent code changes",
)
def debug_failures(
    cache_dir: Path,
    error_type: str | None,
    pipeline: str | None,
    recent_hours: int | None,
    summary: bool,
    clear_fixed: bool,
) -> None:
    """Debug pipeline elimination failures using the cached failure database.
    
    This command helps analyze and fix pipeline failures by providing detailed
    information about what's failing and why. Use the various filter options
    to narrow down to specific issues.
    
    Examples:
    \b
        # Show summary of all failures
        giflab debug-failures --summary

        # Show detailed gifski failures
        giflab debug-failures --error-type gifski

        # Show recent failures (last 24 hours)
        giflab debug-failures --recent-hours 24

        # Clear failures that should be fixed
        giflab debug-failures --clear-fixed
    """
    cache_db_path = cache_dir / "pipeline_results_cache.db"
    
    if not cache_db_path.exists():
        click.echo(f"❌ Cache database not found at {cache_db_path}")
        click.echo("   Run pipeline elimination to generate failure data")
        return
    
    try:
        from .experimental import PipelineResultsCache, ErrorTypes
        import git

        # Get current git commit
        try:
            repo = git.Repo(".")
            git_commit = repo.head.commit.hexsha[:8]
        except Exception:
            git_commit = "unknown"

        # Initialize cache connection
        cache = PipelineResultsCache(cache_db_path, git_commit)

        if clear_fixed:
            # Clear specific failure types that should be fixed by recent code changes
            import sqlite3
            
            with sqlite3.connect(cache_db_path) as conn:
                # Clear lossy level range errors (fixed by lossy_extended.py update)
                cursor = conn.execute("""
                    DELETE FROM pipeline_failures 
                    WHERE error_message LIKE '%between 0 and 200%'
                       OR error_message LIKE '%between 0 and 300%'
                """)
                lossy_cleared = cursor.rowcount
                
                # Clear frame size inconsistency errors (fixed by gifski.py update)
                cursor = conn.execute("""
                    DELETE FROM pipeline_failures 
                    WHERE error_message LIKE '%wrong size%'
                       OR error_message LIKE '%Frame % has wrong size%'
                """)
                frame_cleared = cursor.rowcount
                
                conn.commit()
                
                total_cleared = lossy_cleared + frame_cleared
                click.echo(f"✅ Cleared {total_cleared} fixed failures:")
                click.echo(f"   📊 Lossy level range errors: {lossy_cleared}")
                click.echo(f"   📐 Frame size errors: {frame_cleared}")
                
                if total_cleared > 0:
                    click.echo("   These failures should no longer occur with the recent fixes")
                else:
                    click.echo("   No fixed failures found to clear")
                    
                return

        # Query failures
        failures = cache.query_failures(
            error_type=error_type,
            pipeline_id=pipeline,
            recent_hours=recent_hours
    )

        if not failures:
            click.echo("✅ No pipeline failures found with the specified criteria")
            return

        if summary:
            # Show summary statistics
            from collections import Counter
            
            click.echo(f"🔍 Pipeline Failure Summary ({len(failures)} total failures)")
            click.echo("=" * 60)
            
            # Error type breakdown
            error_types = Counter(f['error_type'] for f in failures)
            click.echo(f"\n📊 Failure Types:")
            for error_type, count in error_types.most_common():
                percentage = (count / len(failures)) * 100
                click.echo(f"   {error_type}: {count} ({percentage:.1f}%)")
            
            # Most problematic pipelines
            pipeline_failures = Counter(f['pipeline_id'] for f in failures)
            click.echo(f"\n🔧 Most Problematic Pipelines:")
            for pipeline_id, count in pipeline_failures.most_common(5):
                click.echo(f"   {pipeline_id}: {count} failures")
            
            # Most problematic GIFs
            gif_failures = Counter(f['gif_name'] for f in failures)
            click.echo(f"\n🎬 Most Problematic GIFs:")
            for gif_name, count in gif_failures.most_common(5):
                click.echo(f"   {gif_name}: {count} failures")
            
            # Common error messages
            error_messages = Counter(f['error_message'][:100] for f in failures)  # Truncate long messages
            click.echo(f"\n⚠️  Most Common Error Messages:")
            for error_msg, count in error_messages.most_common(3):
                click.echo(f"   {error_msg}... : {count} occurrences")

        else:
            # Show detailed failures
            click.echo(f"🔍 Pipeline Failures ({len(failures)} found)")
            click.echo("=" * 80)
            
            for i, failure in enumerate(failures[:20]):  # Limit to first 20 for readability
                click.echo(f"\n[{i+1}] {failure['error_type'].upper()} | {failure['gif_name']} | {failure['pipeline_id']}")
                click.echo(f"    Error: {failure['error_message']}")
                click.echo(f"    Time: {failure['created_at']}")
                # Use semantic applied_* parameters if available, fallback to test_* for backward compatibility  
                colors = failure.get('applied_colors', failure.get('test_colors', 'N/A'))
                lossy = failure.get('applied_lossy', failure.get('test_lossy', 'N/A'))
                frames = failure.get('applied_frame_ratio', failure.get('test_frame_ratio', 'N/A'))
                click.echo(f"    Applied: colors={colors}, lossy={lossy}, frames={frames}")
                
                if failure['tools_used']:
                    try:
                        import json
                        tools = json.loads(failure['tools_used']) if isinstance(failure['tools_used'], str) else failure['tools_used']
                        click.echo(f"    Tools: {' → '.join(tools)}")
                    except:
                        click.echo(f"    Tools: {failure['tools_used']}")
            
            if len(failures) > 20:
                click.echo(f"\n... and {len(failures) - 20} more failures")
                click.echo("Use --summary for an overview or add filters to narrow results")
    
    except Exception as e:
        click.echo(f"❌ Error analyzing failures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
