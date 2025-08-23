"""Comprehensive GIF compression analysis and optimization command."""

import multiprocessing
from pathlib import Path
from typing import Any

import click

from .utils import (
    handle_generic_error,
    handle_keyboard_interrupt,
    validate_and_get_raw_dir,
    validate_and_get_worker_count,
)


@click.command()
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
)
@click.option(
    "--output-dir",
    "-o", 
    type=click.Path(path_type=Path),
    default=Path("results/runs"),
    help="Base directory for timestamped results (default: results/runs)",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=0,
    help=f"Number of worker processes (default: {multiprocessing.cpu_count()} = CPU count)",
)
@click.option(
    "--sampling",
    type=click.Choice(["representative", "full", "factorial", "progressive", "targeted", "quick"]),
    default="representative",
    help="Sampling strategy to optimize testing time (default: representative)",
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
    help="Enable GPU acceleration for quality metrics calculation",
)
@click.option(
    "--use-cache",
    is_flag=True,
    help="Enable cache for pipeline test results (faster but may use stale results)",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear the pipeline results cache before running (forces fresh start)",
)
@click.option(
    "--preset",
    "-p",
    type=str,
    default=None,
    help="Use targeted preset (e.g., 'frame-focus', 'color-optimization'). Use 'list' to see available presets.",
)
@click.option(
    "--list-presets",
    is_flag=True,
    help="List all available presets and exit",
)
def run(
    raw_dir: Path | None,
    output_dir: Path,
    workers: int,
    sampling: str,
    threshold: float,
    max_pipelines: int,
    resume: bool,
    estimate_time: bool,
    use_gpu: bool,
    use_cache: bool,
    clear_cache: bool,
    preset: str,
    list_presets: bool,
) -> None:
    """Run comprehensive GIF compression analysis and optimization.

    This command tests pipeline combinations on GIFs in RAW_DIR with diverse
    characteristics and eliminates underperforming pipelines based on quality
    metrics like SSIM, compression ratio, and processing speed.

    Results are saved in timestamped directories to preserve historical data.
    Smart caching and resume functionality make large analysis runs efficient.

    RAW_DIR: Directory containing original GIF files to analyze
    """
    try:
        # Import from core module
        from ..core import GifLabRunner

        # Handle operations that don't need RAW_DIR
        if list_presets:
            # Create pipeline runner with settings
            runner = GifLabRunner(output_dir, use_gpu=use_gpu, use_cache=use_cache)
            try:
                presets = runner.list_available_presets()
                click.echo("🎯 Available Presets:")
                click.echo()
                for preset_id, description in presets.items():
                    click.echo(f"  {preset_id}")
                    click.echo(f"    {description}")
                    click.echo()
                return
            except Exception as e:
                click.echo(f"❌ Error listing presets: {e}")
                return

        # Validate RAW_DIR is provided for other operations (except estimate-time with presets)
        if raw_dir is None:
            if estimate_time and preset:
                # Allow estimate-time with presets to work without RAW_DIR
                # Create a dummy path that won't be used
                validated_raw_dir = Path(".")
            else:
                click.echo("❌ Error: RAW_DIR argument is required unless using --list-presets")
                click.echo("💡 Use 'giflab run --help' for usage information")
                return
        else:
            # Validate RAW_DIR input
            validated_raw_dir = validate_and_get_raw_dir(raw_dir, require_gifs=not estimate_time)

        # Validate worker count
        validated_workers = validate_and_get_worker_count(workers)

        # Create pipeline runner with settings
        runner = GifLabRunner(output_dir, use_gpu=use_gpu, use_cache=use_cache)

        # Clear cache if requested
        if clear_cache and runner.cache:
            click.echo("🗑️ Clearing pipeline results cache...")
            runner.cache.clear_cache()

        # Determine analysis approach
        if preset:
            # Use targeted preset approach
            try:
                click.echo(f"🎯 Using targeted preset: {preset}")
                
                if estimate_time:
                    # Quick time estimate for preset
                    synthetic_gifs = runner.get_targeted_synthetic_gifs()
                    test_pipelines = runner.generate_targeted_pipelines(preset)
                    total_jobs = len(synthetic_gifs) * len(test_pipelines) * len(runner.test_params)
                    estimated_time = runner._estimate_execution_time(total_jobs)
                    
                    click.echo(f"📊 Total jobs: {total_jobs:,}")
                    click.echo(f"⏱️ Estimated time: {estimated_time}")
                    click.echo("✅ Time estimation complete. Remove --estimate-time to run analysis.")
                    return
                
                # Run targeted analysis
                elimination_result = runner.run_targeted_experiment(
                    preset_id=preset,
                    quality_threshold=threshold,
                    use_targeted_gifs=True,
                )
                analysis_type = f"targeted preset ({preset})"
                
            except Exception as e:
                click.echo(f"❌ Error with preset '{preset}': {e}")
                click.echo("💡 Use --list-presets to see available presets")
                return
        else:
            # Use traditional sampling approach
            from ..dynamic_pipeline import generate_all_pipelines
            
            all_pipelines = generate_all_pipelines()
            
            if sampling != "full":
                test_pipelines = runner.select_pipelines_intelligently(all_pipelines, sampling)
                strategy_info = runner.SAMPLING_STRATEGIES[sampling]
                click.echo(f"🧠 Sampling strategy: {strategy_info.name}")
                click.echo(f"📋 {strategy_info.description}")
                analysis_type = f"sampling ({sampling})"
            elif max_pipelines > 0 and max_pipelines < len(all_pipelines):
                test_pipelines = all_pipelines[:max_pipelines]
                click.echo(f"⚠️ Limited testing: Using {max_pipelines} of {len(all_pipelines)} available pipelines")
                analysis_type = f"limited testing ({max_pipelines} pipelines)"
            else:
                test_pipelines = all_pipelines
                click.echo("🔬 Full comprehensive testing: Using all available pipelines")
                analysis_type = "full comprehensive"

            # Calculate job estimates
            synthetic_gifs = runner.generate_synthetic_gifs()
            total_jobs = len(synthetic_gifs) * len(test_pipelines) * len(runner.test_params)
            estimated_time = runner._estimate_execution_time(total_jobs)

            if estimate_time:
                click.echo(f"📊 Total jobs: {total_jobs:,}")
                click.echo(f"⏱️ Estimated time: {estimated_time}")
                click.echo("✅ Time estimation complete. Remove --estimate-time to run analysis.")
                return

            # Run traditional analysis
            elimination_result = runner.run_analysis(
                test_pipelines=test_pipelines,
                quality_threshold=threshold,
                use_targeted_gifs=False,
            )

        # Display results header
        click.echo("🧪 GifLab Compression Analysis & Optimization")
        click.echo(f"📁 Input directory: {validated_raw_dir}")
        click.echo(f"📁 Output directory: {output_dir}")
        click.echo(f"👥 Workers: {validated_workers}")
        click.echo(f"🎯 Quality threshold: {threshold}")
        click.echo(f"🔄 Resume enabled: {resume}")
        click.echo(f"🧠 Analysis approach: {analysis_type}")

        # Display GPU status
        if use_gpu:
            click.echo("🚀 GPU acceleration: Enabled")
        else:
            click.echo("🖥️ GPU acceleration: Disabled")

        # Display results summary
        click.echo("\n📊 Analysis Results Summary:")
        click.echo(f"   📉 Eliminated pipelines: {len(elimination_result.eliminated_pipelines)}")
        click.echo(f"   ✅ Retained pipelines: {len(elimination_result.retained_pipelines)}")
        
        total_pipelines = len(elimination_result.eliminated_pipelines) + len(elimination_result.retained_pipelines)
        if total_pipelines > 0:
            elimination_rate = (len(elimination_result.eliminated_pipelines) / total_pipelines * 100)
            click.echo(f"   📈 Elimination rate: {elimination_rate:.1f}%")

        # Show top performers
        if elimination_result.retained_pipelines:
            click.echo("\n🏆 Top performing pipelines:")
            for i, pipeline in enumerate(list(elimination_result.retained_pipelines)[:5], 1):
                click.echo(f"   {i}. {pipeline}")

        click.echo("\n✅ Compression analysis complete!")
        click.echo(f"📁 Results saved to: {output_dir}")
        click.echo(f"💡 Use 'giflab select-pipelines {output_dir}/latest/results.csv --top 3' to get production configs")

    except KeyboardInterrupt:
        handle_keyboard_interrupt("Compression analysis")
    except Exception as e:
        handle_generic_error("Compression analysis", e)