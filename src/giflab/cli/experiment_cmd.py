"""Experimental pipeline testing command."""

from pathlib import Path

import click

from .utils import (
    check_gpu_availability,
    handle_generic_error,
    handle_keyboard_interrupt,
)


@click.command()
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
    try:
        # Import from experimental.py file directly (not the directory)
        import sys
        from pathlib import Path
        _parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(_parent_dir))
        try:
            import experimental
            ExperimentalRunner = experimental.ExperimentalRunner
        finally:
            sys.path.remove(str(_parent_dir))
        
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
        click.echo(check_gpu_availability(use_gpu))
        
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

    except KeyboardInterrupt:
        handle_keyboard_interrupt("Experimental pipeline testing")
    except Exception as e:
        handle_generic_error("Experimental pipeline testing", e)