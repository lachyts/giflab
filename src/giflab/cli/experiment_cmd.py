"""Experimental pipeline testing command."""

from pathlib import Path

import click

from .utils import (
    check_gpu_availability,
    handle_generic_error,
    handle_keyboard_interrupt,
)


def create_custom_preset_from_cli(variable_slots, lock_slots, slot_params_list):
    """Create a custom experiment preset from CLI arguments.

    Args:
        variable_slots: Tuple of variable slot specifications
        lock_slots: Tuple of lock slot specifications
        slot_params_list: Tuple of parameter specifications

    Returns:
        ExperimentPreset object created from CLI arguments

    Raises:
        ValueError: If slot specifications are invalid
    """
    from ast import literal_eval

    # Import preset classes
    from giflab.experimental.targeted_presets import ExperimentPreset, SlotConfiguration

    # Initialize slot configurations with defaults (all locked to common implementations)
    slots = {
        "frame": SlotConfiguration(
            type="locked", implementation="animately-frame", parameters={}
        ),
        "color": SlotConfiguration(
            type="locked", implementation="ffmpeg-color", parameters={"colors": 32}
        ),
        "lossy": SlotConfiguration(
            type="locked",
            implementation="animately-advanced-lossy",
            parameters={"level": 40},
        ),
    }

    # Parse variable slot specifications
    for slot_spec in variable_slots:
        if "=" not in slot_spec:
            raise ValueError(
                f"Invalid variable slot format: {slot_spec}. Expected 'slot=scope'"
            )

        slot_name, scope_str = slot_spec.split("=", 1)
        slot_name = slot_name.strip()

        if slot_name not in ["frame", "color", "lossy"]:
            raise ValueError(
                f"Invalid slot name: {slot_name}. Must be 'frame', 'color', or 'lossy'"
            )

        # Parse scope
        if scope_str.strip() == "*":
            scope = ["*"]
        else:
            scope = [tool.strip() for tool in scope_str.split(",")]

        slots[slot_name] = SlotConfiguration(
            type="variable", scope=scope, parameters={}
        )

    # Parse lock slot specifications
    for slot_spec in lock_slots:
        if "=" not in slot_spec:
            raise ValueError(
                f"Invalid lock slot format: {slot_spec}. Expected 'slot=implementation'"
            )

        slot_name, implementation = slot_spec.split("=", 1)
        slot_name = slot_name.strip()
        implementation = implementation.strip()

        if slot_name not in ["frame", "color", "lossy"]:
            raise ValueError(
                f"Invalid slot name: {slot_name}. Must be 'frame', 'color', or 'lossy'"
            )

        slots[slot_name] = SlotConfiguration(
            type="locked", implementation=implementation, parameters={}
        )

    # Parse slot parameters
    for param_spec in slot_params_list:
        if "=" not in param_spec:
            raise ValueError(
                f"Invalid slot parameter format: {param_spec}. Expected 'slot=param:value'"
            )

        slot_name, param_str = param_spec.split("=", 1)
        slot_name = slot_name.strip()

        if slot_name not in ["frame", "color", "lossy"]:
            raise ValueError(
                f"Invalid slot name: {slot_name}. Must be 'frame', 'color', or 'lossy'"
            )

        if ":" not in param_str:
            raise ValueError(
                f"Invalid parameter format: {param_str}. Expected 'param:value'"
            )

        param_name, value_str = param_str.split(":", 1)
        param_name = param_name.strip()
        value_str = value_str.strip()

        # Parse parameter value
        try:
            if value_str.startswith("[") and value_str.endswith("]"):
                # List parameter
                param_value = literal_eval(value_str)
            elif value_str.startswith("{") and value_str.endswith("}"):
                # Dict parameter
                param_value = literal_eval(value_str)
            elif value_str.lower() in ["true", "false"]:
                # Boolean parameter
                param_value = value_str.lower() == "true"
            elif "." in value_str:
                # Float parameter
                param_value = float(value_str)
            else:
                # Try int, fallback to string
                try:
                    param_value = int(value_str)
                except ValueError:
                    param_value = value_str
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid parameter value '{value_str}': {e}")

        # Add parameter to slot configuration
        slots[slot_name].parameters[param_name] = param_value

    # Validate that at least one slot is variable
    variable_count = sum(1 for slot in slots.values() if slot.type == "variable")
    if variable_count == 0:
        raise ValueError(
            "At least one slot must be variable. Use --variable-slot to specify variable slots."
        )

    # Create preset name from slot configurations
    variable_slots_names = [
        name for name, slot in slots.items() if slot.type == "variable"
    ]
    preset_name = f"Custom {'+'.join(variable_slots_names).title()} Study"
    preset_description = (
        f"Custom experiment varying {', '.join(variable_slots_names)} slots"
    )

    # Create and return the preset
    return ExperimentPreset(
        name=preset_name,
        description=preset_description,
        frame_slot=slots["frame"],
        color_slot=slots["color"],
        lossy_slot=slots["lossy"],
        tags=["custom", "cli-generated"],
        author="CLI User",
    )


@click.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("results/experiments"),
    help="Base directory for timestamped experiment results (default: results/experiments)",
)
@click.option(
    "--sampling",
    type=click.Choice(
        ["full", "representative", "factorial", "progressive", "targeted", "quick"]
    ),
    default="representative",
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
    "--use-cache",
    is_flag=True,
    help="Enable cache for pipeline test results (faster but may use stale results during development)",
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
    help="Use targeted experiment preset (e.g., 'frame-focus', 'color-optimization'). Use 'list' to see available presets.",
)
@click.option(
    "--list-presets",
    is_flag=True,
    help="List all available experiment presets and exit",
)
@click.option(
    "--variable-slot",
    multiple=True,
    help="Define variable slot: 'frame=*' or 'color=ffmpeg-color,gifsicle-color'. Can be used multiple times.",
)
@click.option(
    "--lock-slot",
    multiple=True,
    help="Lock slot to specific implementation: 'frame=animately-frame' or 'lossy=animately-advanced-lossy'. Can be used multiple times.",
)
@click.option(
    "--slot-params",
    multiple=True,
    help="Specify slot parameters: 'frame=ratios:[1.0,0.8,0.5]' or 'color=colors:[64,32]'. Can be used multiple times.",
)
def experiment(
    output_dir: Path,
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
    variable_slot: tuple,
    lock_slot: tuple,
    slot_params: tuple,
):
    """Run comprehensive experimental pipeline testing with intelligent sampling.

    This command tests pipeline combinations on synthetic GIFs with diverse
    characteristics and eliminates underperforming pipelines based on quality
    metrics like SSIM, compression ratio, and processing speed.

    Results are saved in timestamped directories to preserve historical data.
    Smart caching avoids re-running identical pipeline tests.
    """
    try:
        # Import from experimental module
        from giflab.dynamic_pipeline import generate_all_pipelines
        from giflab.experimental import ExperimentalRunner
        from giflab.experimental.sampling import SAMPLING_STRATEGIES

        # Create pipeline runner with cache settings (disabled by default)
        # use_cache flag is now directly used (defaults to False)
        # Determine preset name for directory naming
        preset_name_for_dir = preset if preset else "custom-experiment"
        runner = ExperimentalRunner(output_dir, use_gpu=use_gpu, use_cache=use_cache, preset_name=preset_name_for_dir)

        # Handle preset listing request
        if list_presets:
            try:
                presets = runner.list_available_presets()
                click.echo("🎯 Available Experiment Presets:")
                click.echo()
                for preset_id, description in presets.items():
                    click.echo(f"  {preset_id}")
                    click.echo(f"    {description}")
                    click.echo()
                return
            except Exception as e:
                click.echo(f"❌ Error listing presets: {e}")
                return

        # Clear cache if requested
        if clear_cache and runner.cache:
            click.echo("🗑️ Clearing pipeline results cache...")
            runner.cache.clear_cache()

        # Determine pipeline generation approach
        has_custom_slots = bool(variable_slot or lock_slot or slot_params)

        if preset and has_custom_slots:
            click.echo(
                "❌ Error: Cannot use both --preset and custom slot options (--variable-slot, --lock-slot, --slot-params)"
            )
            click.echo(
                "💡 Use either --preset for predefined configurations or slot options for custom experiments"
            )
            return
        elif preset:
            # Predefined preset approach
            try:
                test_pipelines = runner.generate_targeted_pipelines(preset)
                click.echo(f"🎯 Using targeted preset: {preset}")
                click.echo(f"🔬 Generated {len(test_pipelines)} targeted pipelines")
                testing_approach = "targeted_preset"
            except Exception as e:
                click.echo(f"❌ Error with preset '{preset}': {e}")
                click.echo("💡 Use --list-presets to see available presets")
                return
        elif has_custom_slots:
            # Custom preset approach
            try:
                custom_preset = create_custom_preset_from_cli(
                    variable_slot, lock_slot, slot_params
                )
                click.echo(f"🔧 Creating custom preset: {custom_preset.name}")
                click.echo(f"📋 {custom_preset.description}")

                # Generate pipelines using custom preset
                from giflab.experimental.targeted_generator import (
                    TargetedPipelineGenerator,
                )

                generator = TargetedPipelineGenerator()
                test_pipelines = generator.generate_targeted_pipelines(custom_preset)

                click.echo(
                    f"🔬 Generated {len(test_pipelines)} custom targeted pipelines"
                )
                testing_approach = "custom_targeted"
            except Exception as e:
                click.echo(f"❌ Error creating custom preset: {e}")
                click.echo("💡 Check your slot configuration syntax")
                return
        else:
            # Traditional approach with sampling
            all_pipelines = generate_all_pipelines()

            if sampling != "full":
                test_pipelines = runner.select_pipelines_intelligently(
                    all_pipelines, sampling
                )
                strategy_info = runner.SAMPLING_STRATEGIES[sampling]
                click.echo(f"🧠 Sampling strategy: {strategy_info.name}")
                click.echo(f"📋 {strategy_info.description}")
                testing_approach = f"sampling_{sampling}"
            elif max_pipelines > 0 and max_pipelines < len(all_pipelines):
                test_pipelines = all_pipelines[:max_pipelines]
                click.echo(
                    f"⚠️  Limited testing: Using {max_pipelines} of {len(all_pipelines)} available pipelines"
                )
                testing_approach = "limited_testing"
            else:
                test_pipelines = all_pipelines
                click.echo(
                    "🔬 Full comprehensive testing: Using all available pipelines"
                )
                testing_approach = "full_comprehensive"

        # Calculate total job estimates
        if (
            sampling == "targeted"
            or (preset and "targeted" in preset.lower())
            or has_custom_slots
        ):
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
        if preset:
            click.echo(f"🎯 Testing approach: Targeted preset ({preset})")
        elif has_custom_slots:
            click.echo("🔧 Testing approach: Custom targeted configuration")
        else:
            click.echo(
                f"🧠 Testing approach: {testing_approach.replace('_', ' ').title()}"
            )

        # Display GPU status
        click.echo(check_gpu_availability(use_gpu))

        if estimate_time:
            click.echo(
                "✅ Time estimation complete. Use without --estimate-time to run actual analysis."
            )
            return

        click.echo("\n🚀 Running comprehensive experimental pipeline testing...")

        # Run the experimental analysis
        use_targeted_gifs = (
            sampling == "targeted"
            or (preset and "targeted" in preset.lower())
            or has_custom_slots
        )

        if preset:
            # Use targeted experiment method for presets (includes parameter locking)
            elimination_result = runner.run_targeted_experiment(
                preset_id=preset,
                quality_threshold=threshold,
                use_targeted_gifs=use_targeted_gifs,
            )
        else:
            # Use traditional experimental analysis for non-preset experiments
            elimination_result = runner.run_experimental_analysis(
                test_pipelines=test_pipelines,
                quality_threshold=threshold,
                use_targeted_gifs=use_targeted_gifs,
            )

        # Display results
        click.echo("\n📊 Experimental Results Summary:")
        click.echo(
            f"   📉 Eliminated pipelines: {len(elimination_result.eliminated_pipelines)}"
        )
        click.echo(
            f"   ✅ Retained pipelines: {len(elimination_result.retained_pipelines)}"
        )
        total_pipelines = len(elimination_result.eliminated_pipelines) + len(
            elimination_result.retained_pipelines
        )
        if total_pipelines > 0:
            elimination_rate = (
                len(elimination_result.eliminated_pipelines) / total_pipelines * 100
            )
            click.echo(f"   📈 Elimination rate: {elimination_rate:.1f}%")

        # Show top performers
        if elimination_result.retained_pipelines:
            click.echo("\n🏆 Top performing pipelines:")
            for i, pipeline in enumerate(
                list(elimination_result.retained_pipelines)[:5], 1
            ):
                click.echo(f"   {i}. {pipeline}")

        click.echo("\n✅ Experimental analysis complete!")
        click.echo(f"📁 Results saved to: {output_dir}")
        click.echo(
            f"💡 Use 'giflab select-pipelines {output_dir}/latest/results.csv --top 3' to get production configs"
        )

    except KeyboardInterrupt:
        handle_keyboard_interrupt("Experimental pipeline testing")
    except Exception as e:
        handle_generic_error("Experimental pipeline testing", e)
