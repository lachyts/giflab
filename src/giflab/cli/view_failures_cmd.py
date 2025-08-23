"""View failures command for analyzing failed pipelines."""

import json
from collections import Counter
from pathlib import Path

import click

from .utils import handle_generic_error


@click.command("view-failures")
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--error-type",
    type=click.Choice(
        [
            "all",
            "gifski",
            "ffmpeg",
            "imagemagick",
            "gifsicle",
            "animately",
            "command",
            "timeout",
            "other",
        ]
    ),
    default="all",
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
def view_failures(results_dir: Path, error_type: str, limit: int, detailed: bool) -> None:
    """
    View detailed information about failed pipelines from elimination testing.

    This command analyzes the failed_pipelines.json file from elimination runs
    and provides a human-readable summary of what went wrong.

    Examples:

        # View top 10 failures from latest run
        giflab view-failures results/runs/latest/

        # View all gifski failures with details
        giflab view-failures results/runs/latest/ --error-type gifski --limit 0 --detailed

        # Quick overview of command execution failures
        giflab view-failures results/runs/latest/ --error-type command --limit 5
    """
    try:
        failed_pipelines_file = results_dir / "failed_pipelines.json"
        if not failed_pipelines_file.exists():
            click.echo(f"❌ No failed pipelines file found at: {failed_pipelines_file}")
            click.echo(
                "   Make sure you're pointing to a directory with elimination results."
            )
            return

        try:
            with open(failed_pipelines_file) as f:
                failed_pipelines = json.load(f)
        except json.JSONDecodeError as e:
            click.echo(f"❌ Error reading failed pipelines file: {e}")
            return

        if not failed_pipelines:
            click.echo("✅ No failed pipelines found!")
            return

        # Filter by error type if specified
        if error_type != "all":
            filtered_failures = []
            error_keywords = {
                "gifski": "gifski",
                "ffmpeg": "ffmpeg",
                "imagemagick": "imagemagick",
                "gifsicle": "gifsicle",
                "animately": "animately",
                "command": "command failed",
                "timeout": "timeout",
                "other": None,  # Will be handled separately
            }

            keyword = error_keywords.get(error_type)
            for failure in failed_pipelines:
                error_msg = failure.get("error_message", "").lower()
                if error_type == "other":
                    # Show failures that don't match any specific tool
                    if not any(
                        tool in error_msg
                        for tool in [
                            "gifski",
                            "ffmpeg",
                            "imagemagick",
                            "gifsicle",
                            "animately",
                            "command failed",
                            "timeout",
                        ]
                    ):
                        filtered_failures.append(failure)
                elif keyword and keyword in error_msg:
                    filtered_failures.append(failure)

            failed_pipelines = filtered_failures

        # Apply limit
        if limit > 0:
            failed_pipelines = failed_pipelines[:limit]

        # Show summary statistics
        all_failures_file = results_dir / "failed_pipelines.json"
        with open(all_failures_file) as f:
            all_failures = json.load(f)

        click.echo(f"📊 Failure Analysis for {results_dir}")
        click.echo(f"   Total failures: {len(all_failures)}")
        if error_type != "all":
            click.echo(f"   Showing {error_type} failures: {len(failed_pipelines)}")

        # Error type breakdown
        error_types: Counter[str] = Counter()
        for failure in all_failures:
            error_msg = failure.get("error_message", "").lower()
            if "gifski" in error_msg:
                error_types["gifski"] += 1
            elif "ffmpeg" in error_msg:
                error_types["ffmpeg"] += 1
            elif "imagemagick" in error_msg:
                error_types["imagemagick"] += 1
            elif "gifsicle" in error_msg:
                error_types["gifsicle"] += 1
            elif "animately" in error_msg:
                error_types["animately"] += 1
            elif "command failed" in error_msg:
                error_types["command"] += 1
            elif "timeout" in error_msg:
                error_types["timeout"] += 1
            else:
                error_types["other"] += 1

        click.echo("   Error type breakdown:")
        for error_type_name, count in error_types.most_common():
            click.echo(f"     {error_type_name}: {count}")

        if not failed_pipelines:
            click.echo(f"\n❌ No failures found matching filter: {error_type}")
            return

        # Show individual failures
        click.echo("\n🔍 Failed Pipeline Details:")
        for i, failure in enumerate(failed_pipelines, 1):
            pipeline_id = failure.get("pipeline_id", "unknown")
            gif_name = failure.get("gif_name", "unknown")
            error_msg = failure.get("error_message", "No error message")
            tools = failure.get("tools_used", [])

            click.echo(f"\n{i:2d}. {pipeline_id}")
            click.echo(
                f"    GIF: {gif_name} ({failure.get('content_type', 'unknown')})"
            )
            click.echo(f"    Tools: {', '.join(tools) if tools else 'unknown'}")
            click.echo(f"    Error: {error_msg}")

            if detailed:
                traceback_info = failure.get("error_traceback", "")
                if traceback_info:
                    click.echo(f"    Traceback: {traceback_info}")

                test_params = failure.get("test_parameters", {})
                if test_params:
                    click.echo(
                        f"    Parameters: colors={test_params.get('colors')}, lossy={test_params.get('lossy')}, frame_ratio={test_params.get('frame_ratio')}"
                    )

                timestamp = failure.get("error_timestamp", failure.get("timestamp"))
                if timestamp:
                    click.echo(f"    Time: {timestamp}")

        click.echo("\n💡 To see more details, use --detailed flag")
        click.echo("💡 To filter by error type, use --error-type <type>")

    except Exception as e:
        handle_generic_error("View failures", e)
