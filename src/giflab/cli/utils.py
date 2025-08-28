"""Shared utilities for CLI commands."""

import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import click

from ..input_validation import ValidationError, validate_raw_dir, validate_worker_count


def handle_generic_error(command_name: str, error: Exception) -> None:
    """Handle generic command errors with consistent formatting."""
    click.echo(f"❌ {command_name} failed: {error}", err=True)
    sys.exit(1)


def handle_keyboard_interrupt(command_name: str) -> None:
    """Handle keyboard interrupt with consistent formatting."""
    click.echo(f"\n⏹️  {command_name} interrupted by user", err=True)
    sys.exit(1)


def validate_and_get_raw_dir(raw_dir: Path, require_gifs: bool = True) -> Path:
    """Validate RAW_DIR input and return validated path."""
    try:
        return validate_raw_dir(raw_dir, require_gifs=require_gifs)
    except ValidationError as e:
        click.echo(f"❌ Invalid RAW_DIR: {e}", err=True)
        click.echo("💡 Please provide a valid directory containing GIF files", err=True)
        sys.exit(1)


def validate_and_get_worker_count(workers: int) -> int:
    """Validate worker count and return validated count."""
    try:
        return validate_worker_count(workers)
    except ValidationError as e:
        click.echo(f"❌ Invalid worker count: {e}", err=True)
        sys.exit(1)


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    return multiprocessing.cpu_count()


def display_worker_info(validated_workers: int) -> None:
    """Display worker count information."""
    actual_workers = validated_workers if validated_workers > 0 else get_cpu_count()
    click.echo(f"👥 Workers: {actual_workers}")


def generate_timestamped_csv_path(base_dir: Path, prefix: str = "results") -> Path:
    """Generate a timestamped CSV file path."""
    timestamp = datetime.now().strftime("%Y%m%d")
    return base_dir / f"{prefix}_{timestamp}.csv"


def ensure_csv_parent_exists(csv_path: Path) -> None:
    """Ensure the parent directory of a CSV file exists."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)


def check_gpu_availability(use_gpu: bool) -> str:
    """Check GPU availability and return status message."""
    if not use_gpu:
        return "📊 GPU acceleration: Disabled (using CPU processing)"

    try:
        import cv2

        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            return f"🚀 GPU acceleration: Enabled ({cuda_devices} CUDA device(s) available)"
        else:
            return "🔄 GPU acceleration: Requested but no CUDA devices found - will use CPU"
    except ImportError:
        return "🔄 GPU acceleration: Requested but OpenCV CUDA not available - will use CPU"
    except Exception:
        return "🔄 GPU acceleration: Requested but initialization failed - will use CPU"


def estimate_pipeline_time(total_jobs: int, seconds_per_job: int = 2) -> str:
    """Estimate execution time for pipeline jobs."""
    estimated_time = total_jobs * seconds_per_job
    estimated_hours = estimated_time / 3600
    return f"~{estimated_time}s (~{estimated_hours:.1f}h)"


def display_common_header(title: str) -> None:
    """Display a common header for CLI commands."""
    click.echo(f"🎞️  {title}")


def display_path_info(label: str, path: Path, emoji: str = "📁") -> None:
    """Display path information with consistent formatting."""
    click.echo(f"{emoji} {label}: {path}")


def display_results_summary(result: dict) -> None:
    """Display a common results summary format."""
    status = result["status"]

    click.echo("\n📊 Results:")
    click.echo(f"   • Status: {status}")

    for key in ["processed", "failed", "skipped", "total_jobs"]:
        if key in result:
            click.echo(f"   • {key.replace('_', ' ').title()}: {result[key]}")

    if "csv_path" in result:
        click.echo(f"   • Results saved to: {result['csv_path']}")

    if "output_path" in result:
        click.echo(f"   • Results saved to: {result['output_path']}")
