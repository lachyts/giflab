"""Select pipelines command for picking best pipelines from experiment results."""

from pathlib import Path

import click
import pandas as pd

from ..utils_pipeline_yaml import write_pipelines_yaml
from .utils import handle_generic_error


@click.command("select-pipelines")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--metric", default="ssim", help="Quality metric to optimise (default: ssim)"
)
@click.option("--top", default=1, help="Top-N pipelines to pick (per variable)")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("winners.yaml"),
)
def select_pipelines(csv_file: Path, metric: str, top: int, output: Path) -> None:
    """Pick the best pipelines from an experiment CSV and write a YAML list."""
    try:
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

    except Exception as e:
        handle_generic_error("Select pipelines", e)
