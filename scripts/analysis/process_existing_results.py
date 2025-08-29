#!/usr/bin/env python3
"""
Process existing experimental results to add enhanced composite quality and efficiency metrics.

This script processes the existing CSV results file and adds the new enhanced metrics
without needing to re-run the full experiment.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the src directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from giflab.config import DEFAULT_METRICS_CONFIG
from giflab.enhanced_metrics import (
    calculate_efficiency_metric,
    calculate_composite_quality,
    process_metrics_with_enhanced_quality,
)


def process_existing_results(csv_path: Path, output_path: Path = None):
    """Process existing results CSV to add enhanced metrics."""

    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return False

    if output_path is None:
        output_path = csv_path.parent / f"enhanced_{csv_path.name}"

    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Found {len(df)} rows to process")
    print(f"Columns: {list(df.columns)}")

    # Process each row to add enhanced metrics
    enhanced_rows = []

    for idx, row in df.iterrows():
        # Convert row to dictionary for processing
        result = row.to_dict()

        # Skip failed results
        if not result.get("success", True):
            # Add null values for new metrics
            result["composite_quality"] = None
            result["efficiency"] = None
            enhanced_rows.append(result)
            continue

        # Process with enhanced metrics system
        try:
            enhanced_result = process_metrics_with_enhanced_quality(
                result, DEFAULT_METRICS_CONFIG
            )
            enhanced_rows.append(enhanced_result)
        except Exception as e:
            print(f"Warning: Failed to process row {idx}: {e}")
            # Add null values for new metrics if processing fails
            result["composite_quality"] = None
            result["efficiency"] = None
            enhanced_rows.append(result)

    # Create enhanced DataFrame
    enhanced_df = pd.DataFrame(enhanced_rows)

    # Save to output file
    enhanced_df.to_csv(output_path, index=False)

    print(f"Enhanced results saved to: {output_path}")
    print("Added columns: composite_quality, efficiency")

    # Show some statistics
    if "composite_quality" in enhanced_df.columns:
        print("\nEnhanced composite quality statistics:")
        print(enhanced_df["composite_quality"].describe())

    if "efficiency" in enhanced_df.columns:
        print("\nEfficiency statistics:")
        print(enhanced_df["efficiency"].describe())

    return True


def main():
    """Main function to process existing results."""

    # Find the most recent results file
    results_dir = Path("test-workspace/frame-comparison-with-gifs")

    # Look for the most recent run directory
    run_dirs = list(results_dir.glob("run_*"))
    if not run_dirs:
        print("No run directories found in test-workspace/frame-comparison-with-gifs/")
        return False

    # Get the most recent run
    latest_run = max(run_dirs, key=lambda p: p.name)
    csv_path = latest_run / "streaming_results.csv"

    print(f"Processing results from: {latest_run}")

    if not csv_path.exists():
        print(f"No streaming_results.csv found in {latest_run}")
        return False

    # Process the results
    output_path = latest_run / "enhanced_streaming_results.csv"
    return process_existing_results(csv_path, output_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
