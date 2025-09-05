"""Validation CLI commands for analyzing and filtering experiment results."""

import json
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd

from ..optimization_validation import ValidationChecker
from .utils import handle_generic_error


@click.group("validate")
def validate() -> None:
    """🔍 Validation tools for analyzing compression pipeline results."""
    pass


@validate.command("results")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV file (default: adds _validated suffix)",
)
@click.option(
    "--content-type",
    default="unknown",
    help="Content type for validation thresholds (default: unknown)",
)
def validate_results(csv_file: Path, output: Path | None, content_type: str) -> None:
    """Re-run validation on existing experiment results and save updated CSV."""
    try:
        click.echo("🔍 Loading experiment results…")
        df = pd.read_csv(csv_file)

        if df.empty:
            click.echo("❌ CSV file is empty", err=True)
            raise SystemExit(1)

        click.echo(f"📊 Found {len(df)} experiment results")

        # Initialize validation checker
        validator = ValidationChecker()
        click.echo("✅ ValidationChecker initialized")

        # Process validation for each result
        validation_updates = []

        for idx, row in df.iterrows():
            if (
                pd.isna(row.get("validation_status"))
                or row.get("validation_status") == "EXECUTION_FAILED"
            ):
                # Re-run validation for rows missing validation data
                try:
                    # Reconstruct validation inputs from CSV data
                    original_info = {
                        "frame_count": int(row.get("orig_frames", 0))
                        if pd.notna(row.get("orig_frames"))
                        else 0,
                        "fps": float(row.get("orig_fps", 0))
                        if pd.notna(row.get("orig_fps"))
                        else 0.0,
                        "file_size_kb": float(row.get("original_size_kb", 0))
                        if pd.notna(row.get("original_size_kb"))
                        else 0.0,
                        "width": int(row.get("orig_width", 0))
                        if pd.notna(row.get("orig_width"))
                        else 0,
                        "height": int(row.get("orig_height", 0))
                        if pd.notna(row.get("orig_height"))
                        else 0,
                    }

                    compressed_info = {
                        "frame_count": int(row.get("compressed_frame_count", 0))
                        if pd.notna(row.get("compressed_frame_count"))
                        else 0,
                        "fps": float(row.get("orig_fps", 0))
                        if pd.notna(row.get("orig_fps"))
                        else 0.0,  # Use original FPS as fallback
                        "file_size_kb": float(row.get("file_size_kb", 0))
                        if pd.notna(row.get("file_size_kb"))
                        else 0.0,
                        "quality_metrics": {
                            "ssim": float(row.get("ssim_mean", 0))
                            if pd.notna(row.get("ssim_mean"))
                            else 0.0,
                            "composite_quality": float(row.get("composite_quality", 0))
                            if pd.notna(row.get("composite_quality"))
                            else 0.0,
                            "efficiency": float(row.get("efficiency", 0))
                            if pd.notna(row.get("efficiency"))
                            else 0.0,
                            "temporal_consistency": float(
                                row.get("temporal_consistency", 0)
                            )
                            if pd.notna(row.get("temporal_consistency"))
                            else 0.0,
                        },
                    }

                    # Create minimal metadata object for validation
                    from ..meta import GifMetadata

                    metadata = GifMetadata(
                        gif_sha="reconstructed",
                        orig_filename=str(
                            row.get("orig_filename", row.get("gif_name", "unknown"))
                        ),
                        orig_kilobytes=original_info["file_size_kb"],
                        orig_width=int(original_info["width"]),
                        orig_height=int(original_info["height"]),
                        orig_frames=int(original_info["frame_count"]),
                        orig_fps=original_info["fps"],
                        orig_n_colors=int(row.get("orig_n_colors", 256))
                        if pd.notna(row.get("orig_n_colors"))
                        else 256,
                        entropy=float(row.get("entropy", 0))
                        if pd.notna(row.get("entropy"))
                        else None,
                        source_platform=str(row.get("source_platform", "unknown")),
                        source_metadata=None,
                    )

                    # Run validation
                    quality_metrics = compressed_info["quality_metrics"]
                    if not isinstance(quality_metrics, dict):
                        raise ValueError(f"Expected dict for quality_metrics, got {type(quality_metrics)}")
                    validation_result = validator.validate_compression_result(
                        original_metadata=metadata,
                        compression_metrics=quality_metrics,
                        pipeline_id=str(row["pipeline_id"]),
                        gif_name=str(row["gif_name"]),
                        content_type=content_type,
                    )

                    # Update validation fields
                    validation_updates.append(
                        {
                            "index": idx,
                            "validation_status": validation_result.status.value,
                            "validation_passed": validation_result.status.value
                            == "PASS",
                            "validation_issues_count": len(validation_result.issues)
                            + len(validation_result.warnings),
                            "validation_messages": "; ".join(
                                [
                                    issue.message
                                    for issue in (
                                        validation_result.issues
                                        + validation_result.warnings
                                    )
                                ]
                            )
                            if (validation_result.issues or validation_result.warnings)
                            else None,
                        }
                    )

                except Exception as e:
                    click.echo(f"⚠️ Validation failed for row {idx}: {e}")

        # Apply validation updates to dataframe
        for update in validation_updates:
            idx = update["index"]
            for key, value in update.items():
                if key != "index":
                    df.at[idx, key] = value

        # Save updated results
        if not output:
            output = csv_file.parent / f"{csv_file.stem}_validated.csv"

        df.to_csv(output, index=False)
        click.echo(f"✅ Updated validation results saved to {output}")
        click.echo(
            f"📊 Processed {len(validation_updates)} rows with validation updates"
        )

    except Exception as e:
        handle_generic_error("Validate results", e)


@validate.command("filter")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--status",
    type=click.Choice(["PASS", "WARNING", "ERROR", "ARTIFACT"], case_sensitive=False),
    help="Filter by validation status",
)
@click.option(
    "--category",
    type=click.Choice(
        ["fps", "quality", "artifacts", "efficiency"], case_sensitive=False
    ),
    help="Filter by validation category",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV file (default: adds filter suffix)",
)
def filter_results(
    csv_file: Path, status: str | None, category: str | None, output: Path | None
) -> None:
    """Filter experiment results by validation status and categories."""
    try:
        click.echo("📊 Loading experiment results…")
        df = pd.read_csv(csv_file)

        if df.empty:
            click.echo("❌ CSV file is empty", err=True)
            raise SystemExit(1)

        original_count = len(df)
        filtered_df = df.copy()

        # Apply status filter
        if status:
            status_upper = status.upper()
            filtered_df = filtered_df[filtered_df["validation_status"] == status_upper]
            click.echo(
                f"🔍 Filtered by status '{status_upper}': {len(filtered_df)}/{original_count} results"
            )

        # Apply category filter (search in validation messages)
        if category:
            category_keywords = {
                "fps": ["fps", "frame rate", "timing"],
                "quality": ["quality", "ssim", "psnr", "composite"],
                "artifacts": ["artifact", "disposal", "corruption"],
                "efficiency": ["efficiency", "compression ratio"],
            }

            keywords = category_keywords.get(category.lower(), [category.lower()])

            def has_category_issue(messages: Any) -> bool:
                if pd.isna(messages):
                    return False
                messages_lower = str(messages).lower()
                return any(keyword in messages_lower for keyword in keywords)

            filtered_df = filtered_df[
                filtered_df["validation_messages"].apply(has_category_issue)
            ]
            click.echo(
                f"🔍 Filtered by category '{category}': {len(filtered_df)}/{original_count} results"
            )

        # Save filtered results
        if not output:
            suffix_parts = []
            if status:
                suffix_parts.append(f"status_{status.lower()}")
            if category:
                suffix_parts.append(f"cat_{category.lower()}")
            suffix = "_".join(suffix_parts) if suffix_parts else "filtered"
            output = csv_file.parent / f"{csv_file.stem}_{suffix}.csv"

        filtered_df.to_csv(output, index=False)

        # Display summary
        click.echo(f"✅ Filtered results saved to {output}")
        click.echo(
            f"📊 Results: {len(filtered_df)}/{original_count} ({len(filtered_df)/original_count*100:.1f}%)"
        )

        if len(filtered_df) > 0:
            # Show validation status distribution
            validation_counts = filtered_df["validation_status"].value_counts().to_dict()
            click.echo("📈 Validation status distribution:")
            for status_val, count in validation_counts.items():
                status_str = str(status_val)  # Ensure string type for dict lookup
                status_emoji = {
                    "PASS": "✅",
                    "WARNING": "⚠️",
                    "ERROR": "❌",
                    "ARTIFACT": "🔍",
                }.get(status_str, "❓")
                click.echo(f"   {status_emoji} {status_val}: {count}")

    except Exception as e:
        handle_generic_error("Filter validation results", e)


@validate.command("report")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format (default: text)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file (default: prints to stdout)",
)
def generate_report(csv_file: Path, format: str, output: Path | None) -> None:
    """Generate comprehensive validation analysis reports."""
    try:
        click.echo("📊 Loading experiment results…")
        df = pd.read_csv(csv_file)

        if df.empty:
            click.echo("❌ CSV file is empty", err=True)
            raise SystemExit(1)

        # Calculate validation statistics
        total_results = len(df)
        validation_counts = df["validation_status"].value_counts()

        # Calculate pass rate
        pass_count = validation_counts.get("PASS", 0)
        pass_rate = (pass_count / total_results) * 100 if total_results > 0 else 0

        # Find most common validation issues
        all_messages = df["validation_messages"].dropna()
        issue_categories: dict[str, int] = {}
        for messages in all_messages:
            if messages:
                for message in str(messages).split("; "):
                    if ":" in message:
                        category = message.split(":")[0].strip()
                        issue_categories[category] = (
                            issue_categories.get(category, 0) + 1
                        )

        # Top validation issues
        top_issues = sorted(issue_categories.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        # Generate report
        report_data = {
            "summary": {
                "total_experiments": total_results,
                "validation_pass_rate": round(pass_rate, 1),
                "validation_status_distribution": dict(validation_counts),
            },
            "top_validation_issues": top_issues,
            "thresholds_analysis": {
                "most_failed_pipelines": df[df["validation_status"] != "PASS"][
                    "pipeline_id"
                ]
                .value_counts()
                .head(5)
                .to_dict()
                if len(df[df["validation_status"] != "PASS"]) > 0
                else {},
                "avg_issues_per_result": round(df["validation_issues_count"].mean(), 2)
                if "validation_issues_count" in df.columns
                else 0,
            },
        }

        # Format output
        if format.lower() == "json":
            output_content = json.dumps(report_data, indent=2)
        else:  # text format
            summary_data: dict[str, Any] = report_data['summary']  # type: ignore[assignment]
            output_content = f"""🔍 Validation Analysis Report
Generated from: {csv_file}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 SUMMARY
Total Experiments: {summary_data['total_experiments']}
Validation Pass Rate: {summary_data['validation_pass_rate']}%

📈 VALIDATION STATUS DISTRIBUTION
"""
            validation_status_dist: dict[str, Any] = summary_data["validation_status_distribution"]
            for status, count in validation_status_dist.items():
                status_str = str(status)  # Ensure string type
                status_emoji = {
                    "PASS": "✅",
                    "WARNING": "⚠️",
                    "ERROR": "❌",
                    "ARTIFACT": "🔍",
                }.get(status_str, "❓")
                percentage = (count / total_results) * 100
                output_content += (
                    f"{status_emoji} {status}: {count} ({percentage:.1f}%)\n"
                )

            output_content += "\n🚨 TOP VALIDATION ISSUES\n"
            for issue, count in top_issues:
                output_content += f"• {issue}: {count} occurrences\n"

            output_content += "\n⚙️ PIPELINE ANALYSIS\n"
            thresholds_analysis: dict[str, Any] = report_data["thresholds_analysis"]  # type: ignore[assignment]
            most_failed_pipelines: dict[str, Any] = thresholds_analysis["most_failed_pipelines"]
            if most_failed_pipelines:
                output_content += "Most Failed Pipelines:\n"
                for pipeline, count in list(most_failed_pipelines.items())[:3]:
                    output_content += f"• {pipeline}: {count} failures\n"

            output_content += f"Average Issues per Result: {thresholds_analysis['avg_issues_per_result']}\n"

        # Output results
        if output:
            output.write_text(output_content)
            click.echo(f"✅ Validation report saved to {output}")
        else:
            click.echo("\n" + output_content)

    except Exception as e:
        handle_generic_error("Generate validation report", e)


@validate.command("threshold")
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--suggest-params",
    is_flag=True,
    help="Suggest parameter adjustments based on validation failures",
)
def analyze_thresholds(csv_file: Path, suggest_params: bool) -> None:
    """Analyze validation threshold violations and suggest parameter adjustments."""
    try:
        click.echo("📊 Loading experiment results…")
        df = pd.read_csv(csv_file)

        if df.empty:
            click.echo("❌ CSV file is empty", err=True)
            raise SystemExit(1)

        # Analyze validation failures
        failed_df = df[df["validation_status"].isin(["ERROR", "WARNING", "ARTIFACT"])]

        click.echo(
            f"🔍 Analyzing {len(failed_df)}/{len(df)} results with validation issues"
        )

        if len(failed_df) == 0:
            click.echo("✅ No validation failures found!")
            return

        # Analyze threshold violations by category
        violation_analysis = {
            "fps_issues": 0,
            "quality_issues": 0,
            "efficiency_issues": 0,
            "artifact_issues": 0,
        }

        for _, row in failed_df.iterrows():
            messages = str(row.get("validation_messages", ""))
            if "fps" in messages.lower() or "frame rate" in messages.lower():
                violation_analysis["fps_issues"] += 1
            if "quality" in messages.lower() or "threshold" in messages.lower():
                violation_analysis["quality_issues"] += 1
            if "efficiency" in messages.lower():
                violation_analysis["efficiency_issues"] += 1
            if "artifact" in messages.lower() or "disposal" in messages.lower():
                violation_analysis["artifact_issues"] += 1

        # Display analysis results
        click.echo("\n🚨 VALIDATION ISSUE BREAKDOWN")
        for issue_type, count in violation_analysis.items():
            if count > 0:
                percentage = (count / len(failed_df)) * 100
                click.echo(
                    f"• {issue_type.replace('_', ' ').title()}: {count} ({percentage:.1f}% of failures)"
                )

        # Pipeline-specific analysis
        pipeline_failures = failed_df["pipeline_id"].value_counts()
        click.echo("\n⚙️ MOST PROBLEMATIC PIPELINES")
        for pipeline, count in pipeline_failures.head(5).items():
            percentage = (count / len(failed_df)) * 100
            click.echo(f"• {pipeline}: {count} failures ({percentage:.1f}%)")

        # Parameter suggestions
        if suggest_params:
            click.echo("\n💡 PARAMETER ADJUSTMENT SUGGESTIONS")

            if violation_analysis["quality_issues"] > len(failed_df) * 0.3:
                click.echo(
                    "• Consider reducing lossy compression levels (lower lossy values)"
                )
                click.echo("• Try pipelines with fewer compression steps")

            if violation_analysis["efficiency_issues"] > len(failed_df) * 0.3:
                click.echo(
                    "• Consider more aggressive color reduction (lower color counts)"
                )
                click.echo(
                    "• Try frame reduction pipelines to improve compression ratios"
                )

            if violation_analysis["fps_issues"] > len(failed_df) * 0.3:
                click.echo(
                    "• Check frame rate preservation in frame reduction pipelines"
                )
                click.echo("• Consider content-type specific validation thresholds")

            if violation_analysis["artifact_issues"] > len(failed_df) * 0.3:
                click.echo(
                    "• Avoid aggressive frame reduction for content with disposal artifacts"
                )
                click.echo(
                    "• Consider less lossy compression for artifact-prone content"
                )

    except Exception as e:
        handle_generic_error("Analyze validation thresholds", e)


# Programmatic access functions for debugging workflows
def get_validation_summary(csv_file: Path) -> dict:
    """Get validation summary for programmatic access."""
    df = pd.read_csv(csv_file)

    if df.empty:
        return {"error": "Empty CSV file"}

    validation_counts = df["validation_status"].value_counts()
    total_results = len(df)

    return {
        "total_results": total_results,
        "pass_rate": (validation_counts.get("PASS", 0) / total_results) * 100,
        "status_distribution": dict(validation_counts),
        "avg_issues_per_result": df["validation_issues_count"].mean()
        if "validation_issues_count" in df.columns
        else 0,
        "most_failed_pipelines": df[df["validation_status"] != "PASS"]["pipeline_id"]
        .value_counts()
        .head(3)
        .to_dict()
        if len(df[df["validation_status"] != "PASS"]) > 0
        else {},
    }


def filter_validation_results(
    csv_file: Path, status: str | None = None, category: str | None = None
) -> pd.DataFrame:
    """Filter validation results programmatically for debugging workflows."""
    df = pd.read_csv(csv_file)

    if status:
        df = df[df["validation_status"] == status.upper()]

    if category:
        category_keywords = {
            "fps": ["fps", "frame rate", "timing"],
            "quality": ["quality", "ssim", "psnr", "composite"],
            "artifacts": ["artifact", "disposal", "corruption"],
            "efficiency": ["efficiency", "compression ratio"],
        }

        keywords = category_keywords.get(category.lower(), [category.lower()])

        def has_category_issue(messages: Any) -> bool:
            if pd.isna(messages):
                return False
            messages_lower = str(messages).lower()
            return any(keyword in messages_lower for keyword in keywords)

        df = df[df["validation_messages"].apply(has_category_issue)]

    return df
