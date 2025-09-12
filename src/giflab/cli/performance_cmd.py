"""
Phase 7: Performance Monitoring CLI Commands

This module provides comprehensive CLI commands for the continuous performance 
monitoring and alerting system, enabling users to:

- Monitor performance baselines and regression detection
- View performance history and trends 
- Control continuous monitoring
- Validate Phase 6 optimization effectiveness
- Integration with CI/CD pipelines

CLI Commands:
- performance status: Show monitoring status and recent alerts
- performance baseline: Manage performance baselines
- performance monitor: Control continuous monitoring
- performance history: View performance trends and history
- performance validate: Validate Phase 6 optimization effectiveness
- performance ci: CI/CD integration commands
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ..config import MONITORING
from ..monitoring.performance_regression import (
    create_performance_monitor,
    RegressionDetector,
    PerformanceHistory
)
from ..benchmarks.phase_4_3_benchmarking import Phase43Benchmarker, BenchmarkScenario

console = Console()


@click.group("performance")
def performance() -> None:
    """Performance monitoring and regression detection commands."""
    pass


@performance.command("status")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed status information")
@click.option("--json", "output_json", is_flag=True, help="Output status as JSON")
def performance_status(verbose: bool, output_json: bool) -> None:
    """Show performance monitoring status and recent alerts."""
    
    try:
        # Get monitoring configuration
        perf_config = MONITORING.get("performance", {})
        enabled = perf_config.get("enabled", False) or os.getenv("GIFLAB_ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"
        
        # Create monitor to get status
        monitor = create_performance_monitor()
        status = monitor.get_monitoring_status()
        
        # Get detector status
        detector_summary = monitor.detector.get_baseline_summary()
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in monitor.alert_manager.get_active_alerts()
            if "performance_regression" in alert.name
        ]
        
        status_data = {
            "performance_monitoring": {
                "enabled": enabled,
                "configured": bool(perf_config),
                "monitoring_active": status.get("enabled", False),
                "monitoring_interval": status.get("monitoring_interval", 0),
                "thread_status": "alive" if status.get("thread_alive", False) else "stopped"
            },
            "baselines": {
                "count": detector_summary.get("baseline_count", 0),
                "scenarios": detector_summary.get("scenarios", []),
                "last_updated": detector_summary.get("last_updated", "never"),
                "regression_threshold": f"{detector_summary.get('regression_threshold', 0) * 100:.1f}%",
                "confidence_level": f"{detector_summary.get('confidence_level', 0) * 100:.0f}%"
            },
            "alerts": {
                "active_count": len(recent_alerts),
                "recent_alerts": [
                    {
                        "name": alert.name,
                        "severity": alert.severity.name,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "details": alert.details
                    }
                    for alert in recent_alerts[:5]  # Show last 5 alerts
                ]
            },
            "configuration": {
                "regression_threshold": f"{perf_config.get('regression_threshold', 0.10) * 100:.1f}%",
                "monitoring_interval": f"{perf_config.get('monitoring_interval', 3600)}s",
                "history_retention": f"{perf_config.get('max_history_days', 30)} days",
                "phase6_validation": perf_config.get("validate_phase6", True)
            }
        }
        
        if output_json:
            click.echo(json.dumps(status_data, indent=2))
            return
        
        # Rich formatted output
        console.print()
        console.print("[bold blue]📊 Performance Monitoring Status[/bold blue]")
        console.print()
        
        # Main status
        status_color = "green" if enabled else "red"
        status_text = "ENABLED" if enabled else "DISABLED"
        console.print(f"[bold]Overall Status:[/bold] [{status_color}]{status_text}[/{status_color}]")
        
        if enabled:
            monitor_status = "RUNNING" if status.get("enabled", False) else "STOPPED"
            monitor_color = "green" if status.get("enabled", False) else "yellow"
            console.print(f"[bold]Monitoring:[/bold] [{monitor_color}]{monitor_status}[/{monitor_color}]")
        
        console.print()
        
        # Baselines table
        if detector_summary.get("baseline_count", 0) > 0:
            baselines_table = Table(title="Performance Baselines", show_header=True)
            baselines_table.add_column("Scenario", style="cyan")
            baselines_table.add_column("Status", style="green")
            baselines_table.add_column("Last Updated", style="dim")
            
            for scenario in detector_summary.get("scenarios", []):
                baselines_table.add_row(
                    scenario,
                    "✅ Active",
                    detector_summary.get("last_updated", "Unknown")
                )
            
            console.print(baselines_table)
        else:
            console.print("[yellow]No performance baselines available[/yellow]")
            console.print("[dim]Run 'giflab performance baseline create' to establish baselines[/dim]")
        
        console.print()
        
        # Recent alerts
        if recent_alerts:
            alerts_table = Table(title="Recent Performance Alerts", show_header=True)
            alerts_table.add_column("Scenario", style="cyan")
            alerts_table.add_column("Severity", style="red")
            alerts_table.add_column("Regression", style="yellow")
            alerts_table.add_column("Time", style="dim")
            
            for alert in recent_alerts[:5]:
                severity_color = {"INFO": "blue", "WARNING": "yellow", "CRITICAL": "red"}.get(alert.severity.name, "white")
                regression_pct = alert.details.get("regression_percentage", "Unknown")
                
                alerts_table.add_row(
                    alert.details.get("scenario", "Unknown"),
                    f"[{severity_color}]{alert.severity.name}[/{severity_color}]",
                    regression_pct,
                    alert.timestamp.strftime("%H:%M:%S")
                )
            
            console.print(alerts_table)
        else:
            console.print("[green]✅ No recent performance alerts[/green]")
        
        # Configuration details (verbose mode)
        if verbose:
            console.print()
            config_table = Table(title="Configuration Details", show_header=True)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_items = [
                ("Regression Threshold", status_data["configuration"]["regression_threshold"]),
                ("Monitoring Interval", status_data["configuration"]["monitoring_interval"]),
                ("History Retention", status_data["configuration"]["history_retention"]),
                ("Phase 6 Validation", "Enabled" if status_data["configuration"]["phase6_validation"] else "Disabled"),
                ("CI Integration", "Enabled" if perf_config.get("fail_ci_on_regression", False) else "Disabled")
            ]
            
            for setting, value in config_items:
                config_table.add_row(setting, str(value))
            
            console.print(config_table)
        
        console.print()
        
        # Enable suggestions
        if not enabled:
            console.print(Panel(
                "To enable performance monitoring:\n\n"
                "1. Set environment variable: GIFLAB_ENABLE_PERFORMANCE_MONITORING=true\n"
                "2. Create baselines: giflab performance baseline create\n"
                "3. Start monitoring: giflab performance monitor start",
                title="[yellow]Enable Performance Monitoring[/yellow]",
                border_style="yellow"
            ))
        
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error getting performance status: {e}[/red]")
        sys.exit(1)


@performance.command("baseline")
@click.argument("action", type=click.Choice(["create", "update", "list", "clear"]))
@click.option("--scenario", "-s", help="Specific scenario to baseline (default: all)")
@click.option("--iterations", "-i", type=int, default=3, help="Number of iterations for baseline creation")
@click.option("--force", "-f", is_flag=True, help="Force overwrite existing baselines")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
def performance_baseline(action: str, scenario: Optional[str], iterations: int, force: bool, output_json: bool) -> None:
    """Manage performance baselines for regression detection."""
    
    try:
        monitor = create_performance_monitor()
        detector = monitor.detector
        benchmarker = monitor.benchmarker
        
        if action == "list":
            summary = detector.get_baseline_summary()
            
            if output_json:
                click.echo(json.dumps(summary, indent=2))
                return
            
            console.print()
            console.print("[bold blue]📈 Performance Baselines[/bold blue]")
            console.print()
            
            if summary.get("baseline_count", 0) == 0:
                console.print("[yellow]No baselines available[/yellow]")
                return
            
            baselines_table = Table(show_header=True, header_style="bold magenta")
            baselines_table.add_column("Scenario", style="cyan", width=25)
            baselines_table.add_column("Processing Time", style="green", width=16)
            baselines_table.add_column("Memory Usage", style="blue", width=16)
            baselines_table.add_column("Samples", style="yellow", width=10)
            baselines_table.add_column("Updated", style="dim", width=12)
            
            for scenario_name in summary.get("scenarios", []):
                baseline = detector.baselines.get(scenario_name)
                if baseline:
                    baselines_table.add_row(
                        scenario_name,
                        f"{baseline.mean_processing_time:.2f}s ± {baseline.std_processing_time:.2f}",
                        f"{baseline.mean_memory_usage:.1f}MB ± {baseline.std_memory_usage:.1f}",
                        str(baseline.sample_count),
                        baseline.last_updated.strftime("%m/%d %H:%M")
                    )
            
            console.print(baselines_table)
        
        elif action == "create" or action == "update":
            # Get scenarios to baseline
            if scenario:
                scenarios = [s for s in benchmarker.get_available_scenarios() if s.name == scenario]
                if not scenarios:
                    raise click.ClickException(f"Scenario '{scenario}' not found")
            else:
                # Use default scenarios for baselining
                scenarios = [
                    BenchmarkScenario("small_gif_basic", "Small GIF baseline", 20, (50, 50), ["processing_time", "memory_usage"], (0.1, 2.0)),
                    BenchmarkScenario("medium_gif_comprehensive", "Medium GIF baseline", 50, (100, 100), ["processing_time", "memory_usage"], (0.5, 5.0))
                ]
            
            results_data = {}
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                for scenario in scenarios:
                    # Check if baseline exists and force not specified
                    if scenario.name in detector.baselines and not force and action == "create":
                        console.print(f"[yellow]Baseline for {scenario.name} already exists (use --force to overwrite)[/yellow]")
                        continue
                    
                    task = progress.add_task(f"Running baseline for {scenario.name}...", total=iterations)
                    
                    # Run benchmark iterations
                    results = []
                    for i in range(iterations):
                        try:
                            iteration_results = benchmarker.run_scenario(scenario, iterations=1)
                            if iteration_results:
                                results.extend(iteration_results)
                            progress.update(task, advance=1)
                        except Exception as e:
                            console.print(f"[red]Failed iteration {i+1} for {scenario.name}: {e}[/red]")
                    
                    progress.remove_task(task)
                    
                    if len(results) >= detector.detector.regression_threshold:
                        # Update baseline
                        detector.update_baseline(scenario.name, results, min_samples=len(results))
                        
                        # Store results
                        results_data[scenario.name] = {
                            "samples": len(results),
                            "mean_time": sum(r.processing_time for r in results) / len(results),
                            "mean_memory": sum(r.mean_memory_usage for r in results) / len(results),
                            "created": datetime.now().isoformat()
                        }
                        
                        console.print(f"[green]✅ Created baseline for {scenario.name} ({len(results)} samples)[/green]")
                    else:
                        console.print(f"[red]❌ Insufficient samples for {scenario.name} baseline[/red]")
            
            if output_json:
                click.echo(json.dumps(results_data, indent=2))
        
        elif action == "clear":
            if scenario:
                if scenario in detector.baselines:
                    del detector.baselines[scenario]
                    detector._save_baselines()
                    console.print(f"[green]✅ Cleared baseline for {scenario}[/green]")
                else:
                    console.print(f"[yellow]No baseline found for {scenario}[/yellow]")
            else:
                detector.baselines.clear()
                detector._save_baselines()
                console.print("[green]✅ Cleared all baselines[/green]")
    
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error managing baselines: {e}[/red]")
        sys.exit(1)


@performance.command("monitor")
@click.argument("action", type=click.Choice(["start", "stop", "status"]))
@click.option("--interval", "-i", type=int, help="Monitoring interval in seconds")
@click.option("--json", "output_json", is_flag=True, help="Output status as JSON")
def performance_monitor(action: str, interval: Optional[int], output_json: bool) -> None:
    """Control continuous performance monitoring."""
    
    try:
        monitor = create_performance_monitor()
        
        if action == "start":
            # Check if performance monitoring is enabled
            enabled = (MONITORING.get("performance", {}).get("enabled", False) or 
                      os.getenv("GIFLAB_ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true")
            
            if not enabled:
                error_msg = ("Performance monitoring is disabled. Enable with:\n"
                           "GIFLAB_ENABLE_PERFORMANCE_MONITORING=true")
                if output_json:
                    click.echo(json.dumps({"error": error_msg}))
                else:
                    console.print(f"[red]{error_msg}[/red]")
                sys.exit(1)
            
            # Check if baselines exist
            if len(monitor.detector.baselines) == 0:
                error_msg = ("No performance baselines available. Create baselines first:\n"
                           "giflab performance baseline create")
                if output_json:
                    click.echo(json.dumps({"error": error_msg}))
                else:
                    console.print(f"[red]{error_msg}[/red]")
                sys.exit(1)
            
            # Set custom interval if provided
            if interval:
                monitor.monitoring_interval = interval
            
            monitor.start_monitoring()
            
            result = {
                "status": "started",
                "monitoring_interval": monitor.monitoring_interval,
                "scenarios": len(monitor.monitoring_scenarios),
                "baselines": len(monitor.detector.baselines)
            }
            
            if output_json:
                click.echo(json.dumps(result, indent=2))
            else:
                console.print("[green]✅ Performance monitoring started[/green]")
                console.print(f"[dim]Interval: {monitor.monitoring_interval}s, Scenarios: {len(monitor.monitoring_scenarios)}[/dim]")
        
        elif action == "stop":
            monitor.stop_monitoring()
            
            result = {"status": "stopped"}
            
            if output_json:
                click.echo(json.dumps(result, indent=2))
            else:
                console.print("[yellow]⏹️ Performance monitoring stopped[/yellow]")
        
        elif action == "status":
            status = monitor.get_monitoring_status()
            
            if output_json:
                click.echo(json.dumps(status, indent=2))
            else:
                console.print()
                console.print("[bold blue]🔍 Continuous Monitoring Status[/bold blue]")
                console.print()
                
                status_color = "green" if status.get("enabled", False) else "red"
                status_text = "RUNNING" if status.get("enabled", False) else "STOPPED"
                console.print(f"[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]")
                
                if status.get("enabled", False):
                    console.print(f"[bold]Interval:[/bold] {status.get('monitoring_interval', 0)}s")
                    console.print(f"[bold]Scenarios:[/bold] {status.get('scenarios_monitored', 0)}")
                    console.print(f"[bold]Baselines:[/bold] {status.get('detector_baselines', 0)}")
                    console.print(f"[bold]Recent Alerts:[/bold] {status.get('recent_alerts', 0)}")
    
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error controlling monitoring: {e}[/red]")
        sys.exit(1)


@performance.command("history")
@click.option("--scenario", "-s", help="Show history for specific scenario")
@click.option("--days", "-d", type=int, default=7, help="Number of days of history to show")
@click.option("--metric", "-m", type=click.Choice(["processing_time", "memory_usage"]), 
              default="processing_time", help="Metric to analyze")
@click.option("--trend", "-t", is_flag=True, help="Show performance trend analysis")
@click.option("--json", "output_json", is_flag=True, help="Output history as JSON")
def performance_history(scenario: Optional[str], days: int, metric: str, trend: bool, output_json: bool) -> None:
    """View performance history and trends."""
    
    try:
        monitor = create_performance_monitor()
        history = monitor.history
        
        # Get scenarios to show
        if scenario:
            scenarios = [scenario] if scenario in monitor.detector.baselines else []
            if not scenarios:
                error_msg = f"No history found for scenario: {scenario}"
                if output_json:
                    click.echo(json.dumps({"error": error_msg}))
                else:
                    console.print(f"[red]{error_msg}[/red]")
                sys.exit(1)
        else:
            scenarios = list(monitor.detector.baselines.keys())
        
        history_data = {}
        
        for scenario_name in scenarios:
            records = history.get_recent_history(scenario_name, days)
            
            if trend:
                trend_slope = history.calculate_trend(scenario_name, metric, days)
            else:
                trend_slope = None
            
            history_data[scenario_name] = {
                "records": records,
                "record_count": len(records),
                "metric": metric,
                "trend_slope": trend_slope,
                "days": days
            }
        
        if output_json:
            click.echo(json.dumps(history_data, indent=2))
            return
        
        # Rich formatted output
        console.print()
        console.print(f"[bold blue]📊 Performance History ({days} days)[/bold blue]")
        console.print()
        
        for scenario_name, data in history_data.items():
            records = data["records"]
            
            if not records:
                console.print(f"[yellow]No history available for {scenario_name}[/yellow]")
                continue
            
            console.print(f"[bold cyan]{scenario_name}[/bold cyan]")
            
            # Summary statistics
            values = [record.get(metric, 0) for record in records]
            if values:
                avg_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)
                
                console.print(f"[dim]Records: {len(records)}, "
                             f"Avg: {avg_value:.2f}, "
                             f"Min: {min_value:.2f}, "
                             f"Max: {max_value:.2f}[/dim]")
                
                if trend and data["trend_slope"] is not None:
                    trend_direction = "📈 Improving" if data["trend_slope"] < 0 else "📉 Degrading" if data["trend_slope"] > 0 else "➡️ Stable"
                    console.print(f"[dim]Trend: {trend_direction}[/dim]")
            
            # Recent records table
            if len(records) > 0:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Time", style="cyan", width=16)
                table.add_column("Processing Time", style="green", width=15)
                table.add_column("Memory Usage", style="blue", width=15)
                table.add_column("Success Rate", style="yellow", width=12)
                
                # Show last 10 records
                for record in records[-10:]:
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    table.add_row(
                        timestamp.strftime("%m/%d %H:%M:%S"),
                        f"{record.get('processing_time', 0):.2f}s",
                        f"{record.get('memory_usage', 0):.1f}MB",
                        f"{record.get('success_rate', 0) * 100:.0f}%"
                    )
                
                console.print(table)
            
            console.print()
    
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error retrieving performance history: {e}[/red]")
        sys.exit(1)


@performance.command("validate")
@click.option("--check-phase6", is_flag=True, help="Validate Phase 6 optimization effectiveness")
@click.option("--scenario", "-s", help="Specific scenario to validate")
@click.option("--json", "output_json", is_flag=True, help="Output validation results as JSON")
def performance_validate(check_phase6: bool, scenario: Optional[str], output_json: bool) -> None:
    """Validate Phase 6 optimization effectiveness and overall performance."""
    
    try:
        monitor = create_performance_monitor()
        perf_config = MONITORING.get("performance", {})
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "phase6_validation": None,
            "baseline_validation": None,
            "overall_status": "unknown"
        }
        
        if check_phase6:
            # Check if Phase 6 optimizations are enabled
            phase6_enabled = os.getenv("GIFLAB_ENABLE_PHASE6_OPTIMIZATION", "false").lower() == "true"
            expected_speedup = perf_config.get("phase6_baseline_speedup", 5.04)
            
            if not phase6_enabled:
                validation_results["phase6_validation"] = {
                    "enabled": False,
                    "status": "disabled",
                    "message": "Phase 6 optimizations are disabled"
                }
            else:
                # Run performance test to validate speedup
                benchmarker = Phase43Benchmarker()
                
                # Run with Phase 6 enabled
                scenario_obj = BenchmarkScenario(
                    "phase6_validation", 
                    "Phase 6 validation test",
                    20,
                    (50, 50),
                    ["processing_time", "memory_usage"],
                    (0.1, 2.0)
                )
                
                with Progress(SpinnerColumn(), TextColumn("Validating Phase 6 performance..."), console=console) as progress:
                    task = progress.add_task("Running validation...", total=1)
                    results = benchmarker.run_scenario(scenario_obj, iterations=1)
                    progress.update(task, advance=1)
                
                if results:
                    result = results[0]
                    baseline_time = monitor.detector.baselines.get("small_gif_basic")
                    
                    if baseline_time:
                        actual_speedup = baseline_time.mean_processing_time / result.processing_time
                        speedup_retention = actual_speedup / expected_speedup
                        
                        validation_results["phase6_validation"] = {
                            "enabled": True,
                            "expected_speedup": expected_speedup,
                            "actual_speedup": actual_speedup,
                            "speedup_retention": speedup_retention,
                            "status": "optimal" if speedup_retention >= 0.9 else "degraded" if speedup_retention >= 0.8 else "critical",
                            "current_time": result.processing_time,
                            "baseline_time": baseline_time.mean_processing_time
                        }
                    else:
                        validation_results["phase6_validation"] = {
                            "enabled": True,
                            "status": "no_baseline",
                            "message": "No baseline available for comparison"
                        }
        
        # Overall validation status
        if validation_results["phase6_validation"]:
            phase6_status = validation_results["phase6_validation"]["status"]
            if phase6_status == "optimal":
                validation_results["overall_status"] = "excellent"
            elif phase6_status in ["degraded", "no_baseline"]:
                validation_results["overall_status"] = "warning"
            elif phase6_status == "disabled":
                validation_results["overall_status"] = "unknown"
            else:
                validation_results["overall_status"] = "critical"
        
        if output_json:
            click.echo(json.dumps(validation_results, indent=2))
            return
        
        # Rich formatted output
        console.print()
        console.print("[bold blue]🔬 Performance Validation[/bold blue]")
        console.print()
        
        # Phase 6 validation results
        if validation_results["phase6_validation"]:
            phase6_data = validation_results["phase6_validation"]
            
            console.print("[bold]Phase 6 Optimization Validation:[/bold]")
            
            if not phase6_data["enabled"]:
                console.print("[yellow]⚠️  Phase 6 optimizations are disabled[/yellow]")
                console.print("[dim]Enable with: GIFLAB_ENABLE_PHASE6_OPTIMIZATION=true[/dim]")
            else:
                status = phase6_data["status"]
                
                if status == "optimal":
                    console.print("[green]✅ Phase 6 optimizations performing optimally[/green]")
                    console.print(f"[dim]Speedup: {phase6_data['actual_speedup']:.1f}x "
                                 f"(Expected: {phase6_data['expected_speedup']:.1f}x, "
                                 f"Retention: {phase6_data['speedup_retention']*100:.1f}%)[/dim]")
                elif status == "degraded":
                    console.print("[yellow]⚠️  Phase 6 performance has degraded[/yellow]")
                    console.print(f"[dim]Current speedup: {phase6_data['actual_speedup']:.1f}x "
                                 f"(Expected: {phase6_data['expected_speedup']:.1f}x)[/dim]")
                elif status == "critical":
                    console.print("[red]❌ Phase 6 performance critically degraded[/red]")
                    console.print(f"[dim]Current speedup: {phase6_data['actual_speedup']:.1f}x "
                                 f"(Expected: {phase6_data['expected_speedup']:.1f}x)[/dim]")
                elif status == "no_baseline":
                    console.print("[yellow]⚠️  No baseline available for validation[/yellow]")
                    console.print("[dim]Create baselines: giflab performance baseline create[/dim]")
        
        console.print()
        
        # Overall status
        overall_status = validation_results["overall_status"]
        if overall_status == "excellent":
            console.print("[bold green]🎯 Overall Performance Status: EXCELLENT[/bold green]")
        elif overall_status == "warning":
            console.print("[bold yellow]⚠️  Overall Performance Status: WARNING[/bold yellow]")
        elif overall_status == "critical":
            console.print("[bold red]🚨 Overall Performance Status: CRITICAL[/bold red]")
        else:
            console.print("[bold dim]❓ Overall Performance Status: UNKNOWN[/bold dim]")
    
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error validating performance: {e}[/red]")
        sys.exit(1)


@performance.command("ci")
@click.argument("action", type=click.Choice(["check", "gate"]))
@click.option("--threshold", "-t", type=float, help="Regression threshold for CI gate (default from config)")
@click.option("--scenarios", "-s", help="Comma-separated list of scenarios to test")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
def performance_ci(action: str, threshold: Optional[float], scenarios: Optional[str], output_json: bool) -> None:
    """CI/CD integration commands for performance validation."""
    
    try:
        monitor = create_performance_monitor()
        perf_config = MONITORING.get("performance", {})
        
        # Get CI configuration
        ci_threshold = threshold or perf_config.get("ci_regression_threshold", 0.15)
        ci_scenarios = scenarios.split(",") if scenarios else perf_config.get("ci_scenarios", ["small_gif_basic"])
        
        results = {
            "action": action,
            "threshold": ci_threshold,
            "scenarios_tested": [],
            "regressions_detected": [],
            "overall_result": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        benchmarker = Phase43Benchmarker()
        
        # Run scenarios
        for scenario_name in ci_scenarios:
            try:
                scenario = BenchmarkScenario(
                    scenario_name,
                    f"CI validation for {scenario_name}",
                    30,
                    (100, 100),
                    ["processing_time", "memory_usage"],
                    (0.5, 5.0)
                )
                
                if not output_json:
                    console.print(f"[dim]Running CI check for {scenario_name}...[/dim]")
                
                # Run benchmark
                benchmark_results = benchmarker.run_scenario(scenario, iterations=1)
                
                if not benchmark_results:
                    results["scenarios_tested"].append({
                        "scenario": scenario_name,
                        "status": "failed",
                        "error": "No benchmark results"
                    })
                    continue
                
                result = benchmark_results[0]
                
                # Check for regressions
                regression_alerts = monitor.detector.detect_regressions(scenario_name, result)
                
                scenario_result = {
                    "scenario": scenario_name,
                    "processing_time": result.processing_time,
                    "memory_usage": result.mean_memory_usage,
                    "success_rate": result.success_rate,
                    "regressions": []
                }
                
                # Check if regressions exceed CI threshold
                significant_regressions = []
                for alert in regression_alerts:
                    if alert.regression_severity >= ci_threshold:
                        significant_regressions.append({
                            "metric": alert.metric_type,
                            "regression_percentage": alert.regression_severity,
                            "current_value": alert.current_value,
                            "baseline_value": alert.baseline_mean
                        })
                
                scenario_result["regressions"] = significant_regressions
                scenario_result["status"] = "failed" if significant_regressions else "passed"
                
                results["scenarios_tested"].append(scenario_result)
                results["regressions_detected"].extend(significant_regressions)
                
            except Exception as e:
                results["scenarios_tested"].append({
                    "scenario": scenario_name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Determine overall result
        if any(s["status"] == "failed" for s in results["scenarios_tested"]):
            results["overall_result"] = "failed"
            exit_code = 1
        elif any(s["status"] == "error" for s in results["scenarios_tested"]):
            results["overall_result"] = "error" 
            exit_code = 1
        else:
            results["overall_result"] = "passed"
            exit_code = 0
        
        if output_json:
            click.echo(json.dumps(results, indent=2))
        else:
            # Rich formatted output
            console.print()
            console.print("[bold blue]🔍 CI Performance Check[/bold blue]")
            console.print()
            
            # Results table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Scenario", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Processing Time", style="green")
            table.add_column("Memory Usage", style="blue")
            table.add_column("Regressions", style="red")
            
            for scenario_result in results["scenarios_tested"]:
                status = scenario_result["status"]
                status_color = {"passed": "green", "failed": "red", "error": "yellow"}.get(status, "white")
                status_text = f"[{status_color}]{status.upper()}[/{status_color}]"
                
                regressions_text = ""
                if scenario_result.get("regressions"):
                    reg_count = len(scenario_result["regressions"])
                    regressions_text = f"{reg_count} detected"
                else:
                    regressions_text = "None"
                
                table.add_row(
                    scenario_result["scenario"],
                    status_text,
                    f"{scenario_result.get('processing_time', 0):.2f}s",
                    f"{scenario_result.get('memory_usage', 0):.1f}MB",
                    regressions_text
                )
            
            console.print(table)
            console.print()
            
            # Overall result
            overall_result = results["overall_result"]
            if overall_result == "passed":
                console.print("[bold green]✅ CI Performance Check: PASSED[/bold green]")
            elif overall_result == "failed":
                console.print("[bold red]❌ CI Performance Check: FAILED[/bold red]")
                console.print(f"[dim]Regression threshold: {ci_threshold * 100:.1f}%[/dim]")
            else:
                console.print("[bold yellow]⚠️  CI Performance Check: ERROR[/bold yellow]")
        
        if action == "gate":
            sys.exit(exit_code)
    
    except Exception as e:
        if output_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error running CI performance check: {e}[/red]")
        sys.exit(1)