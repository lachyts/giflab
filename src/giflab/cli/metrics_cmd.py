"""
CLI commands for performance metrics monitoring.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

from ..monitoring import get_metrics_collector, MetricType
from ..monitoring.backends import create_backend
from ..config import MONITORING

console = Console()


@click.group()
def metrics():
    """Performance monitoring and metrics commands."""
    pass


@metrics.command()
@click.option(
    "--window",
    "-w",
    type=int,
    default=300,
    help="Time window in seconds (default: 300)",
)
@click.option(
    "--system",
    "-s",
    type=click.Choice(["frame", "validation", "resize", "sampling", "lazy", "all"]),
    default="all",
    help="System to show metrics for",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def status(window: int, system: str, format: str):
    """View current metrics status and statistics."""
    collector = get_metrics_collector()
    
    # Get summaries for the specified window
    summaries = collector.get_summary(window_seconds=window)
    
    if not summaries:
        console.print("[yellow]No metrics available for the specified window.[/yellow]")
        return
    
    # Filter by system if specified
    if system != "all":
        system_prefix = {
            "frame": "cache.frame",
            "validation": "cache.validation",
            "resize": "cache.resize",
            "sampling": "sampling",
            "lazy": "lazy_import",
        }[system]
        summaries = [s for s in summaries if s.name.startswith(system_prefix)]
    
    if format == "json":
        # JSON output
        data = [
            {
                "name": s.name,
                "type": s.metric_type.value,
                "count": s.count,
                "mean": s.mean,
                "median": s.median,
                "p95": s.p95,
                "p99": s.p99,
                "min": s.min,
                "max": s.max,
                "tags": s.tags,
            }
            for s in summaries
        ]
        click.echo(json.dumps(data, indent=2))
    
    elif format == "csv":
        # CSV output
        click.echo("name,type,count,mean,median,p95,p99,min,max")
        for s in summaries:
            click.echo(
                f"{s.name},{s.metric_type.value},{s.count},"
                f"{s.mean:.3f},{s.median:.3f},{s.p95:.3f},"
                f"{s.p99:.3f},{s.min:.3f},{s.max:.3f}"
            )
    
    else:
        # Table output
        table = Table(title=f"Metrics Status (Last {window}s)")
        table.add_column("Metric", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Count", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")
        table.add_column("Max", justify="right")
        
        for summary in summaries:
            # Format values based on metric type
            if "duration" in summary.name or summary.metric_type == MetricType.TIMER:
                mean_str = f"{summary.mean * 1000:.2f}ms"
                p95_str = f"{summary.p95 * 1000:.2f}ms"
                p99_str = f"{summary.p99 * 1000:.2f}ms"
                max_str = f"{summary.max * 1000:.2f}ms"
            elif "rate" in summary.name or "ratio" in summary.name:
                mean_str = f"{summary.mean:.1%}"
                p95_str = f"{summary.p95:.1%}"
                p99_str = f"{summary.p99:.1%}"
                max_str = f"{summary.max:.1%}"
            else:
                mean_str = f"{summary.mean:.2f}"
                p95_str = f"{summary.p95:.2f}"
                p99_str = f"{summary.p99:.2f}"
                max_str = f"{summary.max:.2f}"
            
            table.add_row(
                summary.name,
                summary.metric_type.value,
                str(summary.count),
                mean_str,
                p95_str,
                p99_str,
                max_str,
            )
        
        console.print(table)


@metrics.command()
@click.option(
    "--interval",
    "-i",
    type=int,
    default=5,
    help="Update interval in seconds",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=0,
    help="Monitor duration in seconds (0 = infinite)",
)
def monitor(interval: int, duration: int):
    """Live monitoring dashboard for all systems."""
    collector = get_metrics_collector()
    start_time = time.time()
    
    def generate_dashboard():
        """Generate dashboard content."""
        # Get recent metrics
        summaries = collector.get_summary(window_seconds=60)
        
        # Calculate cache hit rates
        cache_stats = {}
        for cache_type in ["frame", "validation", "resize"]:
            hits = sum(s.sum for s in summaries if s.name == f"cache.{cache_type}.hits")
            misses = sum(s.sum for s in summaries if s.name == f"cache.{cache_type}.misses")
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            cache_stats[cache_type] = {
                "hits": int(hits),
                "misses": int(misses),
                "hit_rate": hit_rate,
            }
        
        # Build dashboard layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        
        # Header
        elapsed = time.time() - start_time
        header_text = Text(
            f"GifLab Performance Monitor | Elapsed: {elapsed:.0f}s | Interval: {interval}s",
            style="bold cyan",
        )
        layout["header"].update(Panel(header_text))
        
        # Body - Cache Statistics
        cache_table = Table(title="Cache Performance", expand=True)
        cache_table.add_column("Cache", style="cyan")
        cache_table.add_column("Hit Rate", justify="right")
        cache_table.add_column("Hits", justify="right", style="green")
        cache_table.add_column("Misses", justify="right", style="red")
        
        for cache_type, stats in cache_stats.items():
            hit_rate_style = "green" if stats["hit_rate"] >= 60 else "yellow" if stats["hit_rate"] >= 40 else "red"
            cache_table.add_row(
                cache_type.capitalize(),
                f"[{hit_rate_style}]{stats['hit_rate']:.1f}%[/{hit_rate_style}]",
                str(stats["hits"]),
                str(stats["misses"]),
            )
        
        # Operation timings
        timing_table = Table(title="Operation Timings (P95)", expand=True)
        timing_table.add_column("Operation", style="cyan")
        timing_table.add_column("Duration", justify="right")
        
        timing_metrics = [s for s in summaries if "duration" in s.name]
        for metric in timing_metrics[:10]:  # Top 10
            timing_table.add_row(
                metric.name.replace(".", " "),
                f"{metric.p95 * 1000:.2f}ms",
            )
        
        layout["body"].split_row(
            Layout(cache_table),
            Layout(timing_table),
        )
        
        # Footer
        footer_text = Text(
            "Press Ctrl+C to exit | Metrics are updated in real-time",
            style="dim",
        )
        layout["footer"].update(Panel(footer_text))
        
        return layout
    
    try:
        with Live(generate_dashboard(), refresh_per_second=1, console=console) as live:
            end_time = time.time() + duration if duration > 0 else float('inf')
            
            while time.time() < end_time:
                time.sleep(interval)
                live.update(generate_dashboard())
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped.[/yellow]")


@metrics.command()
@click.option(
    "--check",
    "-c",
    type=click.Choice(["cache-hit-rate", "memory-usage", "response-time", "all"]),
    default="all",
    help="Alert check to run",
)
@click.option(
    "--window",
    "-w",
    type=int,
    default=300,
    help="Time window for checks in seconds",
)
def alerts(check: str, window: int):
    """Check for alert conditions and display warnings."""
    collector = get_metrics_collector()
    summaries = collector.get_summary(window_seconds=window)
    
    alerts_config = MONITORING.get("alerts", {})
    triggered_alerts = []
    
    # Check cache hit rates
    if check in ["cache-hit-rate", "all"]:
        for cache_type in ["frame", "validation", "resize"]:
            hits = sum(s.sum for s in summaries if s.name == f"cache.{cache_type}.hits")
            misses = sum(s.sum for s in summaries if s.name == f"cache.{cache_type}.misses")
            total = hits + misses
            
            if total > 0:
                hit_rate = hits / total
                
                if hit_rate < alerts_config.get("cache_hit_rate_critical", 0.2):
                    triggered_alerts.append({
                        "level": "CRITICAL",
                        "system": f"{cache_type} cache",
                        "message": f"Hit rate {hit_rate:.1%} below critical threshold",
                    })
                elif hit_rate < alerts_config.get("cache_hit_rate_warning", 0.4):
                    triggered_alerts.append({
                        "level": "WARNING",
                        "system": f"{cache_type} cache",
                        "message": f"Hit rate {hit_rate:.1%} below warning threshold",
                    })
    
    # Check memory usage
    if check in ["memory-usage", "all"]:
        memory_metrics = [s for s in summaries if "memory_usage" in s.name]
        for metric in memory_metrics:
            # Assuming metrics are in MB, check against limits
            if "frame" in metric.name:
                limit = 500  # From FRAME_CACHE config
            elif "validation" in metric.name:
                limit = 100  # From VALIDATION_CACHE config
            elif "resize" in metric.name:
                limit = 200  # From config
            else:
                continue
            
            usage_ratio = metric.max / limit if limit > 0 else 0
            
            if usage_ratio > alerts_config.get("memory_usage_critical", 0.95):
                triggered_alerts.append({
                    "level": "CRITICAL",
                    "system": metric.name,
                    "message": f"Memory usage {metric.max:.1f}MB ({usage_ratio:.1%} of limit)",
                })
            elif usage_ratio > alerts_config.get("memory_usage_warning", 0.8):
                triggered_alerts.append({
                    "level": "WARNING",
                    "system": metric.name,
                    "message": f"Memory usage {metric.max:.1f}MB ({usage_ratio:.1%} of limit)",
                })
    
    # Display alerts
    if triggered_alerts:
        console.print("\n[bold red]⚠ ALERTS TRIGGERED ⚠[/bold red]\n")
        
        for alert in triggered_alerts:
            color = "red" if alert["level"] == "CRITICAL" else "yellow"
            console.print(
                f"[{color}][{alert['level']}][/{color}] "
                f"[cyan]{alert['system']}[/cyan]: {alert['message']}"
            )
    else:
        console.print("[green]✓ No alerts triggered[/green]")


@metrics.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Export file path (JSON format)",
)
@click.option(
    "--start",
    "-s",
    type=str,
    help="Start time (ISO format or relative like '-1h')",
)
@click.option(
    "--end",
    "-e",
    type=str,
    help="End time (ISO format or relative like 'now')",
)
def export(output: Optional[str], start: Optional[str], end: Optional[str]):
    """Export metrics data for external analysis."""
    collector = get_metrics_collector()
    
    # Parse time range
    end_time = time.time()
    start_time = end_time - 3600  # Default: last hour
    
    if start:
        if start.startswith("-"):
            # Relative time like "-1h", "-30m"
            value = start[1:-1]
            unit = start[-1]
            multiplier = {"h": 3600, "m": 60, "s": 1}.get(unit, 1)
            start_time = end_time - (float(value) * multiplier)
        else:
            # ISO format
            start_time = datetime.fromisoformat(start).timestamp()
    
    if end and end != "now":
        end_time = datetime.fromisoformat(end).timestamp()
    
    # Get metrics from backend
    metrics = collector.get_metrics(start_time=start_time, end_time=end_time)
    
    # Convert to serializable format
    data = {
        "export_time": datetime.now().isoformat(),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "metrics_count": len(metrics),
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "type": m.metric_type.value,
                "timestamp": m.timestamp,
                "tags": m.tags,
            }
            for m in metrics
        ],
    }
    
    if output:
        # Write to file
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]✓ Exported {len(metrics)} metrics to {output}[/green]")
    else:
        # Print to stdout
        click.echo(json.dumps(data, indent=2))


@metrics.command()
@click.confirmation_option(prompt="Clear all metrics data?")
def clear():
    """Clear all collected metrics data."""
    collector = get_metrics_collector()
    collector.clear()
    console.print("[green]✓ All metrics data cleared[/green]")


@metrics.command()
def backend_stats():
    """Show metrics backend statistics."""
    collector = get_metrics_collector()
    stats = collector.backend.get_stats()
    
    table = Table(title="Backend Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in stats.items():
        if isinstance(value, float):
            if "timestamp" in key and value:
                value = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            elif "mb" in key.lower():
                value = f"{value:.2f} MB"
            else:
                value = f"{value:.2f}"
        elif value is None:
            value = "N/A"
        else:
            value = str(value)
        
        table.add_row(key.replace("_", " ").title(), value)
    
    console.print(table)


@metrics.command()
def config():
    """Display current monitoring configuration."""
    table = Table(title="Monitoring Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    # Basic settings
    table.add_row("Enabled", str(MONITORING.get("enabled", True)))
    table.add_row("Backend", MONITORING.get("backend", "memory"))
    table.add_row("Buffer Size", str(MONITORING.get("buffer_size", 10000)))
    table.add_row("Flush Interval", f"{MONITORING.get('flush_interval', 10.0)}s")
    table.add_row("Sampling Rate", f"{MONITORING.get('sampling_rate', 1.0):.1%}")
    
    # Backend-specific settings
    backend_type = MONITORING.get("backend", "memory")
    if backend_type == "sqlite" and "sqlite" in MONITORING:
        table.add_row("", "")  # Separator
        table.add_row("[bold]SQLite Settings[/bold]", "")
        for key, value in MONITORING["sqlite"].items():
            table.add_row(f"  {key}", str(value))
    elif backend_type == "statsd" and "statsd" in MONITORING:
        table.add_row("", "")  # Separator
        table.add_row("[bold]StatsD Settings[/bold]", "")
        for key, value in MONITORING["statsd"].items():
            table.add_row(f"  {key}", str(value))
    
    # System monitoring settings
    if "systems" in MONITORING:
        table.add_row("", "")  # Separator
        table.add_row("[bold]System Monitoring[/bold]", "")
        for system, enabled in MONITORING["systems"].items():
            status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
            table.add_row(f"  {system}", status)
    
    console.print(table)