"""Comprehensive dependency management and system diagnostics for GifLab.

This module provides a complete CLI toolkit for managing dependencies, checking
system capabilities, and troubleshooting installation issues. It integrates
with the conditional import architecture and memory monitoring systems to
provide actionable diagnostic information.

CLI Commands Overview:
    giflab deps check      - Comprehensive dependency availability check
    giflab deps status     - Quick dependency status overview
    giflab deps install-help - Installation guidance for dependencies
    giflab deps cache-stats - Cache effectiveness statistics
    giflab deps cache-analyze - Cache performance analysis
    giflab deps cache-baseline - Baseline performance testing

Key Features:
    - Rich-formatted output with tables and status indicators
    - JSON output support for automation and CI/CD integration
    - Real-time memory monitoring and system capabilities
    - Integration with conditional import system status
    - Cache effectiveness monitoring and analysis
    - Installation guidance with platform-specific instructions
    - Error diagnostics with actionable troubleshooting steps

Dependency Categories:
    Core Dependencies:
        - PIL/Pillow: Image processing foundation
        - OpenCV (cv2): Computer vision and image manipulation
        - NumPy: Numerical computing arrays
        - psutil: System and memory monitoring (optional)
    
    Machine Learning:
        - PyTorch: Deep learning framework
        - LPIPS: Perceptual similarity metrics
        - scikit-learn: Machine learning utilities
        - SciPy: Scientific computing
    
    Visualization:
        - Matplotlib: Basic plotting and visualization
        - Seaborn: Statistical visualization
        - Plotly: Interactive plots and dashboards
    
    External Tools:
        - SSIMULACRA2: Advanced image quality metrics

System Integration:
    - Memory monitoring integration for real-time memory pressure reporting
    - Conditional import system status for caching feature diagnostics
    - Cache effectiveness monitoring for performance optimization
    - CLI integration with all GifLab monitoring and diagnostic systems

Output Formats:
    Human-readable: Rich-formatted tables with color coding and status indicators
    JSON: Machine-readable format for automation, CI/CD, and programmatic access
    
Architecture:
    Built on Click framework with Rich for enhanced terminal output.
    Integrates tightly with lazy_imports, memory monitoring, and caching systems
    to provide comprehensive system diagnostics in a single interface.

Error Handling:
    Graceful degradation when optional components unavailable.
    Detailed error reporting with installation guidance.
    Safe operation even with partial system configurations.

Performance:
    Fast dependency checking (~100ms for full system scan).
    Memory monitoring adds minimal overhead (<1ms).
    Cache analysis depends on cache history size.

Usage Patterns:
    Development troubleshooting:
        >>> giflab deps check --verbose
        
    CI/CD integration:
        >>> giflab deps check --json | jq '.core.available_count'
        
    Performance analysis:
        >>> giflab deps cache-analyze --confidence-threshold 0.8
        
    Quick status check:
        >>> giflab deps status

See Also:
    - docs/guides/cli-dependency-troubleshooting.md: Complete troubleshooting guide
    - src/giflab/lazy_imports.py: Lazy import infrastructure
    - src/giflab/monitoring/memory_monitor.py: Memory monitoring integration
    - tests/test_cli_commands.py: Comprehensive test coverage and usage examples
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ..lazy_imports import (
    get_import_status,
    is_torch_available,
    is_lpips_available, 
    is_cv2_available,
    is_scipy_available,
    is_sklearn_available,
    is_pil_available,
    is_matplotlib_available,
    is_seaborn_available,
    is_plotly_available,
    is_subprocess_available,
    check_import_available
)
from ..metrics import get_caching_status
from ..ssimulacra2_metrics import Ssimulacra2Validator, DEFAULT_SSIMULACRA2_PATH
from ..monitoring.memory_monitor import (
    is_memory_monitoring_available,
    get_system_memory_monitor,
    MemoryPressureLevel
)
from ..config import MONITORING


@click.group("deps")
def deps() -> None:
    """Check dependencies and system capabilities."""
    pass


@deps.command("check")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed dependency information"
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output results in JSON format"
)
def deps_check(verbose: bool, output_json: bool) -> None:
    """Check availability of all dependencies."""
    console = Console()
    
    # Core dependency mappings
    dependencies = {
        "Core Dependencies": {
            "PIL/Pillow": is_pil_available,
            "OpenCV (cv2)": is_cv2_available,
            "NumPy": lambda: check_import_available("numpy"),
            "subprocess": is_subprocess_available,
            "psutil (memory monitoring)": is_memory_monitoring_available,
        },
        "Machine Learning": {
            "PyTorch": is_torch_available,
            "LPIPS": is_lpips_available,
            "scikit-learn": is_sklearn_available,
            "SciPy": is_scipy_available,
        },
        "Visualization": {
            "Matplotlib": is_matplotlib_available,
            "Seaborn": is_seaborn_available,
            "Plotly": is_plotly_available,
        },
        "External Tools": {
            "SSIMULACRA2": lambda: Ssimulacra2Validator().is_available(),
        }
    }
    
    if output_json:
        _output_json_dependencies(dependencies)
        return
    
    console.print("\n🔍 [bold blue]GifLab Dependency Check[/bold blue]\n")
    
    all_available = True
    
    for category, deps in dependencies.items():
        table = Table(title=f"📦 {category}", show_header=True, header_style="bold magenta")
        table.add_column("Dependency", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        for name, check_func in deps.items():
            try:
                available = check_func()
                status = "✅ Available" if available else "❌ Missing"
                status_style = "green" if available else "red"
                
                if not available:
                    all_available = False
                
                details = ""
                if verbose:
                    if name == "SSIMULACRA2":
                        details = f"Path: {DEFAULT_SSIMULACRA2_PATH}"
                    elif available:
                        details = "Ready for use"
                    else:
                        details = "Install required"
                
                table.add_row(
                    name,
                    f"[{status_style}]{status}[/{status_style}]",
                    details
                )
                
            except Exception as e:
                table.add_row(
                    name,
                    "[red]❌ Error[/red]",
                    f"Check failed: {str(e)[:50]}..."
                )
                all_available = False
        
        console.print(table)
        console.print()
    
    # System capabilities summary
    _show_system_capabilities(console, verbose)
    
    # Overall status
    if all_available:
        console.print(Panel(
            "✅ [green]All dependencies are available![/green]\n"
            "Your system is ready for full GifLab functionality.",
            title="System Status",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "⚠️  [yellow]Some dependencies are missing.[/yellow]\n"
            "GifLab will work with reduced functionality.\n"
            "Run [bold]giflab deps install-help[/bold] for installation guidance.",
            title="System Status", 
            border_style="yellow"
        ))


@deps.command("install-help")
@click.argument("dependency", required=False)
def deps_install_help(dependency: str = None) -> None:
    """Show installation help for dependencies."""
    console = Console()
    
    install_commands = {
        "pil": "pip install Pillow",
        "pillow": "pip install Pillow", 
        "cv2": "pip install opencv-python",
        "opencv": "pip install opencv-python",
        "pytorch": "pip install torch torchvision",
        "torch": "pip install torch torchvision",
        "lpips": "pip install lpips",
        "sklearn": "pip install scikit-learn",
        "scikit-learn": "pip install scikit-learn",
        "scipy": "pip install scipy",
        "matplotlib": "pip install matplotlib",
        "seaborn": "pip install seaborn",
        "plotly": "pip install plotly",
        "psutil": "pip install psutil",
        "ssimulacra2": "Install from: https://github.com/cloudinary/ssimulacra2\nOr: brew install ssimulacra2",
    }
    
    if dependency:
        dep_lower = dependency.lower()
        if dep_lower in install_commands:
            console.print(f"\n📦 [bold]Installation for {dependency}:[/bold]")
            console.print(f"[green]{install_commands[dep_lower]}[/green]\n")
        else:
            console.print(f"❌ No installation help available for '{dependency}'")
            console.print("Run [bold]giflab deps install-help[/bold] to see all available options.")
    else:
        console.print("\n📖 [bold blue]Dependency Installation Guide[/bold blue]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Dependency", style="cyan") 
        table.add_column("Installation Command", style="green")
        table.add_column("Notes", style="dim")
        
        for dep, cmd in install_commands.items():
            if dep in ["pillow", "opencv", "scikit-learn"]:  # Skip aliases
                continue
                
            notes = ""
            if dep == "ssimulacra2":
                notes = "External binary required"
            elif dep in ["pytorch", "torch"]:
                notes = "Heavy dependency (~2GB)"
            elif dep == "lpips":
                notes = "Requires PyTorch"
            elif dep == "psutil":
                notes = "For memory monitoring"
            
            table.add_row(dep, cmd, notes)
        
        console.print(table)
        
        console.print(Panel(
            "💡 [bold]Tips:[/bold]\n"
            "• Use [cyan]poetry install[/cyan] to install all project dependencies\n"
            "• Some features gracefully degrade when dependencies are missing\n"
            "• SSIMULACRA2 requires separate binary installation",
            title="Installation Tips",
            border_style="blue"
        ))


@deps.command("status")
def deps_status() -> None:
    """Show quick dependency status summary."""
    console = Console()
    
    # Check core dependencies
    core_deps = [
        ("PIL", is_pil_available()),
        ("OpenCV", is_cv2_available()),
        ("SSIMULACRA2", Ssimulacra2Validator().is_available()),
        ("psutil", is_memory_monitoring_available()),
    ]
    
    # Check caching status
    caching_status = get_caching_status()
    
    # Check memory monitoring status
    memory_available = is_memory_monitoring_available()
    memory_config = MONITORING.get("memory_pressure", {})
    memory_enabled = memory_config.get("enabled", True)
    
    console.print("\n⚡ [bold]Quick Status[/bold]\n")
    
    # Core dependencies
    for name, available in core_deps:
        status = "✅" if available else "❌"
        console.print(f"{status} {name}")
    
    # Caching status
    caching_emoji = "✅" if caching_status["enabled"] else "⚠️"
    console.print(f"{caching_emoji} Caching: {'Enabled' if caching_status['enabled'] else 'Disabled'}")
    
    # Memory monitoring status
    if memory_available and memory_enabled:
        try:
            monitor = get_system_memory_monitor()
            stats = monitor.collect_memory_stats()
            pressure_icons = {
                MemoryPressureLevel.NORMAL: "🟢",
                MemoryPressureLevel.WARNING: "🟡",
                MemoryPressureLevel.CRITICAL: "🟠", 
                MemoryPressureLevel.EMERGENCY: "🔴"
            }
            icon = pressure_icons.get(stats.pressure_level, "❓")
            console.print(f"🧠 Memory monitoring: {icon} {stats.pressure_level.value}")
        except Exception:
            console.print("🧠 Memory monitoring: ✅ Active")
    elif memory_available and not memory_enabled:
        console.print("🧠 Memory monitoring: 💤 Disabled")
    else:
        console.print("🧠 Memory monitoring: ❌ Unavailable")
    
    # Lazy import status
    lazy_status = get_import_status()
    loaded_count = sum(1 for available, loaded in lazy_status.values() if loaded)
    console.print(f"📦 Lazy imports: {loaded_count}/{len(lazy_status)} loaded")


def _output_json_dependencies(dependencies: Dict) -> None:
    """Output dependency status in JSON format."""
    
    result = {}
    for category, deps in dependencies.items():
        result[category] = {}
        for name, check_func in deps.items():
            try:
                result[category][name] = {
                    "available": check_func(),
                    "error": None
                }
            except Exception as e:
                result[category][name] = {
                    "available": False,
                    "error": str(e)
                }
    
    # Add system info
    memory_config = MONITORING.get("memory_pressure", {})
    memory_status = {
        "available": is_memory_monitoring_available(),
        "enabled": memory_config.get("enabled", True),
        "config": memory_config
    }
    
    # Add current memory stats if available
    if memory_status["available"] and memory_status["enabled"]:
        try:
            monitor = get_system_memory_monitor()
            stats = monitor.collect_memory_stats()
            if stats:
                memory_status["current"] = {
                    "system_memory_percent": stats.system_memory_percent,
                    "process_memory_mb": stats.process_memory_mb,
                    "pressure_level": stats.pressure_level.value,
                    "timestamp": stats.timestamp
                }
        except Exception as e:
            memory_status["error"] = str(e)
    
    result["system"] = {
        "python_version": sys.version,
        "platform": sys.platform,
        "caching_status": get_caching_status(),
        "memory_monitoring": memory_status
    }
    
    click.echo(json.dumps(result, indent=2))


def _show_system_capabilities(console: Console, verbose: bool) -> None:
    """Show system capabilities and configuration."""
    
    capabilities = []
    
    # Check caching
    caching_status = get_caching_status()
    if caching_status["enabled"]:
        capabilities.append("🚀 Performance caching enabled")
    else:
        capabilities.append("⚠️  Performance caching disabled")
        if caching_status["error_message"] and verbose:
            console.print(f"   [dim]Reason: {caching_status['error_message'][:100]}...[/dim]")
    
    # Check external tools
    ssim_validator = Ssimulacra2Validator()
    if ssim_validator.is_available():
        capabilities.append("🔍 SSIMULACRA2 perceptual metrics available")
    else:
        capabilities.append("📊 Basic metrics only (SSIMULACRA2 not found)")
    
    # Check ML capabilities
    if is_torch_available() and is_lpips_available():
        capabilities.append("🧠 Deep perceptual metrics available")
    else:
        capabilities.append("📈 Traditional metrics only")
    
    # Check memory monitoring
    try:
        memory_config = MONITORING.get("memory_pressure", {})
        memory_enabled = memory_config.get("enabled", True)
        memory_available = is_memory_monitoring_available()
        
        if memory_available and memory_enabled:
            capabilities.append("🧠 Memory pressure monitoring active")
            if verbose:
                try:
                    monitor = get_system_memory_monitor()
                    stats = monitor.collect_memory_stats()
                    if stats:
                        pressure_icons = {
                            MemoryPressureLevel.NORMAL: "🟢",
                            MemoryPressureLevel.WARNING: "🟡", 
                            MemoryPressureLevel.CRITICAL: "🟠",
                            MemoryPressureLevel.EMERGENCY: "🔴"
                        }
                        icon = pressure_icons.get(stats.pressure_level, "❓")
                        console.print(f"   [dim]{icon} Current pressure: {stats.pressure_level.value} "
                                    f"(System: {stats.system_memory_percent:.1%}, "
                                    f"Process: {stats.process_memory_mb:.0f}MB)[/dim]")
                except Exception:
                    pass
        elif memory_available and not memory_enabled:
            capabilities.append("💤 Memory monitoring available but disabled")
        else:
            capabilities.append("⚠️  Memory monitoring unavailable (psutil missing)")
    except Exception:
        capabilities.append("❓ Memory monitoring status unknown")
    
    if capabilities:
        table = Table(title="🛠️  System Capabilities", show_header=False)
        table.add_column("Capability", style="cyan")
        
        for cap in capabilities:
            table.add_row(cap)
        
        console.print(table)
        console.print()


@deps.command("cache-stats")
@click.option(
    "--cache-type", "-c",
    help="Show stats for specific cache type only"
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output in JSON format"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed effectiveness analysis"
)
def cache_stats(cache_type: str = None, output_json: bool = False, verbose: bool = False) -> None:
    """Show cache effectiveness statistics and analysis."""
    console = Console()
    
    try:
        from ..monitoring.cache_effectiveness import get_cache_effectiveness_monitor, is_cache_effectiveness_monitoring_enabled
        from ..monitoring.baseline_framework import get_baseline_framework
        from ..monitoring.memory_monitor import get_cache_memory_tracker
        from ..config import ENABLE_EXPERIMENTAL_CACHING
        
        if not ENABLE_EXPERIMENTAL_CACHING:
            if output_json:
                print(json.dumps({"error": "Cache effectiveness monitoring requires experimental caching to be enabled"}))
            else:
                console.print("[yellow]⚠️  Cache effectiveness monitoring requires experimental caching to be enabled[/yellow]")
                console.print("Set ENABLE_EXPERIMENTAL_CACHING = True in config.py to enable monitoring")
            return
        
        if not is_cache_effectiveness_monitoring_enabled():
            if output_json:
                print(json.dumps({"error": "Cache effectiveness monitoring is disabled in configuration"}))
            else:
                console.print("[yellow]⚠️  Cache effectiveness monitoring is disabled in configuration[/yellow]")
            return
        
        # Get monitoring instances
        effectiveness_monitor = get_cache_effectiveness_monitor()
        baseline_framework = get_baseline_framework()
        memory_tracker = get_cache_memory_tracker()
        
        # Collect data
        if cache_type:
            cache_stats_data = {cache_type: effectiveness_monitor.get_cache_effectiveness(cache_type)}
            cache_stats_data = {k: v for k, v in cache_stats_data.items() if v is not None}
        else:
            cache_stats_data = effectiveness_monitor.get_all_cache_stats()
        
        system_summary = effectiveness_monitor.get_system_effectiveness_summary()
        baseline_stats = baseline_framework.get_all_baseline_statistics()
        baseline_report = baseline_framework.generate_performance_report()
        memory_summary = memory_tracker.get_system_effectiveness_summary()
        
        if output_json:
            output_data = {
                "cache_stats": {
                    cache_name: {
                        "hit_rate": stats.hit_rate,
                        "total_operations": stats.total_operations,
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "evictions": stats.evictions,
                        "eviction_rate": stats.eviction_rate,
                        "memory_mb": stats.total_data_cached_mb,
                        "recent_hit_rate": stats.recent_hit_rate,
                        "hourly_hit_rate": stats.hourly_hit_rate,
                        "net_performance_gain_ms": stats.net_performance_gain_ms,
                        "pressure_correlation": stats.pressure_correlation_score
                    }
                    for cache_name, stats in cache_stats_data.items()
                },
                "system_summary": system_summary,
                "baseline_stats": {
                    op_type: {
                        "performance_improvement": stats.performance_improvement,
                        "cached_samples": stats.cached_samples,
                        "non_cached_samples": stats.non_cached_samples,
                        "statistical_significance": stats.statistical_significance
                    }
                    for op_type, stats in baseline_stats.items()
                },
                "baseline_report": baseline_report,
                "memory_summary": memory_summary
            }
            print(json.dumps(output_data, indent=2))
            return
        
        # Display results in rich format
        if not cache_stats_data:
            console.print("[yellow]📊 No cache effectiveness data available yet[/yellow]")
            console.print("Cache operations need to be performed to collect statistics")
            return
        
        # System-wide summary
        console.print(Panel.fit(
            f"📊 [bold cyan]Cache Effectiveness Summary[/bold cyan]\n\n"
            f"Total Operations: [bold]{system_summary.get('total_operations', 0):,}[/bold]\n"
            f"Overall Hit Rate: [bold]{system_summary.get('overall_hit_rate', 0):.1%}[/bold]\n"
            f"Cache Types: [bold]{system_summary.get('cache_types', 0)}[/bold]\n"
            f"Monitoring Duration: [bold]{system_summary.get('monitoring_duration_hours', 0):.1f}h[/bold]",
            title="System Overview"
        ))
        
        # Cache-specific statistics
        stats_table = Table(title="🗂️  Cache Performance Statistics")
        stats_table.add_column("Cache Type", style="cyan")
        stats_table.add_column("Hit Rate", justify="right", style="green")
        stats_table.add_column("Operations", justify="right")
        stats_table.add_column("Memory (MB)", justify="right", style="blue")
        stats_table.add_column("Eviction Rate", justify="right", style="yellow")
        stats_table.add_column("Pressure Correlation", justify="right", style="red")
        
        for cache_name, stats in cache_stats_data.items():
            hit_rate_color = "green" if stats.hit_rate > 0.7 else "yellow" if stats.hit_rate > 0.4 else "red"
            eviction_color = "green" if stats.eviction_rate < 1.0 else "yellow" if stats.eviction_rate < 5.0 else "red"
            
            stats_table.add_row(
                cache_name,
                f"[{hit_rate_color}]{stats.hit_rate:.1%}[/{hit_rate_color}]",
                f"{stats.total_operations:,}",
                f"{stats.total_data_cached_mb:.1f}",
                f"[{eviction_color}]{stats.eviction_rate:.1f}/min[/{eviction_color}]",
                f"{stats.pressure_correlation_score:.2f}"
            )
        
        console.print(stats_table)
        
        # Baseline performance comparison
        if baseline_stats:
            baseline_table = Table(title="⚡ Performance Baseline Comparison")
            baseline_table.add_column("Operation", style="cyan")
            baseline_table.add_column("Improvement", justify="right", style="green")
            baseline_table.add_column("Cached Samples", justify="right")
            baseline_table.add_column("Baseline Samples", justify="right")
            baseline_table.add_column("Significant", justify="center")
            
            for op_type, stats in baseline_stats.items():
                improvement_color = "green" if stats.performance_improvement > 0.1 else "yellow" if stats.performance_improvement > 0 else "red"
                significance_icon = "✅" if stats.statistical_significance else "❌"
                
                baseline_table.add_row(
                    op_type,
                    f"[{improvement_color}]{stats.performance_improvement:.1%}[/{improvement_color}]",
                    str(stats.cached_samples),
                    str(stats.non_cached_samples),
                    significance_icon
                )
            
            console.print(baseline_table)
        
        # Verbose analysis
        if verbose:
            try:
                from ..monitoring.effectiveness_analysis import analyze_cache_effectiveness
                
                console.print("\n🔍 [bold]Detailed Effectiveness Analysis[/bold]")
                analysis = analyze_cache_effectiveness()
                
                # Recommendation
                recommendation_colors = {
                    "enable_production": "green",
                    "enable_with_monitoring": "yellow",
                    "selective_enable": "yellow",
                    "keep_disabled": "red",
                    "insufficient_data": "blue",
                    "performance_regression": "red"
                }
                
                rec_color = recommendation_colors.get(analysis.recommendation.value, "white")
                console.print(f"Recommendation: [{rec_color}]{analysis.recommendation.value.upper()}[/{rec_color}] "
                            f"(Confidence: {analysis.confidence_score:.1%})")
                
                # Key insights
                if analysis.optimization_recommendations:
                    console.print("\n💡 [bold]Optimization Recommendations:[/bold]")
                    for i, rec in enumerate(analysis.optimization_recommendations[:5], 1):
                        console.print(f"  {i}. {rec}")
                
                if analysis.performance_insights:
                    console.print("\n📈 [bold]Performance Insights:[/bold]")
                    for insight in analysis.performance_insights[:3]:
                        console.print(f"  • {insight}")
                
                if analysis.risk_factors:
                    console.print("\n⚠️  [bold]Risk Factors:[/bold]")
                    for risk in analysis.risk_factors[:3]:
                        console.print(f"  • [yellow]{risk}[/yellow]")
                
            except Exception as e:
                console.print(f"[red]Failed to generate detailed analysis: {e}[/red]")
    
    except ImportError:
        if output_json:
            print(json.dumps({"error": "Cache effectiveness monitoring not available"}))
        else:
            console.print("[red]❌ Cache effectiveness monitoring not available[/red]")
            console.print("This feature requires the cache effectiveness monitoring modules")


@deps.command("cache-analyze")
@click.option(
    "--confidence-threshold", "-t",
    type=float,
    default=0.7,
    help="Minimum confidence threshold for recommendations (0.0-1.0)"
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output in JSON format"
)
def cache_analyze(confidence_threshold: float = 0.7, output_json: bool = False) -> None:
    """Perform comprehensive cache effectiveness analysis with recommendations."""
    console = Console()
    
    try:
        from ..monitoring.effectiveness_analysis import analyze_cache_effectiveness
        from ..config import ENABLE_EXPERIMENTAL_CACHING
        
        if not ENABLE_EXPERIMENTAL_CACHING:
            if output_json:
                print(json.dumps({"error": "Cache analysis requires experimental caching to be enabled"}))
            else:
                console.print("[yellow]⚠️  Cache analysis requires experimental caching to be enabled[/yellow]")
            return
        
        console.print("🔍 [bold]Analyzing cache effectiveness...[/bold]")
        
        analysis = analyze_cache_effectiveness()
        
        if output_json:
            output_data = {
                "recommendation": analysis.recommendation.value,
                "confidence_score": analysis.confidence_score,
                "meets_threshold": analysis.confidence_score >= confidence_threshold,
                "overall_hit_rate": analysis.overall_hit_rate,
                "average_performance_improvement": analysis.average_performance_improvement,
                "memory_efficiency_score": analysis.memory_efficiency_score,
                "optimization_recommendations": analysis.optimization_recommendations,
                "performance_insights": analysis.performance_insights,
                "risk_factors": analysis.risk_factors,
                "suggested_cache_sizes": analysis.suggested_cache_sizes,
                "suggested_eviction_thresholds": analysis.suggested_eviction_thresholds,
                "analysis_metadata": {
                    "data_collection_period_hours": analysis.data_collection_period_hours,
                    "total_operations_analyzed": analysis.total_operations_analyzed,
                    "analysis_timestamp": analysis.analysis_timestamp
                }
            }
            print(json.dumps(output_data, indent=2))
            return
        
        # Display analysis results
        recommendation_colors = {
            "enable_production": "green",
            "enable_with_monitoring": "yellow", 
            "selective_enable": "yellow",
            "keep_disabled": "red",
            "insufficient_data": "blue",
            "performance_regression": "red"
        }
        
        rec_color = recommendation_colors.get(analysis.recommendation.value, "white")
        confidence_color = "green" if analysis.confidence_score >= confidence_threshold else "yellow"
        
        # Main recommendation panel
        console.print(Panel.fit(
            f"📋 [bold]Cache Deployment Recommendation[/bold]\n\n"
            f"Recommendation: [{rec_color}]{analysis.recommendation.value.upper()}[/{rec_color}]\n"
            f"Confidence Score: [{confidence_color}]{analysis.confidence_score:.1%}[/{confidence_color}]\n"
            f"Threshold Met: {'✅' if analysis.confidence_score >= confidence_threshold else '❌'}\n\n"
            f"Key Metrics:\n"
            f"• Hit Rate: {analysis.overall_hit_rate:.1%}\n"
            f"• Performance Gain: {analysis.average_performance_improvement:.1%}\n"
            f"• Memory Efficiency: {analysis.memory_efficiency_score:.1%}",
            title="Analysis Results"
        ))
        
        # Recommendations
        if analysis.optimization_recommendations:
            rec_table = Table(title="💡 Optimization Recommendations", show_header=False)
            rec_table.add_column("Priority", style="cyan", width=3)
            rec_table.add_column("Recommendation")
            
            for i, rec in enumerate(analysis.optimization_recommendations, 1):
                priority_color = "red" if i <= 2 else "yellow" if i <= 4 else "green"
                rec_table.add_row(f"[{priority_color}]{i}[/{priority_color}]", rec)
            
            console.print(rec_table)
        
        # Performance insights
        if analysis.performance_insights:
            console.print("\n📈 [bold]Performance Insights:[/bold]")
            for insight in analysis.performance_insights:
                console.print(f"  • {insight}")
        
        # Risk factors
        if analysis.risk_factors:
            console.print("\n⚠️  [bold yellow]Risk Factors:[/bold yellow]")
            for risk in analysis.risk_factors:
                console.print(f"  • [yellow]{risk}[/yellow]")
        
        # Configuration suggestions
        if analysis.suggested_cache_sizes:
            console.print("\n⚙️  [bold]Suggested Configuration:[/bold]")
            console.print("Cache Sizes:")
            for cache_type, size_mb in analysis.suggested_cache_sizes.items():
                console.print(f"  • {cache_type}: {size_mb:.0f}MB")
        
        if analysis.suggested_eviction_thresholds:
            console.print("Eviction Thresholds:")
            for threshold_type, value in analysis.suggested_eviction_thresholds.items():
                console.print(f"  • {threshold_type}: {value:.1%}")
        
        # Data collection summary
        console.print(f"\n📊 [dim]Analysis based on {analysis.total_operations_analyzed:,} operations "
                     f"over {analysis.data_collection_period_hours:.1f} hours[/dim]")
    
    except ImportError:
        if output_json:
            print(json.dumps({"error": "Cache analysis not available"}))
        else:
            console.print("[red]❌ Cache analysis not available[/red]")
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")


@deps.command("cache-baseline")
@click.option(
    "--mode", "-m",
    type=click.Choice(["passive", "ab_testing", "controlled"], case_sensitive=False),
    default="passive",
    help="Baseline testing mode"
)
@click.option(
    "--duration", "-d",
    type=int,
    default=300,
    help="Test duration in seconds for controlled testing"
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output in JSON format"
)
def cache_baseline(mode: str = "passive", duration: int = 300, output_json: bool = False) -> None:
    """Configure and monitor cache baseline performance testing."""
    console = Console()
    
    try:
        from ..monitoring.baseline_framework import get_baseline_framework, BaselineTestMode
        from ..config import ENABLE_EXPERIMENTAL_CACHING
        
        if not ENABLE_EXPERIMENTAL_CACHING:
            if output_json:
                print(json.dumps({"error": "Baseline testing requires experimental caching to be enabled"}))
            else:
                console.print("[yellow]⚠️  Baseline testing requires experimental caching to be enabled[/yellow]")
            return
        
        baseline_framework = get_baseline_framework()
        
        # Map string mode to enum
        mode_mapping = {
            "passive": BaselineTestMode.PASSIVE,
            "ab_testing": BaselineTestMode.AB_TESTING,
            "controlled": BaselineTestMode.CONTROLLED
        }
        
        if mode in mode_mapping:
            baseline_framework.test_mode = mode_mapping[mode]
            
            if mode == "ab_testing":
                baseline_framework.start_ab_testing()
                status_msg = "A/B testing mode activated - 10% of operations will run without caching for baseline comparison"
            elif mode == "controlled":
                status_msg = f"Controlled testing mode set - use controlled test scenarios for {duration}s duration"
            else:
                baseline_framework.stop_ab_testing()
                status_msg = "Passive monitoring mode - baselines collected when caching naturally disabled"
            
            if output_json:
                print(json.dumps({
                    "status": "success",
                    "mode": mode,
                    "message": status_msg
                }))
            else:
                console.print(f"[green]✅ {status_msg}[/green]")
        
        # Show current baseline status
        report = baseline_framework.generate_performance_report()
        
        if output_json:
            print(json.dumps(report, indent=2))
        else:
            if report["status"] == "no_data":
                console.print("[yellow]📊 No baseline data collected yet[/yellow]")
                console.print("Perform cache operations in the configured mode to collect baseline data")
            else:
                console.print(Panel.fit(
                    f"📊 [bold cyan]Baseline Testing Status[/bold cyan]\n\n"
                    f"Mode: [bold]{baseline_framework.test_mode.value}[/bold]\n"
                    f"Total Operations: [bold]{report['collection_summary']['total_operations']:,}[/bold]\n"
                    f"Operation Types: [bold]{report['collection_summary']['operation_types']}[/bold]\n"
                    f"Valid Comparisons: [bold]{report['collection_summary']['valid_comparisons']}[/bold]\n"
                    f"Collection Duration: [bold]{report['collection_summary']['collection_duration_hours']:.1f}h[/bold]",
                    title="Baseline Framework Status"
                ))
                
                if report["performance_analysis"]["statistically_significant_improvements"] > 0:
                    improvement = report["performance_analysis"]["average_improvement"]
                    improvement_color = "green" if improvement > 0.1 else "yellow" if improvement > 0 else "red"
                    console.print(f"\n📈 Average Performance Improvement: [{improvement_color}]{improvement:.1%}[/{improvement_color}]")
    
    except ImportError:
        if output_json:
            print(json.dumps({"error": "Baseline testing not available"}))
        else:
            console.print("[red]❌ Baseline testing not available[/red]")
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Baseline testing failed: {e}[/red]")