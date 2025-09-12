# Phase 1-4 Integration Guide: System Interactions and Patterns

This document provides comprehensive guidance on how the Phase 1-4 architectural improvements integrate with each other and with existing GifLab infrastructure.

## Overview

Phases 1-4 introduced multiple interconnected systems that work together to provide a robust, safe, and maintainable architecture:

- **Phase 1**: Build stability and type safety foundation
- **Phase 2**: Conditional import architecture and CLI enhancements  
- **Phase 3**: Memory monitoring infrastructure
- **Phase 4**: Comprehensive testing and validation

These phases build upon each other to create a cohesive system that maintains backward compatibility while enabling safe experimentation with new features.

---

## System Integration Architecture

### High-Level System Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GifLab Integration Ecosystem                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Feature Flag Layer                         │   │
│  │                                                              │   │
│  │  ENABLE_EXPERIMENTAL_CACHING ────┬──── Memory Monitoring    │   │
│  │  MONITORING[memory_pressure]      │     Configuration       │   │
│  │  LOG_VALIDATION_FAILURES ─────────┼──── Error Handling      │   │
│  │  FRAME_CACHE[enabled] ────────────┘     Strategies          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                 Conditional Import System                    │   │
│  │                                                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐    │   │
│  │  │ Import Safety   │  │ Fallback        │  │ Error       │    │   │
│  │  │                 │  │ Implementations │  │ Handling    │    │   │
│  │  │ • try/catch     │  │                 │  │             │    │   │
│  │  │ • Feature flags │  │ • resize_frame_ │  │ • Detailed  │    │   │
│  │  │ • Graceful      │  │   fallback      │  │   messages  │    │   │
│  │  │   degradation   │  │ • Basic         │  │ • User      │    │   │
│  │  │                 │  │   operations    │  │   guidance  │    │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Memory Monitoring Infrastructure                │   │
│  │                                                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐    │   │
│  │  │ Pressure        │  │ Automatic       │  │ Integration │    │   │
│  │  │ Detection       │  │ Eviction        │  │ Layer       │    │   │
│  │  │                 │  │                 │  │             │    │   │
│  │  │ • Real-time     │  │ • Cache         │  │ • CLI Status│    │   │
│  │  │   monitoring    │  │   coordination  │  │ • Alerts    │    │   │
│  │  │ • Threshold     │  │ • Conservative  │  │ • Metrics   │    │   │
│  │  │   management    │  │   policies      │  │   collection│    │   │
│  │  │                 │  │                 │  │             │    │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                CLI Dependency Management                     │   │
│  │                                                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐    │   │
│  │  │ Comprehensive   │  │ Rich UI         │  │ Automation  │    │   │
│  │  │ Checking        │  │ Reporting       │  │ Support     │    │   │
│  │  │                 │  │                 │  │             │    │   │
│  │  │ • Core deps     │  │ • Tables        │  │ • JSON      │    │   │
│  │  │ • Optional deps │  │ • Status icons  │  │   output    │    │   │
│  │  │ • External tools│  │ • Installation  │  │ • CI/CD     │    │   │
│  │  │ • System status │  │   guidance      │  │   integration│   │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Existing GifLab Infrastructure                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Metrics System • Configuration Manager • Cache System • CLI        │
│ Alert Manager • Testing Framework • Validation System • Monitoring │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Integration Patterns

### 1. Feature Flag Coordination

All Phase 1-4 systems coordinate through a unified feature flag system that ensures consistent behavior across components.

#### Feature Flag Hierarchy

```python
# Primary feature flags (config.py)
ENABLE_EXPERIMENTAL_CACHING = False        # Controls conditional imports
MONITORING = {
    "memory_pressure": {"enabled": True},   # Controls memory monitoring
    "dependency_checking": {"enabled": True} # Controls CLI dependency features
}

# Derived flags (computed at runtime)
CACHING_ENABLED = ENABLE_EXPERIMENTAL_CACHING and imports_successful
MEMORY_MONITORING_ACTIVE = MONITORING["memory_pressure"]["enabled"] and psutil_available
CLI_ENHANCED_MODE = MONITORING["dependency_checking"]["enabled"] and rich_available
```

#### Cross-System Feature Detection

```python
def get_system_capabilities() -> dict:
    """Get comprehensive system capability report."""
    from giflab.config import ENABLE_EXPERIMENTAL_CACHING, MONITORING
    from giflab.metrics import CACHING_ENABLED
    from giflab.lazy_imports import is_rich_available
    
    capabilities = {
        # Core feature flags
        "feature_flags": {
            "experimental_caching": ENABLE_EXPERIMENTAL_CACHING,
            "memory_monitoring": MONITORING["memory_pressure"]["enabled"],
            "dependency_checking": MONITORING["dependency_checking"]["enabled"]
        },
        
        # Runtime system status
        "runtime_status": {
            "caching_active": CACHING_ENABLED,
            "memory_monitoring_active": _is_memory_monitoring_active(),
            "cli_enhanced_active": is_rich_available()
        },
        
        # Integration status
        "integration_status": {
            "metrics_system": _check_metrics_integration(),
            "alert_system": _check_alert_integration(), 
            "cli_system": _check_cli_integration()
        }
    }
    
    return capabilities
```

### 2. Conditional Import Integration with Memory Monitoring

Memory monitoring integrates with conditional imports to provide safe resource management for experimental features.

#### Memory-Aware Feature Activation

```python
class MemoryAwareFeatureManager:
    """Manages feature activation based on memory availability."""
    
    def __init__(self):
        self._memory_monitor = None
        if MONITORING["memory_pressure"]["enabled"]:
            try:
                from giflab.monitoring.memory_monitor import SystemMemoryMonitor
                self._memory_monitor = SystemMemoryMonitor()
            except ImportError:
                logger.warning("Memory monitoring unavailable for feature management")
    
    def should_enable_caching(self) -> bool:
        """Determine if caching should be enabled based on memory."""
        if not ENABLE_EXPERIMENTAL_CACHING:
            return False
        
        if not self._memory_monitor:
            return True  # No memory info, allow caching
        
        stats = self._memory_monitor.get_memory_stats()
        
        # Disable caching if memory pressure is high
        if stats.memory_percent > 0.85:  # 85% system memory usage
            logger.warning("Disabling caching due to high memory pressure")
            return False
        
        return True
    
    def adaptive_cache_limits(self) -> dict:
        """Calculate cache limits based on available memory."""
        default_limits = {
            "memory_limit_mb": 500,
            "disk_limit_gb": 2
        }
        
        if not self._memory_monitor:
            return default_limits
        
        stats = self._memory_monitor.get_memory_stats()
        available_gb = stats.available_memory / (1024**3)
        
        # Adaptive limits based on available memory
        if available_gb < 2.0:  # Less than 2GB available
            return {
                "memory_limit_mb": 100,  # Very conservative
                "disk_limit_gb": 1
            }
        elif available_gb < 4.0:  # 2-4GB available
            return {
                "memory_limit_mb": 250,  # Conservative
                "disk_limit_gb": 1.5
            }
        else:  # 4GB+ available
            return {
                "memory_limit_mb": min(int(available_gb * 200), 1000),  # Up to 1GB
                "disk_limit_gb": min(available_gb / 2, 5)  # Up to 5GB
            }

# Integration usage
feature_manager = MemoryAwareFeatureManager()

# Use in conditional imports
if feature_manager.should_enable_caching():
    # Proceed with caching imports
    cache_limits = feature_manager.adaptive_cache_limits()
    # Apply limits to cache configuration
```

### 3. CLI Integration with System Status

The CLI system provides a unified interface for monitoring and troubleshooting all Phase 1-4 systems.

#### Unified Status Reporting

```python
def generate_unified_status_report() -> dict:
    """Generate comprehensive status report across all systems."""
    
    report = {
        "timestamp": time.time(),
        "overall_status": "unknown",
        "systems": {}
    }
    
    # Phase 1: Build stability status
    report["systems"]["build_stability"] = {
        "imports_working": _test_core_imports(),
        "type_errors_count": _count_mypy_errors(),
        "test_pass_rate": _get_test_pass_rate()
    }
    
    # Phase 2: Conditional imports status  
    from giflab.metrics import CACHING_ENABLED, get_frame_cache
    report["systems"]["conditional_imports"] = {
        "feature_flag_enabled": ENABLE_EXPERIMENTAL_CACHING,
        "imports_successful": CACHING_ENABLED,
        "functions_available": {
            "get_frame_cache": get_frame_cache is not None,
            "resize_frame_cached": resize_frame_cached is not None
        },
        "fallback_mode": not CACHING_ENABLED and ENABLE_EXPERIMENTAL_CACHING
    }
    
    # Phase 3: Memory monitoring status
    memory_status = {"enabled": False, "active": False}
    if MONITORING["memory_pressure"]["enabled"]:
        try:
            from giflab.monitoring.memory_monitor import SystemMemoryMonitor
            monitor = SystemMemoryMonitor()
            stats = monitor.get_memory_stats()
            memory_status = {
                "enabled": True,
                "active": True,
                "usage_percent": stats.memory_percent,
                "available_gb": stats.available_memory / (1024**3),
                "pressure_level": _determine_pressure_level(stats.memory_percent)
            }
        except Exception as e:
            memory_status = {"enabled": True, "active": False, "error": str(e)}
    
    report["systems"]["memory_monitoring"] = memory_status
    
    # Phase 4: Testing and validation status
    report["systems"]["testing_validation"] = {
        "total_tests": _get_total_test_count(),
        "passing_tests": _get_passing_test_count(),
        "test_coverage_percent": _get_test_coverage(),
        "validation_systems_active": _check_validation_systems()
    }
    
    # Overall status determination
    critical_issues = []
    warnings = []
    
    # Check each system for issues
    if not report["systems"]["build_stability"]["imports_working"]:
        critical_issues.append("Core imports failing")
    
    if (report["systems"]["conditional_imports"]["feature_flag_enabled"] and 
        report["systems"]["conditional_imports"]["fallback_mode"]):
        warnings.append("Caching in fallback mode")
    
    if (report["systems"]["memory_monitoring"]["enabled"] and 
        not report["systems"]["memory_monitoring"]["active"]):
        warnings.append("Memory monitoring configured but not active")
    
    # Determine overall status
    if critical_issues:
        report["overall_status"] = "CRITICAL"
        report["critical_issues"] = critical_issues
    elif warnings:
        report["overall_status"] = "WARNINGS" 
        report["warnings"] = warnings
    else:
        report["overall_status"] = "HEALTHY"
    
    return report
```

#### CLI Command Integration

```python
@deps.command("system-status")
@click.option("--detailed", is_flag=True, help="Show detailed system breakdown")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
def system_status(detailed: bool, output_json: bool):
    """Show comprehensive system status across all Phase 1-4 components."""
    
    try:
        report = generate_unified_status_report()
        
        if output_json:
            import json
            console.print(json.dumps(report, indent=2))
            return
        
        # Rich formatted output
        status_colors = {
            "HEALTHY": "green",
            "WARNINGS": "yellow", 
            "CRITICAL": "red"
        }
        
        status_icons = {
            "HEALTHY": "✅",
            "WARNINGS": "⚠️",
            "CRITICAL": "❌"
        }
        
        overall_status = report["overall_status"]
        icon = status_icons.get(overall_status, "❓")
        color = status_colors.get(overall_status, "white")
        
        console.print(f"\n{icon} [bold {color}]System Status: {overall_status}[/bold {color}]")
        
        # Show critical issues first
        if "critical_issues" in report:
            console.print(f"\n[bold red]🚨 Critical Issues:[/bold red]")
            for issue in report["critical_issues"]:
                console.print(f"  • {issue}")
        
        # Show warnings
        if "warnings" in report:
            console.print(f"\n[bold yellow]⚠️ Warnings:[/bold yellow]")
            for warning in report["warnings"]:
                console.print(f"  • {warning}")
        
        if detailed:
            _show_detailed_system_status(report)
        else:
            _show_summary_system_status(report)
            
    except Exception as e:
        console.print(f"❌ Error generating system status: {e}")

def _show_summary_system_status(report: dict):
    """Show summary view of system status."""
    systems = report["systems"]
    
    console.print(f"\n📊 System Summary:")
    
    # Build stability
    build = systems["build_stability"]
    build_icon = "✅" if build["imports_working"] else "❌"
    console.print(f"  {build_icon} Build Stability: Imports working, {build['test_pass_rate']:.1%} tests passing")
    
    # Conditional imports
    imports = systems["conditional_imports"] 
    if imports["feature_flag_enabled"]:
        import_icon = "✅" if imports["imports_successful"] else "⚠️"
        mode = "Active" if imports["imports_successful"] else "Fallback"
        console.print(f"  {import_icon} Conditional Imports: {mode} mode")
    else:
        console.print(f"  ℹ️ Conditional Imports: Disabled (safe default)")
    
    # Memory monitoring
    memory = systems["memory_monitoring"]
    if memory["enabled"] and memory["active"]:
        pressure = memory.get("pressure_level", "unknown")
        usage = memory.get("usage_percent", 0) * 100
        memory_icon = "🟢" if usage < 70 else ("🟡" if usage < 85 else "🔴")
        console.print(f"  {memory_icon} Memory Monitoring: {usage:.1f}% usage ({pressure})")
    elif memory["enabled"]:
        console.print(f"  ⚠️ Memory Monitoring: Configured but inactive")
    else:
        console.print(f"  ➖ Memory Monitoring: Disabled")
    
    # Testing validation
    testing = systems["testing_validation"]
    test_icon = "✅" if testing["passing_tests"] == testing["total_tests"] else "⚠️"
    console.print(f"  {test_icon} Testing: {testing['passing_tests']}/{testing['total_tests']} tests passing")

def _show_detailed_system_status(report: dict):
    """Show detailed breakdown of each system."""
    systems = report["systems"]
    
    # Build Stability Details
    console.print(f"\n🏗️ [bold]Build Stability (Phase 1)[/bold]")
    build = systems["build_stability"]
    
    table = Table(title="Build Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="magenta")
    
    table.add_row("Core Imports", 
                  "✅ PASS" if build["imports_working"] else "❌ FAIL",
                  "All modules importable")
    table.add_row("Type Safety", 
                  f"⚠️ {build['type_errors_count']} errors" if build['type_errors_count'] > 0 else "✅ CLEAN",
                  f"MyPy analysis")
    table.add_row("Test Suite",
                  f"✅ {build['test_pass_rate']:.1%}" if build['test_pass_rate'] > 0.95 else f"⚠️ {build['test_pass_rate']:.1%}",
                  f"Pass rate")
    
    console.print(table)
    
    # Conditional Imports Details  
    console.print(f"\n🔄 [bold]Conditional Imports (Phase 2)[/bold]")
    imports = systems["conditional_imports"]
    
    table = Table(title="Import System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="magenta")
    
    table.add_row("Feature Flag",
                  "✅ ENABLED" if imports["feature_flag_enabled"] else "➖ DISABLED",
                  "ENABLE_EXPERIMENTAL_CACHING")
    table.add_row("Import Success",
                  "✅ SUCCESS" if imports["imports_successful"] else "❌ FAILED", 
                  "Caching modules loaded")
    table.add_row("Function Availability",
                  f"✅ {sum(imports['functions_available'].values())}/{len(imports['functions_available'])}",
                  "Required functions available")
    
    console.print(table)
    
    # Memory Monitoring Details
    console.print(f"\n🧠 [bold]Memory Monitoring (Phase 3)[/bold]")
    memory = systems["memory_monitoring"]
    
    if memory["active"]:
        table = Table(title="Memory Status")
        table.add_column("Metric", style="cyan")  
        table.add_column("Value", justify="right", style="magenta")
        table.add_column("Status", justify="center")
        
        usage_percent = memory["usage_percent"] * 100
        status_icon = "🟢" if usage_percent < 70 else ("🟡" if usage_percent < 85 else "🔴")
        
        table.add_row("System Usage", f"{usage_percent:.1f}%", status_icon)
        table.add_row("Available Memory", f"{memory['available_gb']:.1f} GB", "")
        table.add_row("Pressure Level", memory["pressure_level"].title(), "")
        
        console.print(table)
    else:
        console.print("Memory monitoring not active")
    
    # Testing and Validation Details
    console.print(f"\n🧪 [bold]Testing & Validation (Phase 4)[/bold]")
    testing = systems["testing_validation"]
    
    table = Table(title="Test Status")
    table.add_column("Component", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Status", justify="center")
    
    pass_rate = testing["passing_tests"] / testing["total_tests"] if testing["total_tests"] > 0 else 0
    test_icon = "✅" if pass_rate >= 0.95 else ("⚠️" if pass_rate >= 0.8 else "❌")
    
    table.add_row("Total Tests", str(testing["total_tests"]), "")
    table.add_row("Passing Tests", str(testing["passing_tests"]), test_icon)
    table.add_row("Test Coverage", f"{testing['test_coverage_percent']:.1f}%", "")
    
    console.print(table)
```

### 4. Error Propagation and Handling

Integrated error handling across all systems provides consistent user experience and actionable guidance.

#### Unified Error Classification

```python
class SystemError:
    """Unified error classification across all Phase 1-4 systems."""
    
    def __init__(self, error_type: str, system: str, severity: str, 
                 message: str, resolution_steps: list, related_systems: list = None):
        self.error_type = error_type
        self.system = system  # "build", "imports", "memory", "deps"
        self.severity = severity  # "critical", "warning", "info"
        self.message = message
        self.resolution_steps = resolution_steps
        self.related_systems = related_systems or []
        self.timestamp = time.time()
    
    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "system": self.system,
            "severity": self.severity,
            "message": self.message,
            "resolution_steps": self.resolution_steps,
            "related_systems": self.related_systems,
            "timestamp": self.timestamp
        }

class SystemErrorHandler:
    """Centralized error handling for all Phase 1-4 systems."""
    
    def __init__(self):
        self._errors = []
        self._error_callbacks = []
    
    def register_error(self, error: SystemError):
        """Register a system error for tracking and resolution."""
        self._errors.append(error)
        
        # Notify callbacks
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
        
        # Log error appropriately
        log_func = {
            "critical": logger.error,
            "warning": logger.warning,
            "info": logger.info
        }.get(error.severity, logger.info)
        
        log_func(f"{error.system.upper()} ERROR: {error.message}")
    
    def get_errors_by_system(self, system: str) -> list:
        """Get all errors for a specific system."""
        return [e for e in self._errors if e.system == system]
    
    def get_critical_errors(self) -> list:
        """Get all critical errors across systems.""" 
        return [e for e in self._errors if e.severity == "critical"]
    
    def clear_resolved_errors(self, system: str = None):
        """Clear errors that have been resolved."""
        if system:
            self._errors = [e for e in self._errors if e.system != system]
        else:
            self._errors.clear()

# Example integration with conditional imports
def handle_conditional_import_error(import_error: ImportError, module_name: str):
    """Handle conditional import errors with system context."""
    
    error = SystemError(
        error_type="import_failure",
        system="imports", 
        severity="warning",  # Not critical since fallbacks exist
        message=f"Failed to import {module_name}: {import_error}",
        resolution_steps=[
            f"Check if {module_name} is installed: poetry show {module_name}",
            "Install missing dependencies: poetry install",
            "Check for circular dependencies in module structure",
            f"Disable feature if issues persist: ENABLE_EXPERIMENTAL_CACHING = False"
        ],
        related_systems=["memory", "deps"]  # Memory monitoring and dependency checking related
    )
    
    system_error_handler.register_error(error)
    return error

# Global error handler instance
system_error_handler = SystemErrorHandler()
```

---

## Cross-System Troubleshooting Workflows

### Workflow 1: System Health Check

Complete health assessment across all Phase 1-4 systems:

```bash
# Step 1: Quick system overview
poetry run python -m giflab deps system-status

# Step 2: Detailed system breakdown
poetry run python -m giflab deps system-status --detailed

# Step 3: Generate diagnostic report
poetry run python -m giflab deps system-status --json > system_health_report.json

# Step 4: Specific system checks if issues found
poetry run python -m giflab deps check --verbose  # Dependency system
poetry run python -c "from giflab.diagnostics import diagnose_conditional_imports; print(diagnose_conditional_imports())"  # Imports
# Memory monitoring built into system-status
```

### Workflow 2: Feature Activation Troubleshooting

When experimental features aren't working as expected:

```python
def diagnose_feature_activation():
    """Comprehensive feature activation diagnosis."""
    
    diagnosis = {
        "feature_flags": {},
        "import_status": {},
        "memory_status": {},
        "dependencies": {},
        "recommendations": []
    }
    
    # Check feature flags
    diagnosis["feature_flags"] = {
        "experimental_caching": ENABLE_EXPERIMENTAL_CACHING,
        "memory_monitoring": MONITORING["memory_pressure"]["enabled"],
        "dependency_checking": MONITORING["dependency_checking"]["enabled"]
    }
    
    # Check import status
    from giflab.metrics import CACHING_ENABLED, get_frame_cache, resize_frame_cached
    diagnosis["import_status"] = {
        "caching_enabled": CACHING_ENABLED,
        "functions_available": {
            "get_frame_cache": get_frame_cache is not None,
            "resize_frame_cached": resize_frame_cached is not None
        }
    }
    
    # Check memory status
    try:
        from giflab.monitoring.memory_monitor import SystemMemoryMonitor
        monitor = SystemMemoryMonitor()
        stats = monitor.get_memory_stats()
        diagnosis["memory_status"] = {
            "monitoring_active": True,
            "usage_percent": stats.memory_percent,
            "available_gb": stats.available_memory / (1024**3),
            "pressure_level": _determine_pressure_level(stats.memory_percent)
        }
    except Exception as e:
        diagnosis["memory_status"] = {
            "monitoring_active": False,
            "error": str(e)
        }
    
    # Check dependencies
    diagnosis["dependencies"] = _check_key_dependencies()
    
    # Generate recommendations
    recommendations = []
    
    if diagnosis["feature_flags"]["experimental_caching"] and not diagnosis["import_status"]["caching_enabled"]:
        recommendations.append("Caching flag enabled but imports failed - check dependencies")
    
    if diagnosis["memory_status"].get("usage_percent", 0) > 0.85:
        recommendations.append("High memory usage may prevent feature activation")
    
    if not diagnosis["dependencies"].get("core_available", True):
        recommendations.append("Core dependencies missing - run 'poetry install'")
    
    diagnosis["recommendations"] = recommendations
    
    return diagnosis

# Usage
diagnosis = diagnose_feature_activation()
print(json.dumps(diagnosis, indent=2))
```

### Workflow 3: Performance Issue Diagnosis

When system performance is degraded:

```python
def diagnose_performance_issues():
    """Diagnose performance issues across all systems."""
    
    performance_report = {
        "timestamp": time.time(),
        "overall_performance": "unknown",
        "bottlenecks": [],
        "measurements": {}
    }
    
    # Test import performance
    import_start = time.perf_counter()
    try:
        import giflab.metrics
        import_time = (time.perf_counter() - import_start) * 1000
        performance_report["measurements"]["import_time_ms"] = import_time
        
        if import_time > 1000:  # >1 second
            performance_report["bottlenecks"].append("Slow module imports")
    except Exception as e:
        performance_report["measurements"]["import_error"] = str(e)
    
    # Test memory monitoring performance
    if MONITORING["memory_pressure"]["enabled"]:
        try:
            from giflab.monitoring.memory_monitor import SystemMemoryMonitor
            monitor = SystemMemoryMonitor()
            
            memory_start = time.perf_counter()
            stats = monitor.get_memory_stats()
            memory_time = (time.perf_counter() - memory_start) * 1000
            
            performance_report["measurements"]["memory_collection_ms"] = memory_time
            
            if memory_time > 100:  # >100ms
                performance_report["bottlenecks"].append("Slow memory collection")
                
        except Exception as e:
            performance_report["measurements"]["memory_error"] = str(e)
    
    # Test CLI performance  
    try:
        cli_start = time.perf_counter()
        from giflab.cli.deps_cmd import get_memory_status
        status = get_memory_status()
        cli_time = (time.perf_counter() - cli_start) * 1000
        
        performance_report["measurements"]["cli_response_ms"] = cli_time
        
        if cli_time > 500:  # >500ms
            performance_report["bottlenecks"].append("Slow CLI responses")
            
    except Exception as e:
        performance_report["measurements"]["cli_error"] = str(e)
    
    # Overall assessment
    if len(performance_report["bottlenecks"]) == 0:
        performance_report["overall_performance"] = "GOOD"
    elif len(performance_report["bottlenecks"]) <= 2:
        performance_report["overall_performance"] = "DEGRADED"
    else:
        performance_report["overall_performance"] = "POOR"
    
    return performance_report
```

---

## Best Practices for System Integration

### 1. Configuration Management

**Centralized Configuration Strategy**:
```python
# Best practice: Single source of truth for all Phase 1-4 settings
def get_integrated_config():
    """Get comprehensive configuration across all systems."""
    
    config = {
        "features": {
            "experimental_caching": ENABLE_EXPERIMENTAL_CACHING,
            "memory_monitoring": MONITORING["memory_pressure"]["enabled"],
            "dependency_checking": MONITORING["dependency_checking"]["enabled"],
            "enhanced_error_handling": LOG_VALIDATION_FAILURES
        },
        
        "memory": MONITORING["memory_pressure"],
        
        "caching": FRAME_CACHE,
        
        "cli": {
            "rich_output": True,
            "json_support": True,
            "verbose_errors": True
        },
        
        "integration": {
            "alert_system": True,
            "metrics_collection": True,
            "automated_diagnostics": True
        }
    }
    
    return config

# Usage in different systems
def init_memory_monitoring():
    config = get_integrated_config()
    if config["features"]["memory_monitoring"]:
        # Initialize with integrated config
        pass

def init_conditional_imports():
    config = get_integrated_config()
    if config["features"]["experimental_caching"]:
        # Proceed with conditional imports
        pass
```

### 2. Monitoring Integration

**Unified Metrics Collection**:
```python
def register_integrated_metrics():
    """Register metrics across all Phase 1-4 systems."""
    
    from giflab.metrics_collector import MetricsCollector
    collector = MetricsCollector.get_instance()
    
    # Phase 1: Build stability metrics
    collector.register_metric(
        "build_import_success_rate",
        description="Percentage of successful core imports",
        collection_func=lambda: _calculate_import_success_rate()
    )
    
    # Phase 2: Conditional import metrics
    collector.register_metric(
        "conditional_imports_active",
        description="Number of active conditional imports", 
        collection_func=lambda: _count_active_conditional_imports()
    )
    
    # Phase 3: Memory monitoring metrics (already implemented)
    # See memory-monitoring-architecture.md
    
    # Phase 4: Testing metrics
    collector.register_metric(
        "test_coverage_percent",
        description="Overall test coverage percentage",
        collection_func=lambda: _get_test_coverage_percentage()
    )
    
    # Integration metrics
    collector.register_metric(
        "system_integration_health",
        description="Overall system integration health score (0-100)",
        collection_func=lambda: _calculate_integration_health_score()
    )
```

### 3. Error Recovery Coordination

**Cross-System Recovery Procedures**:
```python
def execute_system_recovery():
    """Coordinated recovery across all systems."""
    
    recovery_report = {
        "timestamp": time.time(),
        "recovery_steps": [],
        "success": False
    }
    
    # Phase 1: Reset build state
    try:
        import gc
        gc.collect()  # Clear any import artifacts
        recovery_report["recovery_steps"].append("Build state reset: SUCCESS")
    except Exception as e:
        recovery_report["recovery_steps"].append(f"Build state reset: FAILED - {e}")
    
    # Phase 2: Reset conditional imports
    try:
        # Disable experimental features temporarily
        original_caching = ENABLE_EXPERIMENTAL_CACHING
        
        # Reset imports
        import importlib
        import giflab.metrics
        importlib.reload(giflab.metrics)
        
        recovery_report["recovery_steps"].append("Conditional imports reset: SUCCESS")
    except Exception as e:
        recovery_report["recovery_steps"].append(f"Conditional imports reset: FAILED - {e}")
    
    # Phase 3: Reset memory monitoring
    try:
        from giflab.monitoring.memory_integration import reset_memory_monitoring_state
        reset_success = reset_memory_monitoring_state()
        
        if reset_success:
            recovery_report["recovery_steps"].append("Memory monitoring reset: SUCCESS")
        else:
            recovery_report["recovery_steps"].append("Memory monitoring reset: FAILED")
    except Exception as e:
        recovery_report["recovery_steps"].append(f"Memory monitoring reset: FAILED - {e}")
    
    # Phase 4: Verify systems
    try:
        # Run basic system checks
        system_report = generate_unified_status_report()
        
        if system_report["overall_status"] in ["HEALTHY", "WARNINGS"]:
            recovery_report["success"] = True
            recovery_report["recovery_steps"].append("System verification: SUCCESS")
        else:
            recovery_report["recovery_steps"].append("System verification: FAILED")
    except Exception as e:
        recovery_report["recovery_steps"].append(f"System verification: FAILED - {e}")
    
    return recovery_report
```

### 4. Testing Integration

**Cross-System Test Patterns**:
```python
class TestSystemIntegration:
    """Integration tests across Phase 1-4 systems."""
    
    def test_feature_flag_coordination(self):
        """Test that feature flags coordinate properly across systems."""
        
        # Test caching disabled scenario
        with patch('giflab.config.ENABLE_EXPERIMENTAL_CACHING', False):
            status_report = generate_unified_status_report()
            
            # Verify conditional imports respect flag
            assert not status_report["systems"]["conditional_imports"]["imports_successful"]
            
            # Verify memory monitoring still works
            if status_report["systems"]["memory_monitoring"]["enabled"]:
                assert status_report["systems"]["memory_monitoring"]["active"]
            
            # Verify CLI still provides useful info
            deps_status = get_memory_status()
            assert "system_memory" in deps_status
    
    def test_error_propagation(self):
        """Test that errors propagate correctly across systems."""
        
        # Simulate import error
        with patch('builtins.__import__', side_effect=ImportError("Test error")):
            
            # Should register error in error handler
            handle_conditional_import_error(ImportError("Test error"), "test_module")
            
            # Should be reflected in status report
            status_report = generate_unified_status_report()
            assert "warnings" in status_report or "critical_issues" in status_report
            
            # Should be visible in CLI output
            # (Test would verify CLI command output contains error info)
    
    def test_memory_integration_with_caching(self):
        """Test memory monitoring integration with conditional caching."""
        
        # Enable caching and memory monitoring
        with patch('giflab.config.ENABLE_EXPERIMENTAL_CACHING', True):
            with patch('giflab.config.MONITORING', {"memory_pressure": {"enabled": True}}):
                
                # Create memory pressure scenario
                mock_stats = MemoryStats(
                    total_memory=1024**3,
                    available_memory=50*1024**2,  # Very low
                    process_memory=900*1024**2,
                    memory_percent=0.95,  # High pressure
                    timestamp=time.time()
                )
                
                with patch.object(SystemMemoryMonitor, 'get_memory_stats', return_value=mock_stats):
                    
                    # Feature manager should disable caching
                    feature_manager = MemoryAwareFeatureManager()
                    assert not feature_manager.should_enable_caching()
                    
                    # Should be reflected in status
                    status_report = generate_unified_status_report()
                    assert status_report["systems"]["memory_monitoring"]["pressure_level"] == "emergency"
    
    def test_cli_integration_completeness(self):
        """Test that CLI provides complete system visibility."""
        
        # Generate system status
        status_report = generate_unified_status_report()
        
        # Verify all Phase 1-4 systems are represented
        required_systems = ["build_stability", "conditional_imports", "memory_monitoring", "testing_validation"]
        
        for system in required_systems:
            assert system in status_report["systems"]
        
        # Verify CLI commands work
        # (Would test actual CLI command execution)
```

---

## Migration and Upgrade Patterns

### Gradual Feature Activation

```python
def gradual_feature_activation_plan():
    """Plan for gradually activating Phase 1-4 features in production."""
    
    activation_plan = {
        "phase_1": {
            "description": "Build stability improvements",
            "activation": "automatic",  # Already active
            "verification": [
                "Check core imports work",
                "Verify test pass rate >95%",
                "Confirm type errors <10"
            ]
        },
        
        "phase_2": {
            "description": "Conditional import architecture", 
            "activation": "gradual",
            "steps": [
                "Deploy with ENABLE_EXPERIMENTAL_CACHING=False (default)",
                "Monitor system stability for 1 week",
                "Enable caching in development environment",
                "Test for 2 weeks in development",
                "Enable in staging environment", 
                "Enable in production during low-traffic period"
            ]
        },
        
        "phase_3": {
            "description": "Memory monitoring infrastructure",
            "activation": "gradual",
            "steps": [
                "Deploy with monitoring enabled but no automatic eviction",
                "Monitor memory patterns for 1 week",
                "Enable automatic eviction with conservative thresholds",
                "Gradually adjust thresholds based on workload patterns"
            ]
        },
        
        "phase_4": {
            "description": "Enhanced testing and validation",
            "activation": "automatic",  # Testing improvements
            "verification": [
                "Verify all 141 tests pass",
                "Check integration test coverage",
                "Confirm no regressions in existing functionality"
            ]
        }
    }
    
    return activation_plan
```

---

## Summary

The Phase 1-4 integration provides:

- **Unified Architecture**: All systems work together through feature flags and shared interfaces
- **Progressive Enhancement**: Features can be enabled gradually with safe defaults
- **Comprehensive Monitoring**: Complete visibility into system health and performance  
- **Coordinated Error Handling**: Consistent error management with actionable guidance
- **Testing Integration**: Comprehensive test coverage across all integration points
- **Production Safety**: Conservative defaults with rollback procedures

This integrated system ensures that the architectural improvements from Phase 1-4 work together seamlessly while maintaining backward compatibility and production safety.

---

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Related Documentation: [Conditional Import Architecture](conditional-import-architecture.md), [Memory Monitoring Architecture](memory-monitoring-architecture.md), [CLI Dependency Troubleshooting](../guides/cli-dependency-troubleshooting.md)*