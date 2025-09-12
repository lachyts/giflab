"""
Performance monitoring infrastructure for GifLab optimization systems.

This module provides comprehensive monitoring for:
- FrameCache performance and hit rates
- ValidationCache metrics by validation type
- ResizedFrameCache and buffer pool efficiency  
- Frame sampling accuracy and speedup
- Lazy loading patterns and import times
"""

from .metrics_collector import (
    MetricsCollector,
    MetricType,
    get_metrics_collector,
    reset_metrics_collector,
)
from .backends import MetricsBackend, InMemoryBackend, SQLiteBackend
from .decorators import (
    track_timing,
    track_counter,
    track_gauge,
    track_histogram,
)
from .integration import (
    instrument_frame_cache,
    instrument_validation_cache,
    instrument_resize_cache,
    instrument_sampling,
    instrument_lazy_imports,
    instrument_all_systems,
)
from .alerting import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertNotifier,
    get_alert_manager,
    get_alert_notifier,
    check_alerts,
)
from .memory_monitor import (
    MemoryStats,
    MemoryPressureLevel,
    CacheMemoryUsage,
    SystemMemoryMonitor,
    CacheMemoryTracker,
    MemoryPressureManager,
    ConservativeEvictionPolicy,
    get_system_memory_monitor,
    get_cache_memory_tracker,
    get_memory_pressure_manager,
    start_memory_monitoring,
    stop_memory_monitoring,
    is_memory_monitoring_available,
)
from .memory_integration import (
    MemoryPressureIntegration,
    get_memory_integration,
    initialize_memory_monitoring,
    instrument_cache_with_memory_tracking,
    instrument_all_caches_memory,
)
from .cache_effectiveness import (
    CacheEffectivenessMonitor,
    CacheOperationType,
    CacheEffectivenessStats,
    BaselineComparison,
    get_cache_effectiveness_monitor,
    is_cache_effectiveness_monitoring_enabled,
)
from .baseline_framework import (
    PerformanceBaselineFramework,
    BaselineTestMode,
    BaselineStatistics,
    PerformanceMeasurement,
    WorkloadScenario,
    baseline_performance_test,
    get_baseline_framework,
    is_baseline_testing_enabled,
)
from .effectiveness_analysis import (
    CacheEffectivenessAnalyzer,
    CacheRecommendation,
    CacheEffectivenessAnalysis,
    analyze_cache_effectiveness,
    get_effectiveness_analyzer,
)
from .performance_regression import (
    PerformanceBaseline,
    RegressionAlert,
    RegressionDetector,
    PerformanceHistory,
    ContinuousMonitor,
    create_performance_monitor,
)

__all__ = [
    # Core classes
    "MetricsCollector",
    "MetricType",
    "MetricsBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    # Singleton access
    "get_metrics_collector",
    "reset_metrics_collector",
    # Decorators
    "track_timing",
    "track_counter",
    "track_gauge",
    "track_histogram",
    # Integration
    "instrument_frame_cache",
    "instrument_validation_cache", 
    "instrument_resize_cache",
    "instrument_sampling",
    "instrument_lazy_imports",
    "instrument_all_systems",
    # Alerting
    "Alert",
    "AlertLevel",
    "AlertManager",
    "AlertNotifier",
    "get_alert_manager",
    "get_alert_notifier",
    "check_alerts",
    # Memory monitoring
    "MemoryStats",
    "MemoryPressureLevel",
    "CacheMemoryUsage",
    "SystemMemoryMonitor",
    "CacheMemoryTracker",
    "MemoryPressureManager",
    "ConservativeEvictionPolicy",
    "get_system_memory_monitor",
    "get_cache_memory_tracker",
    "get_memory_pressure_manager",
    "start_memory_monitoring",
    "stop_memory_monitoring",
    "is_memory_monitoring_available",
    # Memory integration
    "MemoryPressureIntegration",
    "get_memory_integration",
    "initialize_memory_monitoring",
    "instrument_cache_with_memory_tracking",
    "instrument_all_caches_memory",
    # Cache effectiveness monitoring
    "CacheEffectivenessMonitor",
    "CacheOperationType",
    "CacheEffectivenessStats",
    "BaselineComparison",
    "get_cache_effectiveness_monitor",
    "is_cache_effectiveness_monitoring_enabled",
    # Performance baseline framework
    "PerformanceBaselineFramework",
    "BaselineTestMode",
    "BaselineStatistics",
    "PerformanceMeasurement",
    "WorkloadScenario",
    "baseline_performance_test",
    "get_baseline_framework",
    "is_baseline_testing_enabled",
    # Effectiveness analysis
    "CacheEffectivenessAnalyzer",
    "CacheRecommendation",
    "CacheEffectivenessAnalysis",
    "analyze_cache_effectiveness",
    "get_effectiveness_analyzer",
    # Performance regression monitoring (Phase 7)
    "PerformanceBaseline",
    "RegressionAlert",
    "RegressionDetector",
    "PerformanceHistory",
    "ContinuousMonitor",
    "create_performance_monitor",
]