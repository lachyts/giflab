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
]