"""Development configuration profile.

Optimized for development with aggressive caching, verbose logging,
and full frame processing for accuracy verification.
"""

DEVELOPMENT_PROFILE = {
    # Frame cache - aggressive caching for fast iteration
    "FRAME_CACHE": {
        "enabled": True,
        "memory_limit_mb": 1000,  # 1GB memory cache
        "disk_limit_mb": 5000,     # 5GB disk cache
        "ttl_seconds": 3600,       # 1 hour TTL
        "resize_cache_enabled": True,
        "resize_cache_memory_mb": 300,
        "resize_cache_ttl_seconds": 1800,
        "enable_buffer_pooling": True,
    },
    
    # Validation cache - keep results for quick re-testing
    "VALIDATION_CACHE": {
        "enabled": True,
        "memory_limit_mb": 200,
        "disk_limit_mb": 2000,
        "ttl_seconds": 3600,  # 1 hour
        "cache_ssim": True,
        "cache_ms_ssim": True,
        "cache_lpips": True,
        "cache_gradient_color": True,
        "cache_ssimulacra2": True,
    },
    
    # Frame sampling - disabled for full accuracy
    "FRAME_SAMPLING": {
        "enabled": False,  # Process all frames in development
        "verbose": True,   # Show sampling decisions when enabled
    },
    
    # Monitoring - full visibility
    "MONITORING": {
        "enabled": True,
        "backend": "sqlite",
        "buffer_size": 50000,  # Large buffer for detailed analysis
        "flush_interval": 5.0,  # Frequent flushes
        "sampling_rate": 1.0,   # Capture everything
        
        "sqlite": {
            "retention_days": 30.0,  # Keep data for a month
            "max_size_mb": 500.0,
        },
        
        "alerts": {
            "cache_hit_rate_warning": 0.3,   # Lower threshold for dev
            "cache_hit_rate_critical": 0.1,
            "memory_usage_warning": 0.9,     # Higher threshold for dev
            "memory_usage_critical": 0.98,
            "eviction_rate_spike": 5.0,      # More tolerant
            "response_time_degradation": 2.0,
        },
        
        "systems": {
            "frame_cache": True,
            "validation_cache": True,
            "resize_cache": True,
            "sampling": True,
            "lazy_imports": True,
            "metrics_calculation": True,
        },
        
        "verbose": True,  # Verbose logging
    },
    
    # Metrics configuration
    "metrics": {
        "SSIM_MODE": "comprehensive",  # Most accurate mode
        "SSIM_MAX_FRAMES": 100,        # Process more frames
        "USE_COMPREHENSIVE_METRICS": True,
        "TEMPORAL_CONSISTENCY_ENABLED": True,
        "RAW_METRICS": True,  # Include raw values for debugging
        "ENABLE_POSITIONAL_SAMPLING": True,
        "ENABLE_DEEP_PERCEPTUAL": True,
        "ENABLE_SSIMULACRA2": True,
        "LPIPS_MAX_FRAMES": 100,
        "SSIMULACRA2_MAX_FRAMES": 50,
    },
    
    # Validation configuration
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": True,
        "FAIL_ON_VALIDATION_ERROR": False,  # Don't break on errors
        "LOG_VALIDATION_FAILURES": True,
        "TIMING_VALIDATION_ENABLED": True,
        "TIMING_VALIDATION_ALERT_ON_FAILURE": True,
    },
}