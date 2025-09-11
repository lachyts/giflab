"""High memory configuration profile.

Optimized for environments with abundant memory resources.
Maximizes caching and processes all frames for highest quality.
"""

HIGH_MEMORY_PROFILE = {
    # Frame cache - maximum caching
    "FRAME_CACHE": {
        "enabled": True,
        "memory_limit_mb": 2000,     # 2GB memory cache
        "disk_limit_mb": 10000,       # 10GB disk cache
        "ttl_seconds": 172800,        # 48 hours TTL
        "resize_cache_enabled": True,
        "resize_cache_memory_mb": 500,  # 500MB for resized frames
        "resize_cache_ttl_seconds": 7200,
        "enable_buffer_pooling": True,
    },
    
    # Validation cache - large cache for results
    "VALIDATION_CACHE": {
        "enabled": True,
        "memory_limit_mb": 500,      # 500MB memory cache
        "disk_limit_mb": 5000,        # 5GB disk cache
        "ttl_seconds": 259200,        # 72 hours
        "cache_ssim": True,
        "cache_ms_ssim": True,
        "cache_lpips": True,
        "cache_gradient_color": True,
        "cache_ssimulacra2": True,
    },
    
    # Frame sampling - disabled for full processing
    "FRAME_SAMPLING": {
        "enabled": False,  # Process all frames when memory available
    },
    
    # Monitoring - comprehensive with large buffers
    "MONITORING": {
        "enabled": True,
        "backend": "sqlite",
        "buffer_size": 100000,  # Very large buffer
        "flush_interval": 30.0,  # Less frequent flushes
        "sampling_rate": 0.5,    # Sample 50% of operations
        
        "sqlite": {
            "retention_days": 30.0,  # Keep data for a month
            "max_size_mb": 1000.0,   # 1GB for metrics
        },
        
        "alerts": {
            "cache_hit_rate_warning": 0.5,   # Expect higher hit rates
            "cache_hit_rate_critical": 0.3,
            "memory_usage_warning": 0.85,    # Allow more memory usage
            "memory_usage_critical": 0.95,
            "eviction_rate_spike": 2.0,      # Lower threshold with more memory
            "response_time_degradation": 1.2,
        },
        
        "systems": {
            "frame_cache": True,
            "validation_cache": True,
            "resize_cache": True,
            "sampling": True,
            "lazy_imports": True,
            "metrics_calculation": True,
        },
        
        "verbose": False,
    },
    
    # Metrics configuration - maximum quality
    "metrics": {
        "SSIM_MODE": "comprehensive",
        "SSIM_MAX_FRAMES": 200,  # Process many frames
        "USE_COMPREHENSIVE_METRICS": True,
        "TEMPORAL_CONSISTENCY_ENABLED": True,
        "RAW_METRICS": False,
        "ENABLE_POSITIONAL_SAMPLING": True,
        "ENABLE_DEEP_PERCEPTUAL": True,
        "ENABLE_SSIMULACRA2": True,
        "LPIPS_MAX_FRAMES": 200,      # Process more frames
        "SSIMULACRA2_MAX_FRAMES": 100,
        "LPIPS_DOWNSCALE_SIZE": 1024,  # Higher resolution
    },
    
    # Validation configuration
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": True,
        "FAIL_ON_VALIDATION_ERROR": False,
        "LOG_VALIDATION_FAILURES": True,
        "TIMING_VALIDATION_ENABLED": True,
        "TIMING_VALIDATION_ALERT_ON_FAILURE": False,
        "VALIDATION_TIMEOUT_MS": 10000,  # Longer timeout
    },
}