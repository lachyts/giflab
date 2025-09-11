"""Low memory configuration profile.

Optimized for memory-constrained environments with minimal caching
and aggressive frame sampling.
"""

LOW_MEMORY_PROFILE = {
    # Frame cache - minimal memory usage
    "FRAME_CACHE": {
        "enabled": True,
        "memory_limit_mb": 100,      # 100MB memory cache
        "disk_limit_mb": 500,         # 500MB disk cache
        "ttl_seconds": 3600,          # 1 hour TTL
        "resize_cache_enabled": True,
        "resize_cache_memory_mb": 50,  # 50MB for resized frames
        "resize_cache_ttl_seconds": 600,  # 10 minutes
        "enable_buffer_pooling": True,
    },
    
    # Validation cache - minimal caching
    "VALIDATION_CACHE": {
        "enabled": True,
        "memory_limit_mb": 25,       # 25MB memory cache
        "disk_limit_mb": 250,         # 250MB disk cache
        "ttl_seconds": 3600,          # 1 hour
        "cache_ssim": True,
        "cache_ms_ssim": True,
        "cache_lpips": False,         # Skip expensive metrics
        "cache_gradient_color": True,
        "cache_ssimulacra2": False,   # Skip expensive metrics
    },
    
    # Frame sampling - aggressive sampling
    "FRAME_SAMPLING": {
        "enabled": True,
        "min_frames_threshold": 10,  # Sample even small GIFs
        "default_strategy": "uniform",
        "confidence_level": 0.90,    # Lower confidence for speed
        
        "uniform": {
            "sampling_rate": 0.2,     # Sample only 20% of frames
        },
        
        "verbose": False,
    },
    
    # Monitoring - minimal overhead
    "MONITORING": {
        "enabled": True,
        "backend": "memory",  # No disk persistence
        "buffer_size": 1000,  # Small buffer
        "flush_interval": 60.0,  # Infrequent flushes
        "sampling_rate": 0.05,   # Sample only 5%
        
        "alerts": {
            "cache_hit_rate_warning": 0.2,   # Lower expectations
            "cache_hit_rate_critical": 0.1,
            "memory_usage_warning": 0.7,     # Alert earlier
            "memory_usage_critical": 0.85,
            "eviction_rate_spike": 5.0,      # Higher tolerance
            "response_time_degradation": 2.0,
        },
        
        "systems": {
            "frame_cache": True,
            "validation_cache": True,
            "resize_cache": False,  # Disable to save memory
            "sampling": True,
            "lazy_imports": True,
            "metrics_calculation": False,  # Reduce overhead
        },
        
        "verbose": False,
    },
    
    # Metrics configuration - minimal processing
    "metrics": {
        "SSIM_MODE": "fast",         # Fastest mode
        "SSIM_MAX_FRAMES": 10,       # Process fewer frames
        "USE_COMPREHENSIVE_METRICS": False,  # Basic metrics only
        "TEMPORAL_CONSISTENCY_ENABLED": False,
        "RAW_METRICS": False,
        "ENABLE_POSITIONAL_SAMPLING": False,
        "ENABLE_DEEP_PERCEPTUAL": False,  # Skip expensive metrics
        "ENABLE_SSIMULACRA2": False,      # Skip expensive metrics
        "LPIPS_MAX_FRAMES": 50,
        "SSIMULACRA2_MAX_FRAMES": 10,
        "LPIPS_DOWNSCALE_SIZE": 256,  # Lower resolution
    },
    
    # Validation configuration
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": True,
        "FAIL_ON_VALIDATION_ERROR": False,
        "LOG_VALIDATION_FAILURES": False,  # Reduce logging
        "TIMING_VALIDATION_ENABLED": False,  # Skip timing validation
        "VALIDATION_TIMEOUT_MS": 2000,  # Shorter timeout
    },
}