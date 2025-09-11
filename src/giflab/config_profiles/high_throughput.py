"""High throughput configuration profile.

Optimized for batch processing with focus on speed over accuracy.
Suitable for large-scale GIF processing pipelines.
"""

HIGH_THROUGHPUT_PROFILE = {
    # Frame cache - optimized for throughput
    "FRAME_CACHE": {
        "enabled": True,
        "memory_limit_mb": 1000,     # 1GB memory cache
        "disk_limit_mb": 5000,        # 5GB disk cache
        "ttl_seconds": 3600,          # 1 hour TTL
        "resize_cache_enabled": True,
        "resize_cache_memory_mb": 300,
        "resize_cache_ttl_seconds": 1800,
        "enable_buffer_pooling": True,
    },
    
    # Validation cache - cache everything for speed
    "VALIDATION_CACHE": {
        "enabled": True,
        "memory_limit_mb": 200,
        "disk_limit_mb": 2000,
        "ttl_seconds": 7200,  # 2 hours
        "cache_ssim": True,
        "cache_ms_ssim": True,
        "cache_lpips": True,
        "cache_gradient_color": True,
        "cache_ssimulacra2": True,
    },
    
    # Frame sampling - progressive for speed/accuracy balance
    "FRAME_SAMPLING": {
        "enabled": True,
        "min_frames_threshold": 20,
        "default_strategy": "progressive",
        "confidence_level": 0.90,  # Lower confidence for speed
        
        "progressive": {
            "initial_rate": 0.1,    # Start with 10%
            "increment_rate": 0.1,
            "max_iterations": 3,    # Fewer iterations
            "target_ci_width": 0.15,  # Wider CI acceptable
        },
        
        "uniform": {
            "sampling_rate": 0.25,  # Fallback to 25% sampling
        },
        
        "verbose": False,
    },
    
    # Monitoring - disabled for maximum throughput
    "MONITORING": {
        "enabled": False,  # Disable to reduce overhead
    },
    
    # Metrics configuration - fast processing
    "metrics": {
        "SSIM_MODE": "optimized",    # Balance speed/accuracy
        "SSIM_MAX_FRAMES": 20,       # Process fewer frames
        "USE_COMPREHENSIVE_METRICS": False,  # Basic metrics only
        "TEMPORAL_CONSISTENCY_ENABLED": False,  # Skip temporal
        "RAW_METRICS": False,
        "ENABLE_POSITIONAL_SAMPLING": False,
        "ENABLE_DEEP_PERCEPTUAL": False,  # Skip expensive metrics
        "ENABLE_SSIMULACRA2": False,      # Skip expensive metrics
        "USE_ENHANCED_COMPOSITE_QUALITY": False,  # Use simple composite
    },
    
    # Validation configuration - minimal validation
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": False,  # Skip validation for speed
        "FAIL_ON_VALIDATION_ERROR": False,
        "LOG_VALIDATION_FAILURES": False,
        "TIMING_VALIDATION_ENABLED": False,
        "VALIDATION_TIMEOUT_MS": 1000,  # Short timeout
    },
    
    # Compression configuration - parallel processing
    "compression": {
        "ENGINES": ["gifsicle"],  # Fastest engine only
    },
}