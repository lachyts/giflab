"""Interactive configuration profile.

Optimized for low-latency, real-time usage with quick feedback.
Suitable for interactive applications and preview generation.
"""

INTERACTIVE_PROFILE = {
    # Frame cache - small, fast cache
    "FRAME_CACHE": {
        "enabled": True,
        "memory_limit_mb": 300,      # 300MB memory cache
        "disk_limit_mb": 0,           # No disk cache for speed
        "ttl_seconds": 300,           # 5 minutes TTL
        "resize_cache_enabled": True,
        "resize_cache_memory_mb": 100,
        "resize_cache_ttl_seconds": 300,
        "enable_buffer_pooling": True,
    },
    
    # Validation cache - minimal for quick response
    "VALIDATION_CACHE": {
        "enabled": True,
        "memory_limit_mb": 50,
        "disk_limit_mb": 0,  # No disk cache
        "ttl_seconds": 600,  # 10 minutes
        "cache_ssim": True,
        "cache_ms_ssim": False,
        "cache_lpips": False,
        "cache_gradient_color": True,
        "cache_ssimulacra2": False,
    },
    
    # Frame sampling - aggressive for quick preview
    "FRAME_SAMPLING": {
        "enabled": True,
        "min_frames_threshold": 5,  # Sample even tiny GIFs
        "default_strategy": "uniform",
        "confidence_level": 0.85,  # Lower confidence for speed
        
        "uniform": {
            "sampling_rate": 0.1,  # Sample only 10% for preview
        },
        
        "verbose": False,
    },
    
    # Monitoring - disabled for responsiveness
    "MONITORING": {
        "enabled": False,
    },
    
    # Metrics configuration - minimal for speed
    "metrics": {
        "SSIM_MODE": "fast",         # Fastest mode
        "SSIM_MAX_FRAMES": 5,        # Very few frames
        "USE_COMPREHENSIVE_METRICS": False,
        "TEMPORAL_CONSISTENCY_ENABLED": False,
        "RAW_METRICS": False,
        "ENABLE_POSITIONAL_SAMPLING": True,  # Quick position check
        "POSITIONAL_METRICS": ["ssim"],  # Only SSIM for positions
        "ENABLE_DEEP_PERCEPTUAL": False,
        "ENABLE_SSIMULACRA2": False,
        "USE_ENHANCED_COMPOSITE_QUALITY": False,
    },
    
    # Validation configuration - skip validation
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": False,
        "FAIL_ON_VALIDATION_ERROR": False,
        "LOG_VALIDATION_FAILURES": False,
        "TIMING_VALIDATION_ENABLED": False,
        "VALIDATION_TIMEOUT_MS": 500,  # Very short timeout
    },
    
    # Compression configuration - quick processing
    "compression": {
        "FRAME_KEEP_RATIOS": [1.0, 0.5],  # Only two options
        "COLOR_KEEP_COUNTS": [256, 64],   # Only two options
        "LOSSY_LEVELS": [0],               # No lossy for preview
        "ENGINES": ["gifsicle"],          # Fastest engine
    },
}