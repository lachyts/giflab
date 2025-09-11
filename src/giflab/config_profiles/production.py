"""Production configuration profile.

Balanced settings optimized for stability, performance, and resource usage
in production environments.
"""

PRODUCTION_PROFILE = {
    # Frame cache - balanced for production
    "FRAME_CACHE": {
        "enabled": True,
        "memory_limit_mb": 500,     # 500MB memory cache
        "disk_limit_mb": 2000,       # 2GB disk cache
        "ttl_seconds": 86400,        # 24 hours TTL
        "resize_cache_enabled": True,
        "resize_cache_memory_mb": 200,
        "resize_cache_ttl_seconds": 3600,
        "enable_buffer_pooling": True,
    },
    
    # Validation cache - longer TTL for production stability
    "VALIDATION_CACHE": {
        "enabled": True,
        "memory_limit_mb": 100,
        "disk_limit_mb": 1000,
        "ttl_seconds": 172800,  # 48 hours
        "cache_ssim": True,
        "cache_ms_ssim": True,
        "cache_lpips": True,
        "cache_gradient_color": True,
        "cache_ssimulacra2": True,
    },
    
    # Frame sampling - enabled for efficiency
    "FRAME_SAMPLING": {
        "enabled": True,
        "min_frames_threshold": 30,
        "default_strategy": "adaptive",
        "confidence_level": 0.95,
        
        "adaptive": {
            "base_rate": 0.2,
            "motion_threshold": 0.1,
            "max_rate": 0.8,
        },
        
        "verbose": False,  # Quiet in production
    },
    
    # Monitoring - lightweight for production
    "MONITORING": {
        "enabled": True,
        "backend": "sqlite",
        "buffer_size": 10000,
        "flush_interval": 10.0,
        "sampling_rate": 0.1,  # Sample 10% for lower overhead
        
        "sqlite": {
            "retention_days": 7.0,   # One week retention
            "max_size_mb": 100.0,
        },
        
        "alerts": {
            "cache_hit_rate_warning": 0.4,
            "cache_hit_rate_critical": 0.2,
            "memory_usage_warning": 0.8,
            "memory_usage_critical": 0.95,
            "eviction_rate_spike": 3.0,
            "response_time_degradation": 1.5,
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
    
    # Metrics configuration
    "metrics": {
        "SSIM_MODE": "optimized",  # Balance speed and accuracy
        "SSIM_MAX_FRAMES": 30,
        "USE_COMPREHENSIVE_METRICS": True,
        "TEMPORAL_CONSISTENCY_ENABLED": True,
        "RAW_METRICS": False,  # No raw metrics in production
        "ENABLE_POSITIONAL_SAMPLING": True,
        "ENABLE_DEEP_PERCEPTUAL": True,
        "ENABLE_SSIMULACRA2": True,
        "LPIPS_MAX_FRAMES": 100,
        "SSIMULACRA2_MAX_FRAMES": 30,
    },
    
    # Validation configuration
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": True,
        "FAIL_ON_VALIDATION_ERROR": False,
        "LOG_VALIDATION_FAILURES": True,
        "TIMING_VALIDATION_ENABLED": True,
        "TIMING_VALIDATION_ALERT_ON_FAILURE": False,  # Don't alert in production
    },
}