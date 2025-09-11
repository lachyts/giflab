"""Testing configuration profile.

Optimized for unit tests with caching disabled and strict validation.
Ensures reproducible test results.
"""

TESTING_PROFILE = {
    # Frame cache - disabled for reproducible tests
    "FRAME_CACHE": {
        "enabled": False,
    },
    
    # Validation cache - disabled for reproducible tests
    "VALIDATION_CACHE": {
        "enabled": False,
    },
    
    # Frame sampling - disabled for deterministic results
    "FRAME_SAMPLING": {
        "enabled": False,
    },
    
    # Monitoring - disabled for test performance
    "MONITORING": {
        "enabled": False,
    },
    
    # Metrics configuration - deterministic settings
    "metrics": {
        "SSIM_MODE": "fast",  # Fast for test speed
        "SSIM_MAX_FRAMES": 10,
        "USE_COMPREHENSIVE_METRICS": False,
        "TEMPORAL_CONSISTENCY_ENABLED": False,
        "RAW_METRICS": False,
        "ENABLE_POSITIONAL_SAMPLING": False,
        "ENABLE_DEEP_PERCEPTUAL": False,  # Avoid model loading
        "ENABLE_SSIMULACRA2": False,      # Avoid external deps
        "USE_ENHANCED_COMPOSITE_QUALITY": False,
    },
    
    # Validation configuration - strict for tests
    "validation": {
        "ENABLE_WRAPPER_VALIDATION": True,
        "FAIL_ON_VALIDATION_ERROR": True,  # Fail fast in tests
        "LOG_VALIDATION_FAILURES": True,
        "TIMING_VALIDATION_ENABLED": False,  # Non-deterministic
        "VALIDATION_TIMEOUT_MS": 5000,
    },
    
    # Compression configuration - minimal for speed
    "compression": {
        "FRAME_KEEP_RATIOS": [1.0],  # No frame dropping
        "COLOR_KEEP_COUNTS": [256],   # No color reduction
        "LOSSY_LEVELS": [0],          # No lossy compression
        "ENGINES": ["gifsicle"],     # Single engine
    },
    
    # Path configuration - use temp directories
    "paths": {
        "TMP_DIR": "/tmp/giflab_test",
        "LOGS_DIR": "/tmp/giflab_test_logs",
    },
}