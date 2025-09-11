"""Configuration profiles for different deployment scenarios."""

from .development import DEVELOPMENT_PROFILE
from .production import PRODUCTION_PROFILE
from .high_memory import HIGH_MEMORY_PROFILE
from .low_memory import LOW_MEMORY_PROFILE
from .high_throughput import HIGH_THROUGHPUT_PROFILE
from .interactive import INTERACTIVE_PROFILE
from .testing import TESTING_PROFILE

__all__ = [
    "DEVELOPMENT_PROFILE",
    "PRODUCTION_PROFILE", 
    "HIGH_MEMORY_PROFILE",
    "LOW_MEMORY_PROFILE",
    "HIGH_THROUGHPUT_PROFILE",
    "INTERACTIVE_PROFILE",
    "TESTING_PROFILE",
]

# Profile descriptions for documentation
PROFILE_DESCRIPTIONS = {
    "development": "Aggressive caching and verbose logging for development",
    "production": "Balanced settings optimized for stability",
    "high_memory": "Maximum caching for memory-rich environments",
    "low_memory": "Conservative settings for constrained environments",
    "high_throughput": "Optimized for batch processing speed",
    "interactive": "Low-latency settings for real-time usage",
    "testing": "Minimal caching for unit tests",
}