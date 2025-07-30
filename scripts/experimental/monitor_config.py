"""Configuration settings for pipeline elimination monitor."""

from pathlib import Path
from typing import List, Dict, Any
import json
import os


class MonitorConfig:
    """Configuration class for pipeline elimination monitoring."""
    
    # Default monitoring settings
    DEFAULT_REFRESH_INTERVAL = 30  # seconds
    DEFAULT_FAILURES_TO_SHOW = 3
    DEFAULT_BUFFER_SIZE = 25
    
    # Progress estimation settings
    DEFAULT_ESTIMATED_TOTAL_JOBS = 93500  # Conservative fallback
    BASE_TIME_PER_JOB = 2.5  # seconds (conservative estimate)
    
    # File search locations (in priority order)
    DEFAULT_SEARCH_LOCATIONS = [
        "elimination_results/latest/streaming_results.csv",
        "elimination_results/streaming_results.csv", 
        "streaming_results.csv",
        "latest/streaming_results.csv"
    ]
    
    # Rate calculation settings
    RATE_HISTORY_SIZE = 10  # Number of measurements to keep for trend analysis
    MIN_RATE_FOR_PROCESSING = 0.1  # Minimum results/minute to consider "actively processing"
    
    # Trend detection thresholds
    ACCELERATING_THRESHOLD = 1.2  # Rate > avg * 1.2 = accelerating
    SLOWING_THRESHOLD = 0.8       # Rate < avg * 0.8 = slowing
    
    # Batching information
    EXPECTED_BATCH_SIZE_MIN = 15
    EXPECTED_BATCH_SIZE_MAX = 25
    
    def __init__(self, config_file: str = None):
        """Initialize configuration.
        
        Args:
            config_file: Optional path to JSON config file to override defaults
        """
        self.refresh_interval = self.DEFAULT_REFRESH_INTERVAL
        self.failures_to_show = self.DEFAULT_FAILURES_TO_SHOW
        self.buffer_size = self.DEFAULT_BUFFER_SIZE
        self.estimated_total_jobs = self.DEFAULT_ESTIMATED_TOTAL_JOBS
        self.base_time_per_job = self.BASE_TIME_PER_JOB
        self.search_locations = self.DEFAULT_SEARCH_LOCATIONS.copy()
        self.rate_history_size = self.RATE_HISTORY_SIZE
        self.min_rate_for_processing = self.MIN_RATE_FOR_PROCESSING
        self.accelerating_threshold = self.ACCELERATING_THRESHOLD
        self.slowing_threshold = self.SLOWING_THRESHOLD
        self.expected_batch_size_min = self.EXPECTED_BATCH_SIZE_MIN
        self.expected_batch_size_max = self.EXPECTED_BATCH_SIZE_MAX
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Load from environment variables if set
        self.load_from_environment()
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update settings from file
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        
                print(f"✅ Loaded configuration from {config_file}")
            else:
                print(f"⚠️  Configuration file not found: {config_file}")
        except Exception as e:
            print(f"❌ Failed to load configuration file: {e}")
    
    def load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'MONITOR_REFRESH_INTERVAL': ('refresh_interval', int),
            'MONITOR_FAILURES_TO_SHOW': ('failures_to_show', int), 
            'MONITOR_BUFFER_SIZE': ('buffer_size', int),
            'MONITOR_ESTIMATED_TOTAL_JOBS': ('estimated_total_jobs', int),
            'MONITOR_BASE_TIME_PER_JOB': ('base_time_per_job', float),
            'MONITOR_MIN_PROCESSING_RATE': ('min_rate_for_processing', float),
        }
        
        for env_var, (attr_name, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    setattr(self, attr_name, type_func(env_value))
                    print(f"🔧 Set {attr_name} = {env_value} from environment")
                except ValueError as e:
                    print(f"⚠️  Invalid environment value for {env_var}: {env_value} ({e})")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file.
        
        Args:
            config_file: Path where to save the configuration
        """
        try:
            config_data = {
                'refresh_interval': self.refresh_interval,
                'failures_to_show': self.failures_to_show,
                'buffer_size': self.buffer_size,
                'estimated_total_jobs': self.estimated_total_jobs,
                'base_time_per_job': self.base_time_per_job,
                'search_locations': self.search_locations,
                'rate_history_size': self.rate_history_size,
                'min_rate_for_processing': self.min_rate_for_processing,
                'accelerating_threshold': self.accelerating_threshold,
                'slowing_threshold': self.slowing_threshold,
                'expected_batch_size_min': self.expected_batch_size_min,
                'expected_batch_size_max': self.expected_batch_size_max,
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"💾 Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def get_search_locations(self) -> List[str]:
        """Get list of file search locations in priority order."""
        return self.search_locations
    
    def get_trend_description(self, current_rate: float, avg_rate: float) -> str:
        """Get trend description based on current vs average rate.
        
        Args:
            current_rate: Current processing rate
            avg_rate: Average processing rate
            
        Returns:
            Trend description string with emoji
        """
        if current_rate > avg_rate * self.accelerating_threshold:
            return "📈 Accelerating"
        elif current_rate < avg_rate * self.slowing_threshold:
            return "📉 Slowing" if current_rate > self.min_rate_for_processing else "⏸️  Batching"
        else:
            return "➡️  Steady"
    
    def is_actively_processing(self, current_rate: float) -> bool:
        """Check if system is actively processing based on rate.
        
        Args:
            current_rate: Current processing rate in results/minute
            
        Returns:
            True if actively processing, False if likely batching
        """
        return current_rate > self.min_rate_for_processing
    
    def estimate_remaining_time(self, remaining_jobs: int) -> str:
        """Estimate remaining execution time.
        
        Args:
            remaining_jobs: Number of jobs remaining
            
        Returns:
            Human-readable time estimate
        """
        estimated_seconds = remaining_jobs * self.base_time_per_job
        
        if estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"
    
    def get_batching_info(self) -> Dict[str, Any]:
        """Get information about expected batching behavior.
        
        Returns:
            Dictionary with batching information
        """
        return {
            'min_batch_size': self.expected_batch_size_min,
            'max_batch_size': self.expected_batch_size_max,
            'description': f"Results are batched every {self.expected_batch_size_min}-{self.expected_batch_size_max} tests for performance",
            'explanation': "Large jumps in counts are normal and expected"
        }
    
    def validate(self) -> List[str]:
        """Validate configuration settings.
        
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if self.refresh_interval < 1:
            warnings.append("Refresh interval should be at least 1 second")
        
        if self.failures_to_show < 0:
            warnings.append("Number of failures to show should be non-negative")
        
        if self.buffer_size < 1:
            warnings.append("Buffer size should be at least 1")
        
        if self.estimated_total_jobs <= 0:
            warnings.append("Estimated total jobs should be positive")
        
        if self.base_time_per_job <= 0:
            warnings.append("Base time per job should be positive")
        
        if not self.search_locations:
            warnings.append("At least one search location should be specified")
        
        if self.min_rate_for_processing < 0:
            warnings.append("Minimum processing rate should be non-negative")
        
        if self.accelerating_threshold <= 1.0:
            warnings.append("Accelerating threshold should be > 1.0")
        
        if self.slowing_threshold >= 1.0:
            warnings.append("Slowing threshold should be < 1.0")
        
        return warnings


# Default global configuration instance
default_config = MonitorConfig()


def create_sample_config_file(filename: str = "monitor_config.json"):
    """Create a sample configuration file with documentation.
    
    Args:
        filename: Name of the config file to create
    """
    sample_config = {
        "_comment": "Pipeline elimination monitor configuration",
        "_description": "Customize monitoring behavior by editing these values",
        
        "refresh_interval": 30,
        "_refresh_interval_help": "Update interval in seconds (minimum: 1)",
        
        "failures_to_show": 3,
        "_failures_to_show_help": "Number of recent failures to display",
        
        "buffer_size": 25,
        "_buffer_size_help": "Internal buffer size for batching updates",
        
        "estimated_total_jobs": 93500,
        "_estimated_total_jobs_help": "Estimated total jobs for progress calculation",
        
        "base_time_per_job": 2.5,
        "_base_time_per_job_help": "Estimated seconds per job for ETA calculation",
        
        "search_locations": [
            "elimination_results/latest/streaming_results.csv",
            "elimination_results/streaming_results.csv",
            "streaming_results.csv", 
            "latest/streaming_results.csv"
        ],
        "_search_locations_help": "File locations to search in priority order",
        
        "min_rate_for_processing": 0.1, 
        "_min_rate_for_processing_help": "Minimum results/minute to consider actively processing",
        
        "accelerating_threshold": 1.2,
        "_accelerating_threshold_help": "Rate multiplier threshold for 'accelerating' status",
        
        "slowing_threshold": 0.8,
        "_slowing_threshold_help": "Rate multiplier threshold for 'slowing' status"
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"📄 Sample configuration file created: {filename}")
        print(f"💡 Edit this file to customize monitoring behavior")
    except Exception as e:
        print(f"❌ Failed to create sample config file: {e}")


if __name__ == "__main__":
    # Create sample configuration file when run directly
    create_sample_config_file()
    
    # Demonstrate configuration usage
    print("\n🔧 Configuration Demo:")
    config = MonitorConfig()
    
    print(f"Default refresh interval: {config.refresh_interval}s")
    print(f"Default failures to show: {config.failures_to_show}")
    print(f"Search locations: {len(config.search_locations)} configured")
    
    warnings = config.validate()
    if warnings:
        print(f"⚠️  Configuration warnings: {warnings}")
    else:
        print("✅ Configuration validation passed")