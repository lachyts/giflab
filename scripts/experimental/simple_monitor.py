#!/usr/bin/env python3
"""Pipeline Elimination Monitor - Enhanced version with batching awareness and configuration support."""

import csv
import time
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Import configuration support
try:
    from monitor_config import MonitorConfig, default_config
except ImportError:
    # Fallback if config module not available
    print("⚠️  Configuration module not found, using hardcoded defaults")
    default_config = None


def find_results_file(custom_path: str = None, config: MonitorConfig = None) -> Path:
    """Find the streaming results CSV file in multiple possible locations.
    
    Args:
        custom_path: Custom path provided by user
        config: Configuration instance (uses default if None)
        
    Returns:
        Path to results file if found
        
    Raises:
        FileNotFoundError: If no results file found in any location
    """
    if custom_path:
        custom_file = Path(custom_path)
        if custom_file.exists():
            return custom_file
    
    # Get search locations from config
    if config:
        possible_locations = [Path(loc) for loc in config.get_search_locations()]
    else:
        # Fallback to hardcoded locations
        possible_locations = [
            Path("elimination_results/latest/streaming_results.csv"),
            Path("elimination_results/streaming_results.csv"),
            Path("streaming_results.csv"),
            Path("latest/streaming_results.csv"),
        ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    # If none found, return the most likely location for better error message
    return possible_locations[0] if possible_locations else Path("elimination_results/latest/streaming_results.csv")


def calculate_estimated_total_jobs(config: MonitorConfig = None):
    """Calculate estimated total jobs dynamically from logs or config.
    
    Args:
        config: Configuration instance with fallback value
        
    Returns:
        Estimated total number of jobs
    """
    try:
        # First, try to find the actual total from CLI output or logs
        log_locations = [
            Path("elimination_results/latest/"),
            Path("elimination_results/"),
        ]
        
        # Look for log files or metadata that might contain the original total
        for log_dir in log_locations:
            if log_dir.exists():
                # Check run metadata first (most reliable)
                metadata_file = log_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if 'total_jobs' in metadata:
                                return int(metadata['total_jobs'])
                    except:
                        pass
                
                # Check for log files containing job count patterns
                for file_path in log_dir.glob("*.log"):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Look for patterns like "Total jobs: 93,500" or "93,500 total pipeline combinations"
                            import re
                            patterns = [
                                r'Total jobs: ([\d,]+)',
                                r'([\d,]+) total pipeline combinations',
                                r'Starting comprehensive testing: ([\d,]+) total',
                            ]
                            for pattern in patterns:
                                match = re.search(pattern, content)
                                if match:
                                    total_str = match.group(1).replace(',', '')
                                    return int(total_str)
                    except:
                        continue
        
        # Use config fallback if available
        if config:
            return config.estimated_total_jobs
        
        # Final fallback based on typical elimination run parameters
        return 93500  # Conservative estimate for full elimination run
        
    except Exception as e:
        print(f"⚠️  Could not calculate job estimate: {e}")
        # Use config fallback or hardcoded value
        return config.estimated_total_jobs if config else 93500


# Global variables for rate tracking
last_check_time = None
last_check_count = 0
recent_rates = []


def calculate_processing_rate(current_count: int, config: MonitorConfig = None) -> tuple:
    """Calculate current processing rate and trend with configurable thresholds.
    
    Args:
        current_count: Current number of completed jobs
        config: Configuration instance for thresholds
    
    Returns:
        tuple: (current_rate, trend_description, is_processing)
    """
    global last_check_time, last_check_count, recent_rates
    
    current_time = datetime.now()
    
    if last_check_time is None:
        last_check_time = current_time
        last_check_count = current_count
        return 0.0, "🔄 Initializing", True
    
    time_delta = (current_time - last_check_time).total_seconds()
    count_delta = current_count - last_check_count
    
    if time_delta > 0:
        current_rate = count_delta / time_delta * 60  # Results per minute
        recent_rates.append(current_rate)
        
        # Keep configurable number of measurements for trend analysis
        max_history = config.rate_history_size if config else 10
        if len(recent_rates) > max_history:
            recent_rates.pop(0)
        
        # Update for next calculation
        last_check_time = current_time
        last_check_count = current_count
        
        # Determine trend and processing status using config thresholds
        avg_rate = sum(recent_rates) / len(recent_rates) if recent_rates else 0
        
        if config:
            is_processing = config.is_actively_processing(current_rate)
            trend = config.get_trend_description(current_rate, avg_rate)
        else:
            # Fallback to hardcoded values
            is_processing = current_rate > 0.1
            if current_rate > avg_rate * 1.2:
                trend = "📈 Accelerating"
            elif current_rate < avg_rate * 0.8:
                trend = "📉 Slowing" if current_rate > 0.1 else "⏸️  Batching"
            else:
                trend = "➡️  Steady"
        
        return current_rate, trend, is_processing
    
    return 0.0, "🔄 Calculating", True


def monitor_elimination(
    refresh_interval=None, show_recent_failures=None, results_file_path=None, config=None
):
    """Monitor pipeline elimination with enhanced batching awareness and configuration support.
    
    Args:
        refresh_interval: Seconds between updates (uses config default if None)
        show_recent_failures: Number of recent failures to show (uses config default if None)
        results_file_path: Custom path to results file (optional)
        config: MonitorConfig instance (creates default if None)
    """
    # Initialize configuration
    if config is None:
        config = default_config or MonitorConfig()
    
    # Use config defaults for None values
    if refresh_interval is None:
        refresh_interval = config.refresh_interval
    if show_recent_failures is None:
        show_recent_failures = config.failures_to_show
    while True:
        # Clear screen
        os.system('clear')
        
        # Header
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f" 🕐 {current_time} - Pipeline Elimination Monitor (Enhanced)")
        print("━" * 70)
        
        # Find streaming results file using config
        try:
            results_file = find_results_file(results_file_path, config)
        except FileNotFoundError:
            # Use first search location from config as fallback
            fallback_locations = config.get_search_locations() if config else ["elimination_results/latest/streaming_results.csv"]
            results_file = Path(fallback_locations[0])
        
        if not results_file.exists():
            print(
                "❌ No streaming results file found. Is the elimination running?"
            )
            print(f"   Expected: {results_file}")
            print("   Searched locations:")
            search_locations = [
                "elimination_results/latest/streaming_results.csv",
                "elimination_results/streaming_results.csv", 
                "streaming_results.csv",
                "latest/streaming_results.csv"
            ]
            for loc in search_locations:
                exists_marker = "✅" if Path(loc).exists() else "❌"
                print(f"     {exists_marker} {loc}")
        else:
            try:
                # Read the CSV file using proper CSV parsing
                with open(results_file, 'r', newline='') as f:
                    csv_reader = csv.reader(f)
                    lines = list(csv_reader)
                
                if len(lines) <= 1:
                    print("📊 No test results yet...")
                else:
                    # Count results (skip header line)
                    successful = 0
                    failed = 0
                    
                    for row in lines[1:]:  # Skip header
                        if len(row) > 3:
                            success_field = row[3].strip()
                            if success_field == 'True':
                                successful += 1
                            elif success_field == 'False':
                                failed += 1
                    
                    total = successful + failed
                    success_rate = (
                        (successful / total * 100) if total > 0 else 0
                    )
                    
                    # Calculate processing rate and trend using config
                    current_rate, trend, is_processing = calculate_processing_rate(total, config)
                    
                    # Get most recent test and check if running
                    recent_gif = "unknown"
                    is_likely_complete = False
                    if len(lines) > 1:
                        last_row = lines[-1]
                        if len(last_row) > 0:
                            recent_gif = last_row[0]
                        
                        # Check if run seems complete by looking at recent activity
                        # If last several entries are all the same GIF, likely done with that test set
                        recent_gifs = set()
                        for row in lines[-10:]:  # Check last 10 entries
                            if len(row) > 0:
                                recent_gifs.add(row[0])
                        is_likely_complete = len(recent_gifs) <= 2 and current_rate < 0.1  # Very few unique GIFs + low rate
                    
                    status_emoji = "🏁" if is_likely_complete else "🔄"
                    status_text = "COMPLETED" if is_likely_complete else "RUNNING"
                    
                    print(f"🔬 Pipeline Elimination Progress - {status_emoji} {status_text}")
                    print("━" * 50)
                    print(f"📊 Total Tests Completed: {total:,}")
                    print(f"✅ Successful: {successful:,}")
                    print(f"❌ Failed: {failed:,}")
                    print(f"📈 Success Rate: {success_rate:.1f}%")
                    
                    # Enhanced processing status
                    if current_rate > 0:
                        print(f"⚡ Processing Rate: {current_rate:.1f} results/min {trend}")
                    else:
                        print(f"⚡ Processing Rate: {trend}")
                    
                    if not is_likely_complete:
                        print(f"🎯 Currently testing: {recent_gif}")
                        if not is_processing:
                            print(f"   💡 System is likely batching results (updates every ~15-25 tests)")
                    else:
                        print(f"🎯 Last tested: {recent_gif}")
                    print(f"📁 Results file: {results_file}")
                    
                    # Progress estimate (dynamically calculated using config)
                    estimated_total = calculate_estimated_total_jobs(config)
                    progress_pct = (
                        (total / estimated_total * 100) 
                        if estimated_total > 0 else 0
                    )
                    remaining = estimated_total - total
                    
                    if not is_likely_complete and current_rate > 0:
                        estimated_time_remaining = remaining / current_rate  # minutes
                        eta_str = f", ETA: {estimated_time_remaining:.0f}min" if estimated_time_remaining < 1000 else ""
                        print(f"📋 Estimated Progress: {progress_pct:.1f}% "
                              f"({remaining:,} remaining{eta_str})")
                    elif not is_likely_complete:
                        print(f"📋 Estimated Progress: {progress_pct:.1f}% "
                              f"({remaining:,} remaining)")
                    else:
                        print(f"📋 Final Results: {total:,} total tests completed")
                    print(
                        f"📊 Estimate based on: ~{estimated_total:,} total jobs"
                    )
                    
                    # Show failure patterns 
                    if failed > 0:
                        print(f"\n🔍 Failure Analysis ({failed} total failures):")
                        
                        # First try to find recent failures (last 50 entries)
                        failure_rows = [
                            row for row in lines[-50:] 
                            if len(row) > 3 and row[3].strip() == 'False'
                        ]
                        
                        if failure_rows:
                            print("   Recent failures:")
                            for row in failure_rows[-show_recent_failures:]:
                                if len(row) >= 3:
                                    gif_name = row[0]
                                    pipeline = (
                                        row[2].replace('_', ' ')[:50] + "..."
                                        if len(row[2]) > 50 else row[2].replace('_', ' ')
                                    )
                                    print(f"   ❌ {gif_name} | {pipeline}")
                        else:
                            # No recent failures, show some earlier ones
                            all_failure_rows = [
                                row for row in lines[1:] 
                                if len(row) > 3 and row[3].strip() == 'False'
                            ]
                            if all_failure_rows:
                                print("   Example failures (from earlier in run):")
                                for row in all_failure_rows[-show_recent_failures:]:
                                    if len(row) >= 3:
                                        gif_name = row[0]
                                        pipeline = (
                                            row[2].replace('_', ' ')[:50] + "..."
                                            if len(row[2]) > 50 else row[2].replace('_', ' ')
                                        )
                                        print(f"   ❌ {gif_name} | {pipeline}")
                    
            except Exception as e:
                print(f"❌ Error reading results: {e}")
                print(f"📁 File path: {results_file}")
        
        print("\n💡 CONTROLS:")
        print("   • Press Ctrl+C to stop monitoring")
        print("   • Check detailed results: elimination_results/latest/")
        print(f"   • Next update in {refresh_interval} seconds...")
        
        # Batching explanation using config
        print("\n🔄 BATCHING INFO:")
        if config:
            batching_info = config.get_batching_info()
            print(f"   • {batching_info['description']}")
            print(f"   • {batching_info['explanation']}")
            print("   • Processing rate shows actual progress between batches")
        else:
            # Fallback to hardcoded messages
            print("   • Results are batched every 15-25 tests for performance")
            print("   • Large jumps in counts are normal and expected")
            print("   • Processing rate shows actual progress between batches")
        
        # Sleep for configured interval
        time.sleep(refresh_interval)


def main():
    """Main function with command line argument parsing and configuration support."""
    parser = argparse.ArgumentParser(
        description="Pipeline Elimination Monitor with Configuration Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_monitor.py                           # Use default config
  python simple_monitor.py --refresh 10             # Override refresh interval
  python simple_monitor.py --failures 5             # Show 5 recent failures
  python simple_monitor.py --file custom_results.csv # Custom results file
  python simple_monitor.py --config monitor.json    # Use custom config file
  python simple_monitor.py --create-config          # Create sample config file
        """
    )
    
    parser.add_argument(
        '--refresh', '-r',
        type=int,
        default=None,
        help='Refresh interval in seconds (uses config default if not specified)'
    )
    
    parser.add_argument(
        '--failures', '-f',
        type=int,
        default=None,
        help='Number of recent failures to show (uses config default if not specified)'
    )
    
    parser.add_argument(
        '--file', '-F',
        type=str,
        default=None,
        help='Custom path to results CSV file (default: auto-detect)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a sample configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        if default_config:
            from monitor_config import create_sample_config_file
            create_sample_config_file("monitor_config.json")
        else:
            print("❌ Configuration module not available")
        return 0
    
    # Initialize configuration
    config = None
    if default_config:
        if args.config:
            print(f"📄 Loading configuration from: {args.config}")
            config = MonitorConfig(args.config)
        else:
            config = default_config
            
        # Validate configuration
        warnings = config.validate()
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
    else:
        print("⚠️  Using fallback settings (configuration module not available)")
    
    # Validate command line arguments
    if args.refresh is not None and args.refresh < 1:
        print("Error: Refresh interval must be at least 1 second")
        return 1
        
    if args.failures is not None and args.failures < 0:
        print("Error: Number of failures to show must be non-negative")
        return 1
    
    # Determine final settings (command line overrides config)
    final_refresh = args.refresh if args.refresh is not None else (config.refresh_interval if config else 30)
    final_failures = args.failures if args.failures is not None else (config.failures_to_show if config else 3)
    
    file_info = f" from {args.file}" if args.file else " (auto-detect)"
    config_info = f" (config: {args.config})" if args.config else " (default config)" if config else " (no config)"
    print(f"🚀 Starting pipeline monitor{config_info}")
    print(f"   Refresh interval: {final_refresh}s")
    print(f"   Failures to show: {final_failures}")
    print(f"   Results file: {file_info}")
    
    try:
        monitor_elimination(
            refresh_interval=final_refresh, 
            show_recent_failures=final_failures,
            results_file_path=args.file,
            config=config
        )
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        print("💾 Pipeline elimination continues running in background")
        print("🔍 Check results in elimination_results/latest/ when complete")
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main()) 