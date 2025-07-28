#!/usr/bin/env python3
"""Pipeline Elimination Monitor - No complex imports, just CSV parsing."""

import csv
import time
import os
import argparse
from pathlib import Path
from datetime import datetime


def find_results_file(custom_path: str = None) -> Path:
    """Find the streaming results CSV file in multiple possible locations.
    
    Args:
        custom_path: Custom path provided by user
        
    Returns:
        Path to results file if found
        
    Raises:
        FileNotFoundError: If no results file found in any location
    """
    if custom_path:
        custom_file = Path(custom_path)
        if custom_file.exists():
            return custom_file
    
    # Try multiple common locations
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
    return Path("elimination_results/latest/streaming_results.csv")


def calculate_estimated_total_jobs():
    """Calculate estimated total jobs dynamically instead of hard-coded value."""
    try:
        # Try to read from pipeline configuration or progress file
        progress_locations = [
            Path("elimination_results/latest/elimination_progress.json"),
            Path("elimination_results/elimination_progress.json"),
            Path("elimination_progress.json"),
        ]
        
        for progress_file in progress_locations:
            if progress_file.exists():
                import json
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    # Look for total job estimate in progress data
                    if 'estimated_total_jobs' in progress_data:
                        return progress_data['estimated_total_jobs']
        
        # Try to estimate from running process or configuration
        # This is a rough estimate based on typical elimination runs
        # Default synthetic GIFs: 25, typical pipelines: ~200-500, test params: ~4
        return 25 * 300 * 4  # Conservative middle estimate
        
    except Exception:
        # Fallback to reasonable default
        return 10000


def monitor_elimination(refresh_interval=30, show_recent_failures=3, results_file_path=None):
    """Monitor pipeline elimination by reading CSV files directly.
    
    Args:
        refresh_interval: Seconds between updates (default: 30)
        show_recent_failures: Number of recent failures to show (default: 3)
        results_file_path: Custom path to results file (optional)
    """
    while True:
        # Clear screen
        os.system('clear')
        
        # Header
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f" 🕐 {current_time} - Pipeline Elimination Monitor")
        print("━" * 60)
        
        # Find streaming results file
        try:
            results_file = find_results_file(results_file_path)
        except FileNotFoundError:
            results_file = Path("elimination_results/latest/streaming_results.csv")
        
        if not results_file.exists():
            print("❌ No streaming results file found. Is the elimination running?")
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
                    
                    # Get most recent test
                    recent_gif = "unknown"
                    if len(lines) > 1:
                        last_row = lines[-1]
                        if len(last_row) > 0:
                            recent_gif = last_row[0]
                    
                    print("🔬 Pipeline Elimination Progress")
                    print("━" * 40)
                    print(f"📊 Total Tests Completed: {total:,}")
                    print(f"✅ Successful: {successful:,}")
                    print(f"❌ Failed: {failed:,}")
                    print(f"📈 Success Rate: {success_rate:.1f}%")
                    print(f"🎯 Currently testing: {recent_gif}")
                    print(f"📁 Results file: {results_file}")
                    
                    # Progress estimate (dynamically calculated)
                    estimated_total = calculate_estimated_total_jobs()
                    progress_pct = (
                        (total / estimated_total * 100) if estimated_total > 0 else 0
                    )
                    remaining = estimated_total - total
                    
                    print(f"📋 Estimated Progress: {progress_pct:.1f}% "
                          f"({remaining:,} remaining)")
                    print(f"📊 Estimate based on: ~{estimated_total:,} total jobs")
                    
                    # Show recent failures if any
                    if failed > 0:
                        print("\n🔍 Recent Failure Patterns:")
                        failure_rows = [
                            row for row in lines[-50:] 
                            if len(row) > 3 and row[3].strip() == 'False'
                        ]
                        if failure_rows:
                            # Last N failures
                            for row in failure_rows[-show_recent_failures:]:
                                if len(row) > 2:
                                    gif_name = row[0]
                                    pipeline = (
                                        row[2].replace('_', ' ')[:50] + "..."
                                    )
                                    print(f"   ❌ {gif_name} | {pipeline}")
                    
            except Exception as e:
                print(f"❌ Error reading results: {e}")
                print(f"📁 File path: {results_file}")
        
        print("\n💡 CONTROLS:")
        print("   • Press Ctrl+C to stop monitoring")
        print("   • Check detailed results: elimination_results/latest/")
        print(f"   • Next update in {refresh_interval} seconds...")
        
        # Sleep for configured interval
        time.sleep(refresh_interval)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Pipeline Elimination Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_monitor.py                    # Default: 30s refresh
  python simple_monitor.py --refresh 10      # 10s refresh interval
  python simple_monitor.py --failures 5      # Show 5 recent failures
  python simple_monitor.py --file custom_results.csv  # Custom results file
        """
    )
    
    parser.add_argument(
        '--refresh', '-r',
        type=int,
        default=30,
        help='Refresh interval in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--failures', '-f',
        type=int,
        default=3,
        help='Number of recent failures to show (default: 3)'
    )
    
    parser.add_argument(
        '--file', '-F',
        type=str,
        default=None,
        help='Custom path to results CSV file (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.refresh < 1:
        print("Error: Refresh interval must be at least 1 second")
        return 1
        
    if args.failures < 0:
        print("Error: Number of failures to show must be non-negative")
        return 1
    
    file_info = f" from {args.file}" if args.file else " (auto-detect)"
    print(f"🚀 Starting pipeline monitor "
          f"(refresh: {args.refresh}s, failures: {args.failures}{file_info})")
    
    try:
        monitor_elimination(
            refresh_interval=args.refresh, 
            show_recent_failures=args.failures,
            results_file_path=args.file
        )
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        print("💾 Pipeline elimination continues running in background")
        print("🔍 Check results in elimination_results/latest/ when complete")
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main()) 