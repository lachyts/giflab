#!/usr/bin/env python3
"""Monitor pipeline elimination progress."""

import json
import time
import fcntl
import errno
from pathlib import Path


def calculate_expected_total_jobs():
    """Calculate expected total jobs dynamically based on current configuration."""
    try:
        # Import here to avoid circular imports and ensure latest config
        from src.giflab.dynamic_pipeline import generate_all_pipelines
        from src.giflab.pipeline_elimination import PipelineEliminator
        
        # Create temporary eliminator to get configuration
        eliminator = PipelineEliminator()
        
        # Get synthetic GIF count (default configuration)
        synthetic_gifs_count = len(eliminator.synthetic_specs)
        
        # Get all pipelines count
        all_pipelines = generate_all_pipelines()
        pipelines_count = len(all_pipelines)
        
        # Get test parameters count
        test_params_count = len(eliminator.test_params)
        
        total_jobs = synthetic_gifs_count * pipelines_count * test_params_count
        
        return total_jobs, {
            'synthetic_gifs': synthetic_gifs_count,
            'pipelines': pipelines_count,
            'test_params': test_params_count
        }
        
    except Exception as e:
        print(f"⚠️  Could not calculate dynamic total jobs: {e}")
        # Fallback to reasonable estimate
        return 100000, {
            'synthetic_gifs': 25,
            'pipelines': 1000,
            'test_params': 4,
            'note': 'Fallback estimate - could not access configuration'
        }


def estimate_processing_rate(completed_jobs: int, progress_data: dict):
    """Estimate processing rate adaptively based on recent performance."""
    if completed_jobs < 10:
        # Not enough data for good estimate, use conservative default
        return 120.0  # jobs per minute
    
    # Try to calculate rate from timestamps in progress data if available
    try:
        timestamps = []
        for job_data in progress_data.values():
            if 'timestamp' in job_data:
                timestamps.append(job_data['timestamp'])
            elif 'error_timestamp' in job_data:
                timestamps.append(job_data['error_timestamp'])
        
        if len(timestamps) >= 5:
            # Sort timestamps and calculate rate from recent jobs
            import datetime
            timestamps.sort()
            recent_timestamps = timestamps[-10:]  # Last 10 jobs
            
            if len(recent_timestamps) >= 2:
                start_time = datetime.datetime.fromisoformat(recent_timestamps[0])
                end_time = datetime.datetime.fromisoformat(recent_timestamps[-1])
                time_diff_minutes = (end_time - start_time).total_seconds() / 60.0
                
                if time_diff_minutes > 0:
                    recent_rate = (len(recent_timestamps) - 1) / time_diff_minutes
                    return max(60.0, min(300.0, recent_rate))  # Clamp between 1-5 jobs/sec
        
    except Exception:
        pass  # Fall through to default calculation
    
    # Adaptive rate based on total completed jobs and reasonable assumptions
    if completed_jobs < 100:
        return 150.0  # Initial slower rate
    elif completed_jobs < 1000:
        return 200.0  # Moderate rate  
    else:
        return 250.0  # Optimized rate for bulk processing


def monitor_progress():
    """Monitor the elimination progress and display stats."""
    results_dir = Path("elimination_results")
    # Look for progress file in the latest symlink directory
    progress_file = results_dir / "latest" / "elimination_progress.json"
    
    if not progress_file.exists():
        print("❌ No progress file found. Is the elimination running?")
        return
    
    # Load progress data with robust file locking and retry mechanism
    progress_data = _read_progress_file_safely(progress_file)
    
    if progress_data is None:
        print("❌ Unable to read progress data after multiple attempts")
        return
    
    # Count completed jobs
    completed_jobs = len(progress_data)
    
    # Calculate expected total jobs dynamically
    total_jobs, job_breakdown = calculate_expected_total_jobs()
    
    # Calculate stats
    successful_jobs = sum(1 for job_data in progress_data.values()
                         if job_data.get('success', False))
    failed_jobs = completed_jobs - successful_jobs
    
    # Check for additional failures in database that might not be in progress file
    try:
        import sqlite3
        cache_db = Path("elimination_results/pipeline_results_cache.db")
        if cache_db.exists():
            with sqlite3.connect(cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_failures")
                db_failures = cursor.fetchone()[0]
                if db_failures > failed_jobs:
                    # Use database count if it's higher (more accurate)
                    failed_jobs = db_failures
    except Exception:
        # If database check fails, use the progress file count
        pass
    
    success_rate = (successful_jobs / completed_jobs * 100) if completed_jobs > 0 else 0
    
    # Progress percentage
    progress_pct = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    
    # Estimate remaining time using adaptive rate calculation
    remaining_jobs = total_jobs - completed_jobs
    estimated_rate = estimate_processing_rate(completed_jobs, progress_data)
    est_minutes = remaining_jobs / estimated_rate if estimated_rate > 0 else 0
    hours = int(est_minutes // 60)
    minutes = int(est_minutes % 60)
    
    # Progress bar
    bar_width = 30
    filled_width = int(bar_width * progress_pct / 100)
    bar = "█" * filled_width + "░" * (bar_width - filled_width)
    
    # Display results
    print("🔬 Pipeline Elimination Progress")
    print("━" * 40)
    print(f"📊 Progress: {completed_jobs:,} / {total_jobs:,} jobs ({progress_pct:.1f}%)")
    print(f"✅ Successful: {successful_jobs:,}")
    print(f"❌ Failed: {failed_jobs:,}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"Progress: |{bar}| {progress_pct:.1f}%")
    print(f"⏱️  Estimated time remaining: {hours}h {minutes}m")
    print(f"🎯 Current rate: {estimated_rate:.1f} jobs/minute")
    
    # Show job breakdown if available
    if 'note' not in job_breakdown:
        print(f"📋 Job Matrix: {job_breakdown['synthetic_gifs']} GIFs × {job_breakdown['pipelines']} pipelines × {job_breakdown['test_params']} params")
    
    # Recent results (last 3)
    print("\n📋 Recent Results (last 3):")
    recent_items = list(progress_data.items())[-3:]
    for _job_id, job_data in recent_items:
        status = "✅" if job_data.get('success', False) else "❌"
        gif_name = job_data.get('gif_name', 'unknown')
        pipeline_id = job_data.get('pipeline_id', 'unknown')
        # Truncate long pipeline names
        if len(pipeline_id) > 40:
            pipeline_id = pipeline_id[:37] + "..."
        
        if job_data.get('success', False):
            ssim = job_data.get('ssim_mean', 0)
            print(f"  {status} {gif_name} | {pipeline_id} | SSIM: {ssim:.3f}")
        else:
            error = job_data.get('error', 'unknown error')
            print(f"  {status} {gif_name} | {pipeline_id} | Error: {error}")


if __name__ == "__main__":
    monitor_progress()
