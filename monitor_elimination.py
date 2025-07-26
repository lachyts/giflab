#!/usr/bin/env python3
"""Monitor pipeline elimination progress."""

import json
import time
from pathlib import Path


def monitor_progress():
    """Monitor the elimination progress and display stats."""
    results_dir = Path("elimination_results")
    # Look for progress file in the latest symlink directory
    progress_file = results_dir / "latest" / "elimination_progress.json"
    
    if not progress_file.exists():
        print("❌ No progress file found. Is the elimination running?")
        return
    
    # Load progress data with retry mechanism
    progress_data = None
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with open(progress_file) as f:
                # Read the entire file content
                content = f.read().strip()
                if not content:
                    if attempt == max_retries - 1:
                        print("⚠️  Progress file is empty, elimination may be starting...")
                        return
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                
                # Parse as JSON object (not lines)
                progress_data = json.loads(content)
                break  # Success, exit retry loop
                
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                print(f"⚠️  Progress file is being written to, unable to read after {max_retries} attempts")
                print(f"     Last error: {e}")
                return
            # Brief delay before retry
            time.sleep(0.2 + attempt * 0.1)  # Increasing delay
            continue
            
        except Exception as e:
            print(f"❌ Error reading progress file: {e}")
            return
    
    if progress_data is None:
        print("❌ Unable to read progress data after multiple attempts")
        return
    
    # Count completed jobs
    completed_jobs = len(progress_data)
    
    # Use the correct total for full strategy (116,875)
    total_jobs = 116875
    
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
    
    # Estimate remaining time (using observed rate of ~237 jobs/minute)
    remaining_jobs = total_jobs - completed_jobs
    est_minutes = remaining_jobs / 237  # observed rate from earlier testing
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
