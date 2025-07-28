#!/usr/bin/env python3
"""Enhanced Pipeline Elimination Monitor with Fix Status."""

import json
import time
import os
import subprocess
from pathlib import Path
from datetime import datetime


def run_monitor():
    """Monitor the elimination progress with enhanced status."""
    while True:
        # Clear screen
        os.system('clear')
        
        # Header
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f" 🕐 {current_time} - Pipeline Elimination Monitor [ENHANCED v3]")
        print("━" * 60)
        
        # Run the base monitoring
        try:
            # Use the correct path to the monitoring script
            script_path = Path(__file__).parent / "monitor_elimination.py"
            result = subprocess.run(['poetry', 'run', 'python', str(script_path)], 
                                  capture_output=True, text=True, timeout=10)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"⚠️  Monitor warnings: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⚠️  Monitor timeout - progress file may be locked")
        except Exception as e:
            print(f"❌ Monitor error: {e}")
        
        print()
        
        # Enhanced status information
        print("🎯 CURRENT CONFIGURATION:")
        print("   • Frame reduction: 50% (0.5 ratio)")
        print("   • Color testing: 32, 128 colors")
        print("   • Lossy testing: 60%, 100% (universal percentages)")
        print("   • Full synthetic GIF suite (25+ test cases)")
        print()
        
        print("🔧 RECENT FIXES APPLIED:")
        print("   ✅ Engine-specific lossy mapping (Gifsicle 0-300, others 0-100)")
        print("   ✅ Universal percentage representation for lossy levels")
        print("   ✅ Frame dimension normalization for Gifski compatibility")
        print("   ✅ Enhanced failure tracking and automatic error recovery")
        print("   ✅ Cleared 159 stale failures from database")
        print()
        
        print("📊 TESTING MATRIX:")
        print("   • 2×2 parameter combinations (colors × lossy)")
        print("   • Systematic engine comparison")
        print("   • Content-type specific analysis")
        print("   • Universal percentage mapping: 60% → Gifsicle:180, Others:60")
        print("   • Universal percentage mapping: 100% → Gifsicle:300, Others:100")
        print()
        
        # Check for any new failures
        try:
            cache_db = Path("elimination_results/pipeline_results_cache.db")
            if cache_db.exists():
                result = subprocess.run([
                    'sqlite3', str(cache_db), 
                    "SELECT COUNT(*) FROM pipeline_failures;"
                ], capture_output=True, text=True)
                if result.stdout.strip().isdigit():
                    failure_count = int(result.stdout.strip())
                    if failure_count == 0:
                        print("✅ FAILURE STATUS: No failures detected")
                    else:
                        print(f"⚠️  FAILURE STATUS: {failure_count} new failures detected")
                        print("   Run 'poetry run python -m giflab debug-failures --summary' for details")
        except Exception:
            print("📊 FAILURE STATUS: Unable to check (database may be busy)")
        
        print()
        print("💡 CONTROLS:")
        print("   • Press Ctrl+C to stop monitoring")
        print("   • View detailed failures: poetry run python -m giflab debug-failures")
        print("   • Clear fixed failures: poetry run python -m giflab debug-failures --clear-fixed")
        
        # Sleep for 30 seconds
        print(f"\n⏱️  Next update in 30 seconds...")
        time.sleep(30)


if __name__ == "__main__":
    try:
        run_monitor()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        print("💾 Pipeline elimination continues running in background")
        print("🔍 Check results in elimination_results/latest/ when complete") 