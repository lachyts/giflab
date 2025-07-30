# Experimental Scripts

This directory contains experimental and utility scripts for the Animately[[memory:3631063]] GIF compression project.

## Scripts

### `simple_monitor.py` - Enhanced Pipeline Elimination Monitor

A real-time monitoring tool for pipeline elimination runs with enhanced batching awareness and configuration support.

**Key Features:**
- ✅ **Real-time progress tracking** with batching-aware updates
- 📈 **Processing rate calculation** and trend analysis  
- ⏱️ **ETA estimates** when actively processing
- 🔍 **Failure analysis** with recent error examples
- 💡 **Batching status detection** (explains when updates pause)
- ⚙️ **Configuration support** with JSON config files and environment variables
- 🧠 **Dynamic job estimation** from logs and metadata
- 📊 **Enhanced cache performance tracking**

**Why Updates Come in Batches:**
The pipeline elimination system uses performance-optimized batching:
- Results are written every 15-25 completed tests (not individually)
- This is **normal behavior** - not a monitoring bug!
- Large jumps (e.g., +25 results) are expected and healthy

**Usage:**
```bash
# Basic monitoring (uses default config)
python scripts/experimental/simple_monitor.py

# Create sample configuration file
python scripts/experimental/simple_monitor.py --create-config

# Use custom configuration file
python scripts/experimental/simple_monitor.py --config monitor_config.json

# Override specific settings (command line overrides config)
python scripts/experimental/simple_monitor.py --refresh 10 --failures 5

# Monitor custom results file
python scripts/experimental/simple_monitor.py --file path/to/custom_results.csv

# Environment variable configuration
MONITOR_REFRESH_INTERVAL=15 python scripts/experimental/simple_monitor.py
```

**Configuration Support:**
The monitor now supports flexible configuration through:
- **JSON config files** (`--config path/to/config.json`)
- **Environment variables** (e.g., `MONITOR_REFRESH_INTERVAL=15`)
- **Command line overrides** (highest priority)

Generate a sample config file with `--create-config` to see all available options.

**Sample Output:**
```
🕐 14:23:45 - Pipeline Elimination Monitor (Enhanced)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 Pipeline Elimination Progress - 🔄 RUNNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Tests Completed: 1,247
✅ Successful: 1,089
❌ Failed: 158
📈 Success Rate: 87.3%
⚡ Processing Rate: 18.5 results/min 📈 Accelerating
🎯 Currently testing: complex_gradient
📁 Results file: elimination_results/latest/streaming_results.csv
📋 Estimated Progress: 13.4% (8,253 remaining, ETA: 447min)
📊 Estimate based on: ~9,500 total jobs

🔄 BATCHING INFO:
• Results are batched every 15-25 tests for performance
• Large jumps in counts are normal and expected  
• Processing rate shows actual progress between batches
```

**Command Line Options:**
- `--refresh, -r`: Update interval in seconds (uses config default if not specified)
- `--failures, -f`: Number of recent failures to show (uses config default if not specified)  
- `--file, -F`: Custom path to results CSV file (auto-detects by default)
- `--config, -c`: Path to JSON configuration file
- `--create-config`: Create a sample configuration file and exit

**Environment Variables:**
- `MONITOR_REFRESH_INTERVAL`: Set refresh interval (seconds)
- `MONITOR_FAILURES_TO_SHOW`: Set number of failures to display
- `MONITOR_BUFFER_SIZE`: Set internal buffer size
- `MONITOR_ESTIMATED_TOTAL_JOBS`: Set estimated total jobs for progress
- `MONITOR_BASE_TIME_PER_JOB`: Set estimated seconds per job for ETA
- `MONITOR_MIN_PROCESSING_RATE`: Set minimum rate for "actively processing" status

**Monitoring Tips:**
- **Batch Updates**: Don't worry about pauses - results come in batches every 15-25 tests
- **Processing Rate**: Shows actual work being done between updates
- **Trend Analysis**: 📈 Accelerating, ➡️ Steady, 📉 Slowing, ⏸️ Batching
- **ETA Estimates**: Only shown when actively processing with measurable rate

**File Detection:**
Automatically searches these locations:
- `elimination_results/latest/streaming_results.csv` (most common)
- `elimination_results/streaming_results.csv`
- `streaming_results.csv` 
- `latest/streaming_results.csv` 