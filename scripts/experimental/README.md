# 🧪 Experimental Scripts

Utility scripts for the experimental testing framework and pipeline elimination process.

## 📊 Pipeline Elimination Monitoring

### Monitor Script

**`simple_monitor.py`** - Pipeline elimination progress monitor with real-time updates

### Usage

```bash
# From project root directory

# Basic monitoring with default settings (30s refresh, auto-detect file)
python scripts/experimental/simple_monitor.py

# Custom refresh interval (10 seconds)
python scripts/experimental/simple_monitor.py --refresh 10

# Show more recent failures (5 instead of default 3)
python scripts/experimental/simple_monitor.py --failures 5

# Custom results file location
python scripts/experimental/simple_monitor.py --file custom_results.csv

# Combine all options
python scripts/experimental/simple_monitor.py --refresh 15 --failures 2 --file my_results.csv
```

### Features

- **Real-time progress tracking** - Live updates of test completion status
- **Dynamic job estimation** - Calculates remaining work based on actual pipeline configuration
- **Configurable refresh intervals** - Set update frequency from 1-60 seconds
- **Recent failure analysis** - Shows recent failed pipeline combinations
- **Success rate monitoring** - Tracks overall elimination success percentage
- **Robust CSV parsing** - Uses Python's csv module for reliable data parsing
- **Auto-detection of results files** - Searches multiple common locations for results
- **Configurable results file path** - Specify custom location for results CSV

## 🎯 Current Configuration

The monitoring script displays the current pipeline elimination configuration:

- **Frame reduction**: 50% (0.5 ratio) - consistent across all tests
- **Color testing**: 32, 128 colors - two palette sizes
- **Lossy testing**: 60%, 100% - universal percentages mapped to engine ranges
- **Test matrix**: 2×2 combinations (colors × lossy) = 4 parameter sets
- **Synthetic GIFs**: 25+ diverse test cases covering different content types

## 🔧 Engine-Specific Lossy Mapping

The system uses universal percentages that are automatically mapped to engine-specific ranges:

| Engine | Range | 60% Maps To | 100% Maps To |
|--------|-------|-------------|--------------|
| Gifsicle | 0-300 | 180 | 300 |
| Animately | 0-100 | 60 | 100 |
| FFmpeg | 0-100 | 60 | 100 |
| Gifski | 0-100 | 60 | 100 |
| ImageMagick | 0-100 | 60 | 100 |

This ensures consistent percentage-based representation while respecting each engine's native parameter ranges.

## 🚀 Integration with Pipeline Elimination

The monitoring script works seamlessly with the main pipeline elimination process:

```bash
# Start pipeline elimination in one terminal
poetry run python -m giflab eliminate-pipelines

# Monitor progress in another terminal
python scripts/experimental/simple_monitor.py

# Or use custom settings
python scripts/experimental/simple_monitor.py --refresh 5 --failures 10
```

The monitor automatically detects the active elimination process and provides real-time updates without interfering with the main process.

## 📋 Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--refresh` | `-r` | 30 | Refresh interval in seconds |
| `--failures` | `-f` | 3 | Number of recent failures to display |

### Examples

```bash
# Quick updates every 5 seconds
python scripts/experimental/simple_monitor.py -r 5

# Show 10 recent failures with 20s refresh
python scripts/experimental/simple_monitor.py -r 20 -f 10

# Minimal updates for long-running jobs
python scripts/experimental/simple_monitor.py --refresh 60 --failures 1
``` 