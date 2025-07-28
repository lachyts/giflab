# 🧪 Experimental Scripts

Utility scripts for the experimental testing framework and pipeline elimination process.

## 📊 Pipeline Elimination Monitoring

### Scripts

- **`monitor_elimination.py`**: Core monitoring script for pipeline elimination progress
- **`monitor_elimination_enhanced.py`**: Enhanced monitoring with detailed status information

### Usage

```bash
# From project root directory

# Basic monitoring
poetry run python scripts/experimental/monitor_elimination.py

# Enhanced monitoring (recommended)
poetry run python scripts/experimental/monitor_elimination_enhanced.py
```

### Features

#### Core Monitoring (`monitor_elimination.py`)
- Real-time progress tracking with job counts
- Success/failure rate statistics
- Estimated completion time with adaptive rate calculation
- Recent results display with SSIM quality metrics
- Dynamic job calculation based on current configuration

#### Enhanced Monitoring (`monitor_elimination_enhanced.py`)
- All core monitoring features
- Detailed configuration display
- Applied fixes status tracking
- Testing matrix visualization
- Universal percentage mapping details:
  - **60%** → Gifsicle: 180, Others: 60
  - **100%** → Gifsicle: 300, Others: 100
- Failure status with database integration
- User-friendly controls and instructions

## 🎯 Current Configuration

The monitoring scripts display the current pipeline elimination configuration:

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

These monitoring scripts work with the main pipeline elimination process:

```bash
# Start pipeline elimination
poetry run python -m giflab eliminate-pipelines

# In another terminal: monitor progress
poetry run python scripts/experimental/monitor_elimination_enhanced.py
```

The monitoring scripts automatically detect the active elimination process and provide real-time updates without interfering with the main process. 