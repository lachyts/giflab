# 🛠️ Scripts Directory

This directory contains utility scripts organized by category to keep the project root clean and maintainable.

## 📁 Directory Structure

```
scripts/
├── experimental/           # Experimental testing utilities
│   ├── simple_monitor.py              # Pipeline elimination monitor
│   └── README.md                      # Experimental scripts documentation
└── README.md              # This file
```

## 🧪 Experimental Scripts (`experimental/`)

Scripts related to the experimental testing framework and pipeline elimination process.

### Pipeline Elimination Monitoring

- **`simple_monitor.py`**: Pipeline elimination progress monitor with real-time updates and configurable options

#### Usage

```bash
# Run basic monitoring (30s refresh, 3 recent failures, auto-detect file)
python scripts/experimental/simple_monitor.py

# Custom refresh interval and failure count
python scripts/experimental/simple_monitor.py --refresh 10 --failures 5

# Custom results file location
python scripts/experimental/simple_monitor.py --file my_results.csv

# Quick monitoring for active development
python scripts/experimental/simple_monitor.py -r 5 -f 2
```

#### Features

- **Real-time progress tracking** with completion estimates and success rates
- **Dynamic job estimation** based on actual pipeline configuration  
- **Configurable refresh intervals** (1-60 seconds) and failure display count
- **Universal percentage display** for lossy levels (60%, 100%)
- **Engine-specific mapping** visualization (Gifsicle: 0-300, Others: 0-100)
- **Recent failure analysis** with pipeline and GIF name details
- **Robust CSV parsing** with proper error handling and auto-detection
- **Auto-detection of results files** - searches multiple common locations

## 🎯 Why This Organization?

1. **Clean Root Directory**: Keeps utility scripts out of the project root
2. **Logical Grouping**: Scripts are organized by their purpose and domain
3. **Maintainability**: Easy to find and maintain related scripts
4. **Documentation**: Clear structure for new contributors

## 📝 Adding New Scripts

When adding new utility scripts:

1. **Choose the right category**: Place scripts in appropriate subdirectories
2. **Document purpose**: Add clear docstrings and usage examples
3. **Update this README**: Add entries for new scripts or categories
4. **Consider integration**: Think about how scripts fit into existing workflows

For experimental testing scripts, add them to `experimental/`. For other categories, create new subdirectories as needed (e.g., `data-processing/`, `analysis/`, etc.). 