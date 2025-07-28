# 🛠️ Scripts Directory

This directory contains utility scripts organized by category to keep the project root clean and maintainable.

## 📁 Directory Structure

```
scripts/
├── experimental/           # Experimental testing utilities
│   ├── monitor_elimination.py              # Core pipeline elimination monitor
│   └── monitor_elimination_enhanced.py     # Enhanced monitor with status info
└── README.md              # This file
```

## 🧪 Experimental Scripts (`experimental/`)

Scripts related to the experimental testing framework and pipeline elimination process.

### Pipeline Elimination Monitoring

- **`monitor_elimination.py`**: Core monitoring script for pipeline elimination progress
- **`monitor_elimination_enhanced.py`**: Enhanced monitoring with detailed status and configuration info

#### Usage

```bash
# Run basic monitoring
poetry run python scripts/experimental/monitor_elimination.py

# Run enhanced monitoring (recommended)
poetry run python scripts/experimental/monitor_elimination_enhanced.py
```

#### Features

- Real-time progress tracking with completion estimates
- Universal percentage display for lossy levels (60%, 100%)
- Engine-specific mapping visualization (Gifsicle: 0-300, Others: 0-100)
- Failure detection and status reporting
- Recent results display with SSIM quality metrics

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