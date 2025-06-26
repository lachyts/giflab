# 🎞️ GifLab

GIF compression and analysis laboratory for systematic performance evaluation.

## Overview

GifLab analyzes GIF compression by generating a grid of variants with different:

- **Frame keep ratios**: 1.00, 0.90, 0.80, 0.70, 0.50
- **Palette sizes**: 256, 128, 64 colors  
- **Lossy levels**: 0, 40, 120
- **Engines**: gifsicle, animately

Each variant is measured for file size, SSIM quality, and render time.

## Quick Start

```bash
# Install dependencies (requires Poetry)
poetry install

# Run compression analysis
poetry run python -m giflab run data/raw data/

# Add AI-generated tags
poetry run python -m giflab tag results.csv data/raw
```

## Project Structure

```
giflab/
├─ data/                 # Data directories
├─ src/giflab/           # Python package
├─ notebooks/            # Analysis notebooks  
├─ tests/                # Test suite
└─ pyproject.toml        # Poetry configuration
```

## Development Status

This project is being developed in stages:

- **✅ S0**: Repo scaffold, Poetry, black/ruff, pytest
- **🚧 S1**: Metadata extraction + SHA + file-name; tests
- **⏳ S2-S10**: Additional functionality (see PROJECT_SCOPE.md)

## Requirements

- Python 3.11+
- Poetry for dependency management
- gifsicle and animately-cli for compression
- FFmpeg for video processing

## Cross-Platform Setup

### macOS
```bash
brew install python@3.11 ffmpeg gifsicle
# Install animately-cli binary to PATH
```

**Engine Paths:**
- Animately engine: `/Users/lachlants/bin/launcher`
- Gifsicle: `gifsicle`

### Windows/WSL
```bash
choco install python ffmpeg gifsicle
# Or use WSL2 with Linux setup
```

## License

MIT License - see LICENSE file for details.