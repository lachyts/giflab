# Binary Tools Directory

This directory contains platform-specific binaries for external tools that are not available through standard package managers.

## Structure

```
bin/
├── README.md                 # This file
├── darwin/                   # macOS binaries
│   └── arm64/               # Apple Silicon (M1/M2/M3)
│       └── animately        # Animately compression tool v1.1.20.0
├── linux/                   # Linux binaries 
│   └── x86_64/              # Intel/AMD 64-bit
│       ├── PLACEHOLDER.md   # Instructions for Linux binary
│       └── animately        # Linux version (from compression-engine releases)
└── windows/                 # Windows binaries
    └── x86_64/              # Intel/AMD 64-bit  
        ├── PLACEHOLDER.md   # Instructions for Windows binary
        └── animately.exe    # Windows version (from compression-engine releases)
```

## Animately Integration

**Animately** is an in-house GIF compression tool developed by the Animately team.

### Current Status
- **macOS ARM64**: ✅ **Available** (`bin/darwin/arm64/animately`) - v1.1.20.0
- **Linux x86_64**: 🔄 **Ready for Binary** (`bin/linux/x86_64/animately`) - Download from releases
- **Windows x86_64**: 🔄 **Ready for Binary** (`bin/windows/x86_64/animately.exe`) - Download from releases  
- **macOS Intel**: ❓ Future support (would go in `bin/darwin/x86_64/`)

### Binary Source
Additional Animately binaries are available from:
**https://github.com/Animately/compression-engine/releases/tag/compression-latest**

Download the appropriate binary for your platform and place it in the corresponding directory structure.

### CI Integration

The CI workflows automatically detect and install the appropriate Animately binary:

1. **macOS CI runner (ARM64)**: Uses `bin/darwin/arm64/animately`
2. **Ubuntu CI runner (x86_64)**: Skips Animately (not compatible)
3. **Docker workflow**: Could use Linux version if available

### Adding New Architectures

To add support for additional platforms:

1. **Obtain binary**: Build or acquire Animately for the target platform
2. **Place in structure**: Add to appropriate `bin/<platform>/<arch>/` directory
3. **Update CI**: Modify `.github/workflows/ci.yml` installation steps
4. **Test**: Ensure tool discovery works for the new platform

### Usage in Code

The tool discovery system (`src/giflab/system_tools.py`) automatically finds Animately:

1. **Environment variable**: `$GIFLAB_ANIMATELY_PATH` (highest priority)
2. **Repository binary**: Checks `bin/<platform>/<arch>/animately`
3. **System PATH**: Falls back to system installation
4. **Graceful degradation**: Skips if not found

### Binary Information

**`bin/darwin/arm64/animately`**:
- **Version**: 1.1.20.0
- **Size**: ~1.76 MB
- **Architecture**: Apple Silicon (ARM64)
- **Compatibility**: macOS 11.0+ (Big Sur and later)

---

**Note**: These binaries are included in the repository for CI/CD purposes. For local development, you may prefer to install Animately in your system PATH or use environment variables to point to your preferred installation. 