# CI Setup Documentation

This document describes the Continuous Integration (CI) setup for the GifLab project, which supports testing with and without external tools.

## Overview

The CI system provides multiple testing tiers to balance speed and comprehensiveness:

1. **Fast Tests** - Quick feedback for basic functionality
2. **Core Tests** - Full test suite without external tools  
3. **External Tools Tests** - Complete integration testing with all engines
4. **Lint and Type Check** - Code quality validation
5. **macOS Compatibility** - Cross-platform validation

## Workflow Files

### Main CI Workflow (`.github/workflows/ci.yml`)

The primary CI workflow runs on every push and pull request to `main` and `develop` branches.

#### Job Breakdown:

**1. Fast Tests (`fast-tests`)**
- **Purpose**: Provide rapid feedback (< 5 minutes)
- **Scope**: Tests marked with `@pytest.mark.fast`
- **Matrix**: Python 3.11 and 3.12
- **Tools**: No external dependencies required

**2. Core Tests (`core-tests`)**
- **Purpose**: Comprehensive testing without external tool dependencies
- **Scope**: All tests except those marked with `@pytest.mark.external_tools`  
- **Matrix**: Python 3.11 and 3.12
- **Tools**: No external dependencies required

**3. External Tools Tests (`external-tools-tests`)**
- **Purpose**: Full integration testing with all supported engines
- **Scope**: All tests, including `@pytest.mark.external_tools`
- **Platform**: Ubuntu only (single Python 3.11)
- **Tools Installed**:
  - ImageMagick (via `apt-get`)
  - FFmpeg (via `apt-get`) 
  - gifsicle (via `apt-get`)
  - gifski (via `cargo install`)
  - Animately (skipped - not compatible with Ubuntu x86_64)

**4. Lint and Type Check (`lint-and-type-check`)**
- **Purpose**: Code quality validation
- **Tools**: ruff, black, mypy
- **Platform**: Ubuntu, Python 3.11

**5. macOS Tests (`macos-tests`)**
- **Purpose**: Cross-platform compatibility validation
- **Scope**: Core tests + available external tools
- **Tools**: Homebrew-installed tools + repository Animately binary

### Docker External Tools Workflow (`.github/workflows/docker-external-tools.yml`)

A supplementary Docker-based workflow for comprehensive external tool testing.

#### Features:
- **Trigger**: Manual dispatch and weekly schedule
- **Environment**: Controlled Docker container with all tools pre-installed
- **Purpose**: Catch external tool compatibility issues over time
- **Tools**: All engines installed in isolated environment

## External Tool Integration

### Tool Installation Strategy

**Ubuntu (GitHub Actions runners):**
```bash
# System package manager
sudo apt-get install imagemagick ffmpeg gifsicle

# Rust ecosystem
cargo install gifski

# Custom/Manual (placeholder)
# Animately installation to be added
```

**macOS (Homebrew + Repository):**
```bash
brew install imagemagick ffmpeg gifsicle
cargo install gifski  # if available
# Animately available from repository bin/darwin/arm64/animately
chmod +x bin/darwin/arm64/animately
```

**Docker (Comprehensive):**
```dockerfile
RUN apt-get install imagemagick ffmpeg gifsicle cargo rustc
RUN cargo install gifski
# All tools guaranteed available
```

### Test Marker Strategy

The CI system uses pytest markers to organize tests:

- `@pytest.mark.fast` - Quick tests for rapid feedback
- `@pytest.mark.external_tools` - Tests requiring external engines
- No marker - Core functionality tests

### Tool Discovery Integration

The CI workflows use GifLab's `system_tools` module for dynamic tool discovery:

```python
from giflab.system_tools import get_available_tools, verify_required_tools

# Check what's available
available = get_available_tools()

# Verify requirements (gracefully handles missing tools)
verify_required_tools()
```

**Tool Discovery Priority:**
1. Environment variable (e.g., `$GIFLAB_ANIMATELY_PATH`)
2. Repository binary (`bin/<platform>/<arch>/tool`)
3. System PATH
4. Graceful failure

## Environment Variables

The CI supports the same environment variable overrides as local development:

- `GIFLAB_IMAGEMAGICK_PATH` - Custom ImageMagick path
- `GIFLAB_FFMPEG_PATH` - Custom FFmpeg path  
- `GIFLAB_FFPROBE_PATH` - Custom FFprobe path
- `GIFLAB_GIFSKI_PATH` - Custom gifski path
- `GIFLAB_GIFSICLE_PATH` - Custom gifsicle path
- `GIFLAB_ANIMATELY_PATH` - Custom Animately path (when available)

## Coverage Reporting

Each test tier generates separate coverage reports:

- `coverage-fast.xml` - Fast tests coverage
- `coverage-core.xml` - Core tests coverage  
- `coverage-external.xml` - External tools tests coverage
- `coverage-complete.xml` - Complete test suite coverage

These are uploaded to Codecov with appropriate flags for analysis.

## Performance Considerations

### Caching Strategy

The workflows use Poetry's virtual environment caching to reduce setup time:

```yaml
- name: Load cached venv
  uses: actions/cache@v3
  with:
    path: .venv
    key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
```

### Parallel Execution

- **Fast tests**: Run in parallel across Python 3.11 and 3.12
- **Core tests**: Run in parallel across Python 3.11 and 3.12  
- **External tools**: Single job (tool installation overhead)
- **Lint/Type**: Independent parallel job

## Failure Handling

### Graceful Degradation

- Missing tools are detected and handled gracefully
- Tests skip unavailable engines rather than failing
- macOS workflow allows tool installation failures

### Error Reporting

- Tool verification steps provide detailed version information
- Test failures include comprehensive tracebacks (`--tb=short`)
- Coverage reports help identify untested code paths

## Local Development

### Running CI Tests Locally

```bash
# Fast tests (like CI fast-tests job)
python -m pytest -m fast -v

# Core tests (like CI core-tests job)  
python -m pytest -m "not external_tools" -v

# External tools tests (requires tools installed)
python -m pytest -m external_tools -v

# Complete test suite
python -m pytest -v
```

### Tool Installation for Local Testing

Follow the CI installation steps for your platform:

**Ubuntu/Debian:**
```bash
sudo apt-get install imagemagick ffmpeg gifsicle cargo rustc
cargo install gifski
```

**macOS:**
```bash
brew install imagemagick ffmpeg gifsicle
cargo install gifski
```

### Environment Variable Testing

```bash
# Test custom tool paths
export GIFLAB_IMAGEMAGICK_PATH="/custom/path/to/magick"
python -m pytest tests/test_system_tools.py -v
```

## Maintenance

### Adding New External Tools

1. **Update workflow tool installation steps**
2. **Add tool to `_FALLBACK_TOOLS` in `system_tools.py`**
3. **Create wrapper in `tool_wrappers.py`**
4. **Add integration tests with `@pytest.mark.external_tools`**
5. **Update this documentation**

### Monitoring CI Health

- **Fast tests** should complete in < 5 minutes
- **External tools tests** should complete in < 15 minutes
- **Weekly Docker workflow** provides long-term stability monitoring
- **Coverage reports** track test completeness over time

## Troubleshooting

### Common Issues

**Tool Installation Failures:**
- Check platform-specific installation commands
- Verify tool availability in package manager
- Use environment variables for custom paths

**Test Failures in External Tools Job:**
- Verify tool versions are compatible
- Check for tool-specific parameter changes
- Review tool discovery logic in `system_tools.py`

**Performance Issues:**
- Check if external tools are taking too long (30s timeout)
- Review test fixture sizes and complexity
- Consider parallelization for independent tests

---

This CI setup provides comprehensive validation while maintaining development velocity through tiered testing and intelligent tool discovery. 