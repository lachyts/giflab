# Troubleshooting Guide: Targeted Experiment Presets

This guide helps resolve common issues encountered when using the targeted experiment presets system.

## Quick Diagnosis

### Is it working at all?
```bash
# Test basic functionality
poetry run python -m giflab experiment --preset quick-test --output-dir troubleshoot_test

# If this fails, check installation and dependencies
# If this succeeds, the issue is with your specific configuration
```

### Check preset availability
```bash
# List all available presets
poetry run python -m giflab experiment --list-presets

# If no presets are listed, there's a module loading issue
# If presets are listed, check your preset name spelling
```

---

## Configuration Errors

### Error: "Unknown preset: 'preset-name'"

**Cause**: Misspelled or non-existent preset name.

**Solutions**:
```bash
# Check exact preset names
poetry run python -m giflab experiment --list-presets

# Common misspellings:
# Wrong: frame_focus, frame-removal, color_optimization
# Right: frame-focus, color-optimization, lossy-quality-sweep
```

**Available Preset Names**:
- `frame-focus`
- `color-optimization` 
- `lossy-quality-sweep`
- `tool-comparison-baseline`
- `dithering-focus`
- `png-optimization`
- `quick-test`

### Error: "Invalid slot name: 'slot-name'"

**Cause**: Using incorrect slot names in `--variable-slot` or `--lock-slot`.

**Valid Slot Names**: `frame`, `color`, `lossy`

**Common Mistakes**:
```bash
# Wrong slot names
--variable-slot frames=*           # Should be 'frame'
--lock-slot colour=ffmpeg-color    # Should be 'color'
--variable-slot compression=*      # Should be 'lossy'
```

**Correct Usage**:
```bash
--variable-slot frame=*
--lock-slot color=ffmpeg-color
--variable-slot lossy=*
```

### Error: "At least one slot must be variable"

**Cause**: All three slots are locked, leaving nothing to compare.

**Problem Example**:
```bash
# This won't work - nothing varies
--lock-slot frame=animately-frame \
--lock-slot color=ffmpeg-color \
--lock-slot lossy=none-lossy
```

**Solutions**:
```bash
# Make at least one slot variable
--variable-slot frame=* \
--lock-slot color=ffmpeg-color \
--lock-slot lossy=none-lossy

# Or make multiple slots variable
--variable-slot frame=* \
--variable-slot color=* \
--lock-slot lossy=none-lossy
```

### Error: "Cannot mix --preset with custom slot options"

**Cause**: Using both `--preset` and custom slot configuration simultaneously.

**Problem Example**:
```bash
# This conflicts
--preset frame-focus --variable-slot color=*
```

**Solutions**:
```bash
# Use preset only
--preset frame-focus

# OR use custom slots only (no preset)
--variable-slot frame=* --lock-slot color=ffmpeg-color --lock-slot lossy=none-lossy
```

---

## Tool and Implementation Errors

### Error: "Implementation 'tool-name' not found for slot 'slot-name'"

**Cause**: Invalid or misspelled tool implementation name.

**Diagnosis**:
```bash
# Check available tools programmatically
poetry run python -c "
from giflab.capability_registry import tools_for
print('Frame tools:', [t.NAME for t in tools_for('frame_reduction')])
print('Color tools:', [t.NAME for t in tools_for('color_reduction')])
print('Lossy tools:', [t.NAME for t in tools_for('lossy_compression')])
"
```

**Common Tool Name Issues**:
```bash
# Wrong tool names (common mistakes)
--lock-slot frame=animately              # Should be 'animately-frame'
--lock-slot color=ffmpeg                 # Should be 'ffmpeg-color'
--lock-slot lossy=gifsicle               # Should be 'gifsicle-lossy'
--lock-slot color=floyd-steinberg        # Should be 'ffmpeg-color-floyd'
```

**Correct Tool Names by Category**:

**Frame Tools**:
- `animately-frame`, `ffmpeg-frame`, `gifsicle-frame`, `imagemagick-frame`, `none-frame`

**Color Tools**:
- `animately-color`, `ffmpeg-color`, `gifsicle-color`, `imagemagick-color`, `none-color`
- Dithering variants: `ffmpeg-color-floyd`, `ffmpeg-color-bayer2`, `imagemagick-color-riemersma`

**Lossy Tools**:
- `animately-lossy`, `animately-advanced-lossy`, `ffmpeg-lossy`, `gifski-lossy`, `gifsicle-lossy`, `none-lossy`

### Error: "No tools found in scope"

**Cause**: Variable slot scope resolves to no available tools.

**Common Causes**:
```bash
# Misspelled tools in scope
--variable-slot frame=animaly-frame,ffmeg-frame  # Typos

# Non-existent tool combinations
--variable-slot color=photoshop-color           # Tool doesn't exist
```

**Solutions**:
```bash
# Use wildcard for all tools
--variable-slot frame=*

# Or check exact tool names first
poetry run python -c "
from giflab.capability_registry import tools_for
for tool in tools_for('frame_reduction'):
    print(f'  {tool.NAME}')
"
```

---

## Parameter Format Errors

### Error: "Invalid parameter format: 'param-spec'"

**Cause**: Incorrect format in `--slot-params`.

**Required Format**: `param:value` or `param:[value1,value2,value3]`

**Common Format Mistakes**:
```bash
# Wrong formats
--slot-params colors=32              # Missing colon, should be colors:32
--slot-params color:colors:32        # Extra slot name, should be colors:32
--slot-params ratios:1.0,0.8,0.5     # List without brackets
--slot-params ratios:[1.0, 0.8, 0.5] # Spaces in list (sometimes problematic)
```

**Correct Formats**:
```bash
# Single values
--slot-params colors:32
--slot-params level:40
--slot-params ratio:0.8

# List values (no spaces)
--slot-params ratios:[1.0,0.8,0.5,0.2]
--slot-params colors:[128,64,32,16]
--slot-params levels:[0,40,80,120]
```

### Error: "Invalid parameter value"

**Cause**: Parameter value is outside valid range or wrong type.

**Parameter Validation Rules**:
- **Frame ratios**: Must be 0.0-1.0
- **Color counts**: Must be 1-256 (integer)
- **Lossy levels**: Must be 0-200 (integer)

**Common Value Errors**:
```bash
# Out of range values
--slot-params ratios:[1.5,0.8]       # 1.5 > 1.0 maximum
--slot-params colors:300              # 300 > 256 maximum  
--slot-params level:-10               # Negative values not allowed
```

**Valid Value Examples**:
```bash
--slot-params ratios:[1.0,0.8,0.5,0.2]    # All between 0.0-1.0
--slot-params colors:[256,128,64,32,16,8]  # All between 1-256
--slot-params levels:[0,20,40,60,80]       # All between 0-200
```

---

## Runtime and Performance Issues

### Issue: "Experiment takes too long"

**Cause**: Configuration generates too many pipeline combinations.

**Diagnosis**: Check estimated pipeline count before running:
```python
# Check pipeline count programmatically
from giflab.experimental.targeted_presets import PRESET_REGISTRY
from giflab.experimental.targeted_generator import TargetedPipelineGenerator

preset = PRESET_REGISTRY.get("your-preset-name")
generator = TargetedPipelineGenerator()
validation = generator.validate_preset_feasibility(preset)
print(f"Estimated pipelines: {validation['estimated_pipelines']}")
```

**Solutions**:
```bash
# Use smaller presets for testing
--preset quick-test                    # ~2 pipelines
--preset frame-focus                   # ~5 pipelines

# Limit pipeline count in custom configurations
--variable-slot frame=animately-frame,ffmpeg-frame  # Only 2 tools instead of all

# Use targeted GIFs for faster testing
--use-targeted-gifs
```

### Issue: "Out of memory during experiment"

**Cause**: Too many concurrent operations or large pipeline combinations.

**Solutions**:
```bash
# Use caching to reduce memory usage
--use-cache

# Use targeted GIF set
--use-targeted-gifs

# Run smaller experiments
--preset quick-test

# Reduce quality threshold (less processing per pipeline)
--quality-threshold 0.1
```

### Issue: "GPU acceleration not working"

**Cause**: GPU not available, drivers missing, or incompatible setup.

**Diagnosis**:
```bash
# Test without GPU first
poetry run python -m giflab experiment --preset quick-test

# Then test with GPU
poetry run python -m giflab experiment --preset quick-test --use-gpu
```

**Solutions**:
- Ensure GPU drivers are installed and up to date
- Check CUDA/OpenCL availability for your system
- Run without `--use-gpu` if GPU acceleration isn't essential
- Check system requirements for GPU-accelerated tools

---

## Output and File Issues

### Issue: "Permission denied writing to output directory"

**Cause**: Insufficient permissions or directory doesn't exist.

**Solutions**:
```bash
# Use a directory you have write access to
--output-dir ~/results

# Create directory manually first
mkdir -p /path/to/output/dir
poetry run python -m giflab experiment --preset frame-focus --output-dir /path/to/output/dir

# Use relative path in current directory
--output-dir ./results
```

### Issue: "No results generated" or "Empty output directory"

**Cause**: Experiment failed silently or results saved elsewhere.

**Diagnosis Steps**:
```bash
# Check if experiment actually ran
poetry run python -m giflab experiment --preset quick-test --output-dir debug_test -v

# Look for error messages in verbose output
# Check if output directory was created
ls -la debug_test/

# Verify preset is generating pipelines
poetry run python -c "
from giflab.experimental.runner import ExperimentalRunner
runner = ExperimentalRunner()
pipelines = runner.generate_targeted_pipelines('quick-test')
print(f'Generated {len(pipelines)} pipelines')
"
```

### Issue: "Results inconsistent between runs"

**Cause**: Non-deterministic behavior or caching issues.

**Solutions**:
```bash
# Clear cache and rerun
rm -rf ~/.cache/giflab*  # Or wherever cache is stored
poetry run python -m giflab experiment --preset frame-focus --output-dir fresh_test

# Use consistent random seeds (if applicable)
# Disable caching for reproducibility testing
# Remove --use-cache flag
```

---

## Installation and Environment Issues

### Issue: "Module not found: giflab.experimental"

**Cause**: Installation incomplete or Python path issues.

**Solutions**:
```bash
# Reinstall dependencies
poetry install

# Verify installation
poetry run python -c "import giflab.experimental.targeted_presets; print('OK')"

# Check Python path
poetry run python -c "import sys; print(sys.path)"

# Try explicit path
PYTHONPATH=src poetry run python -m giflab experiment --list-presets
```

### Issue: "Poetry command not found"

**Cause**: Poetry not installed or not in PATH.

**Solutions**:
```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Or use pip
pip install poetry

# Or run directly with python
python -m giflab experiment --preset quick-test
```

### Issue: "Dependencies missing or conflicting"

**Cause**: Environment setup issues or version conflicts.

**Solutions**:
```bash
# Clean environment and reinstall
poetry env remove python
poetry install

# Check dependency status
poetry show

# Update dependencies
poetry update

# Create fresh virtual environment
poetry shell
```

---

## Advanced Troubleshooting

### Enable Debug Logging

```bash
# Enable verbose output
poetry run python -m giflab experiment --preset quick-test -v

# Enable debug logging in Python
export GIFLAB_LOG_LEVEL=DEBUG
poetry run python -m giflab experiment --preset quick-test

# Or programmatically
poetry run python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from giflab.experimental.runner import ExperimentalRunner
runner = ExperimentalRunner()
pipelines = runner.generate_targeted_pipelines('quick-test')
print(f'Debug: Generated {len(pipelines)} pipelines')
"
```

### Test Individual Components

```python
# Test preset loading
from giflab.experimental.targeted_presets import PRESET_REGISTRY
print(f"Available presets: {list(PRESET_REGISTRY.list_presets().keys())}")

# Test pipeline generation
from giflab.experimental.targeted_generator import TargetedPipelineGenerator
generator = TargetedPipelineGenerator()
preset = PRESET_REGISTRY.get("quick-test")
validation = generator.validate_preset_feasibility(preset)
print(f"Validation: {validation}")

# Test tool availability
from giflab.capability_registry import tools_for
print(f"Frame tools: {len(list(tools_for('frame_reduction')))}")
print(f"Color tools: {len(list(tools_for('color_reduction')))}")
print(f"Lossy tools: {len(list(tools_for('lossy_compression')))}")
```

### Performance Profiling

```bash
# Profile memory usage
poetry run python -m memory_profiler -m giflab experiment --preset quick-test

# Profile execution time
time poetry run python -m giflab experiment --preset quick-test

# Profile with cProfile
poetry run python -m cProfile -o profile.stats -m giflab experiment --preset quick-test
```

---

## Getting Additional Help

### Self-Diagnosis Checklist

Before seeking help, try these steps:

1. **Basic functionality**: Does `--preset quick-test` work?
2. **Tool availability**: Can you list available tools?
3. **Preset listing**: Does `--list-presets` show results?
4. **Permissions**: Can you write to the output directory?
5. **Dependencies**: Are all required packages installed?
6. **Configuration**: Are you using valid slot names and tool names?

### Collect Diagnostic Information

When reporting issues, include:

```bash
# System information
poetry --version
poetry run python --version
poetry run python -c "import platform; print(platform.platform())"

# Package versions
poetry show | grep giflab

# Environment test
poetry run python -c "
import giflab.experimental.targeted_presets as tp
import giflab.experimental.targeted_generator as tg
print(f'Presets available: {len(tp.PRESET_REGISTRY.list_presets())}')
generator = tg.TargetedPipelineGenerator()
print('Generator created successfully')
"

# Failed command with verbose output
poetry run python -m giflab experiment --preset quick-test --output-dir diagnostic_test -v
```

### Common Resolution Patterns

1. **Configuration errors** → Usually fixed by checking syntax and valid values
2. **Tool name errors** → Check exact tool names with diagnostic commands
3. **Performance issues** → Use smaller presets and targeted GIFs for testing
4. **Installation issues** → Clean reinstall with `poetry install`
5. **Permission issues** → Use directories you own with write access

### Further Resources

- **User Guide**: `/docs/user-guides/targeted-experiment-presets.md`
- **CLI Reference**: `/docs/reference/cli-reference.md`
- **Preset Types**: `/docs/reference/preset-types-reference.md`
- **GitHub Issues**: Report bugs or request help on the project repository
- **Development Team**: Contact via project communication channels

Most issues can be resolved by carefully checking command syntax and ensuring all tool names and parameters are correctly specified.