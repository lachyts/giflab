# Test Fixtures

This directory contains minimal GIF fixtures for testing the external engine implementations.

## Fixtures

- **`simple_4frame.gif`** (747B) - 4 frames, 16 colors, 64×64px for basic functionality tests
- **`single_frame.gif`** (306B) - 1 frame, 8 colors, 32×32px for edge case testing  
- **`many_colors.gif`** (13KB) - 4 frames, 256 colors, 64×64px for palette stress testing

**These GIFs are committed to the repository** to ensure consistent, deterministic testing across all environments.

## Generation

Run `python generate_fixtures.py` to regenerate the test fixtures.

⚠️ **Important**: If you modify the generation script, regenerate the fixtures and commit the updated GIFs to maintain test consistency.

## Maintenance

- **Total size**: ~24KB (acceptable for essential test infrastructure)
- **Update policy**: Only regenerate when test requirements change
- **Validation**: New fixtures should be validated by running the full test suite

## Usage in Tests

The fixtures are used by:
- `test_external_engines.py` - Unit tests for helper functions
- `test_wrapper_integration.py` - Integration tests for wrappers
- `test_engine_smoke.py` - Updated smoke tests with real functionality validation

## Test Categories

### Unit Tests (No External Tools Required)
```bash
# Parameter validation tests
pytest tests/test_external_engines.py::TestParameterValidation -v

# Common utility tests  
pytest tests/test_external_engines.py::test_run_command_success -v
```

### Integration Tests (Require External Tools)
```bash
# ImageMagick tests (require ImageMagick)
pytest tests/test_external_engines.py::TestImageMagickHelpers -m external_tools

# FFmpeg tests (require FFmpeg)
pytest tests/test_external_engines.py::TestFFmpegHelpers -m external_tools

# gifski tests (require gifski + ImageMagick)
pytest tests/test_external_engines.py::TestGifskiHelpers -m external_tools

# Cross-engine consistency
pytest tests/test_wrapper_integration.py::TestCrossEngineConsistency -m external_tools
```

### Smoke Tests (Updated for New Engines)
```bash
# Run all smoke tests (skips unavailable engines)
pytest tests/test_engine_smoke.py -v

# Run only if specific engines are available
pytest tests/test_engine_smoke.py -m external_tools
```

## Expected Behavior

**Without External Tools:**
- Parameter validation tests should pass
- External tool tests should be skipped with clear messages
- COMBINE_GROUP and availability method tests should pass

**With External Tools:**
- All tests should pass and validate actual functionality
- Color reduction should reduce palette sizes
- Frame reduction should reduce frame counts  
- Lossy compression should produce smaller files
- All engines should return consistent metadata schemas 