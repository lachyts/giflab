# Testing Best Practices

## 🎯 Overview

This document establishes clear guidelines for testing in GifLab to maintain a clean, professional codebase and prevent root directory pollution.

## 📁 Directory Structure

### Proper Testing Locations

```
giflab/
├── tests/                    # ✅ Unit & integration tests (pytest)
│   ├── fixtures/            # Test GIF files for automated tests
│   └── test_*.py           # Automated test files
├── test-workspace/          # ✅ Manual testing & debugging
│   ├── manual/             # Manual testing sessions
│   ├── debug/              # Debug investigations  
│   ├── temp/               # Temporary files (auto-cleaned)
│   └── samples/            # Test samples & reference files
└── docs/guides/            # ✅ Testing documentation
```

### ❌ NEVER Put Testing Files In:
- **Root directory** (`/`) - Keep it clean!
- **src/** - Source code only
- **docs/** - Documentation only

## 🔧 Testing Workflows

### 1. **Unit/Integration Tests** (`tests/`)
```bash
# Add new test files
pytest tests/test_new_feature.py

# Run all tests  
pytest
```

### 2. **Manual Testing** (`test-workspace/manual/`)
```bash
# Create session directory
mkdir test-workspace/manual/feature-debugging-YYYYMMDD
cd test-workspace/manual/feature-debugging-YYYYMMDD

# Run your tests
python -m giflab experiment sample.gif

# Clean up when done (see cleanup section)
```

### 3. **Debug Investigations** (`test-workspace/debug/`)
```bash
# Create debug session
mkdir test-workspace/debug/pipeline-issue-investigation
cd test-workspace/debug/pipeline-issue-investigation

# Run debugging
python debug_script.py
```

## 🧹 Cleanup Protocols

### 1. **Auto-Cleanup** (temp files)
The `test-workspace/temp/` directory is automatically cleaned:
- Files older than 7 days
- Run `make clean-temp` or automated daily

### 2. **Manual Cleanup** (after testing)
```bash
# After completing manual testing session
cd test-workspace/manual/
rm -rf completed-session-name/

# Keep only active investigations
```

### 3. **Quarterly Cleanup**
- Review `test-workspace/` quarterly
- Archive important findings to `docs/analysis/`
- Delete outdated debug sessions

## 📋 Testing Session Checklist

### Before Starting Testing:
- [ ] Create properly named directory in `test-workspace/`
- [ ] Use descriptive session names: `feature-name-YYYYMMDD` or `bug-investigation-brief-desc`
- [ ] Document purpose in session README if investigation is complex

### After Completing Testing:
- [ ] Move important findings to appropriate `docs/` location
- [ ] Delete temporary files and unsuccessful experiments
- [ ] Clean up test session directory
- [ ] Update any relevant documentation

## 🚨 Emergency Cleanup

If you find testing files polluting the root directory:

```bash
# Run the cleanup script
make clean-testing-mess

# Or manually with confirmation
python scripts/clean_testing_workspace.py --interactive
```

## 🎯 Best Practices Summary

1. **Plan Before Testing** - Choose the right directory structure
2. **Use Descriptive Names** - `debug-ffmpeg-corruption-20240728` not `test1`
3. **Clean As You Go** - Don't let temporary files accumulate  
4. **Document Important Findings** - Move insights to `docs/`
5. **Respect the Root** - Keep it professional and clean

## 🔧 Tooling Support

### Makefile Targets
```bash
make clean-temp          # Clean temporary test files
make clean-testing-mess  # Emergency cleanup of root pollution
make test-workspace      # Create proper test workspace structure
```

### VS Code/Cursor Integration
- Add `test-workspace/temp/` to `.gitignore`
- Configure workspace to suggest proper testing locations
- Use workspace snippets for testing directory creation

## 📝 Examples

### ✅ Good Testing Session
```bash
mkdir test-workspace/debug/gifski-memory-leak-20240728
cd test-workspace/debug/gifski-memory-leak-20240728
echo "Investigating gifski memory usage with large GIFs" > README.md
# ... run tests ...
# Document findings in docs/troubleshooting/
rm -rf ../gifski-memory-leak-20240728
```

### ❌ Bad Testing Practice  
```bash
# DON'T DO THIS - pollutes root directory
cd /
python -m giflab experiment sample.gif  # Creates files in root
mkdir debug_test                        # Clutters root
# ... leaves mess everywhere ...
```

---

*Following these practices ensures GifLab maintains a professional, navigable codebase that's easy for both humans and AI assistants to work with.* 