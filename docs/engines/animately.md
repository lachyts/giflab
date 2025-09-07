# Animately – Engine Reference

Animately is a cross-platform GIF compression engine written in C++ with optional WASM bindings for in-browser usage.  GifLab uses the **command-line launcher** in native environments and may call the WASM API inside notebooks / JS demos.

---
## 1 Releases & versioning

• **Compression engine tags** `compression/v{major}.{minor}.{patch}`  
• **Video-conversion tags**  `video/v{major}.{minor}.{patch}`

Create and push a new release:
```bash
# example – bump compression engine to 1.1.1
git tag compression/v1.1.1
git push --tags
```
GitHub CI then builds artefacts for Linux, macOS, Windows and uploads them to the release page.

---
## 2 Building from source

### 2.1 Native build (gcc / clang)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # optional Debug
make -j$(nproc)
# run unit-tests (if ENABLE_TESTS was left ON – default)
make test
```

### 2.2 Emscripten / WASM build
```bash
emcmake cmake ..
make -j$(nproc)
```
Add `-DENABLE_TESTS=OFF` to skip the C++ tests for faster builds.

---
## 3 CLI launcher usage

After building you have a binary called **`animately`**.  All parameters are **optional**; unspecified ones leave the corresponding property unchanged.

**Current version:** 1.1.20.0

### 🚨 **CRITICAL: Flag-Based Arguments Required**

Animately **requires explicit flag-based arguments** for all operations. Positional arguments will cause silent failures.

#### ❌ **WRONG** - Will fail silently (exit code 1/2):
```bash
animately input.gif output.gif --lossy 60
animately source.gif compressed.gif
```

#### ✅ **CORRECT** - Always use flags:
```bash
animately --input input.gif --output output.gif --lossy 60
animately --input source.gif --output compressed.gif
```

| Flag | Short | Value type | Description |
|------|-------|-----------|-------------|
| `--input` | `-i` | file path | Path to input gif file |
| `--output` | `-o` | file path | Path to output gif file |
| `--scale` | `-s` | value | Scale |
| `--crop` | `-c` | value | Crop |
| `--lossy` | `-l` | `0–100` | Lossy |
| `--advanced-lossy` | `-a` | JSON file | Advanced lossy |
| `--delay` | `-d` | value | Delay |
| `--trim-frames` | `-t` | value | Trim |
| `--trim-ms` | `-m` | value | Trim in milliseconds |
| `--reduce` | `-f` | value | Reduce frames |
| `--colors` | `-p` | `1–256` | Reduce palette colors |
| `--frames` | `-g` | value | Frames |
| `--zoom` | `-z` | value | Zoom |
| `--tone` | `-u` | value | Duotone |
| `--repeated-frame` | `-r` | (flag) | Repeated Frame |
| `--meta` | `-e` | value | Gif meta information |
| `--loops` | `-y` | int | Loops in output gif |
| `--help` | `-h` | (flag) | List of available options |

**Wrapper mapping** in GifLab:
* `AnimatelyColorReducer` → `--colors {count}`
* `AnimatelyFrameReducer` → `--reduce {ratio}` (or `--trimByFrames`)
* `AnimatelyLossyCompressor` → `--lossy {level}`

The helper function constructs exactly these flags, measures runtime, and returns the metadata dict.

### 3.1 Advanced Lossy Compression

The `--advanced-lossy` (or `-a`) flag allows for more sophisticated compression by accepting a JSON configuration file that specifies individual frame properties and global settings.

**Usage:**
```bash
animately --input source.gif --output compressed.gif --advanced-lossy config.json
```

**JSON Configuration Format:**
```json
{
  "lossy": 60,
  "colors": 64,
  "frames": [
    {"png": "path/to/frame1.png", "delay": 150},
    {"png": "path/to/frame2.png", "delay": 200},
    {"png": "path/to/frame3.png", "delay": 150}
  ]
}
```

**Configuration Properties:**
- `lossy` (0-100): Global lossy compression level
- `colors` (1-256): Color palette size for the output GIF
- `frames`: Array of frame objects, each containing:
  - `png`: Path to the PNG file for this frame
  - `delay`: Frame delay in milliseconds

**Use Cases:**
- Creating GIFs from multiple PNG files
- Converting single PNG files to single-frame GIFs
- Fine-grained control over per-frame timing
- Batch processing with consistent compression settings

### 3.2 Effectiveness of Advanced Lossy PNG Sequence Input

The advanced lossy mode with PNG sequence input provides significant advantages over traditional GIF-to-GIF compression pipelines for many content types:

**Key Benefits:**
- **Superior compression for gradients:** Up to 22% smaller file sizes compared to direct GIF processing while maintaining visual quality
- **Dramatic size reduction for static content:** Significant reduction in file size for animations with static or text-heavy content
- **Better color fidelity:** PNG intermediate format preserves color information, enabling more effective palette optimization
- **Improved stability:** More reliable processing across various dithering methods compared to direct GIF compression

**Content-Specific Performance:**
- **Gradient animations:** Consistently smaller output with better visual quality
- **Mixed content:** Better handling of animations combining different visual elements
- **Static/text animations:** Exceptional compression ratios with minimal quality loss
- **Geometric animations:** May sometimes favor traditional GIF processing, depending on complexity

**Pipeline Comparison:**
- Traditional: `GIF → palette reduction → GIF → animately`
- Advanced: `GIF → palette reduction → PNG sequence → animately --advanced-lossy`

The PNG sequence approach enables the compression engine to perform more intelligent frame optimization and palette reduction, particularly beneficial for modern web animations requiring efficient size reduction.

---
## 4 JavaScript / WASM API

After `emcmake` build you obtain `animately_gif.js`, `animately_gif.wasm`, and type-definitions.

```javascript
const AnimatelyGif = require('./animately_gif');

(async () => {
  const engine = await AnimatelyGif();
  const data = new Uint8Array(/* GIF bytes */);
  const gif = engine.decode(data);

  const opts = { lossy: 80, reduce: 0.5, colors: 64 };
  engine.process(gif, opts, () => {/* progress cb */});

  const result = engine.encode(gif); // Uint8Array of new GIF
})();
```

**Options object keys (all optional):**
* `crop`, `zoom`, `scale`, `lossy`, `repeatedFrame`, `delay`, `trimByFrames`, `trimByMs`, `reduce`, `colors`, `frames`, `loops`, `tone`, `threads` – see README excerpt above for detailed meaning.

---
## 5 CI services
* **Travis-CI / AppVeyor** – compile & run unit-tests on every commit if enabled.  
* **Coveralls / Codecov** – code-coverage upload when token present.

---
## 6 Environment variables in GifLab
| Variable | Description |
|----------|-------------|
| `GIFLAB_ANIMATELY_PATH` | Override auto-detected path to `animately` launcher. |

### Common Usage Examples

```bash
# Basic compression
animately --input source.gif --output compressed.gif

# Lossy compression with quality setting
animately --input source.gif --output lossy.gif --lossy 60

# Color palette reduction
animately --input source.gif --output reduced.gif --colors 64

# Scaling and cropping
animately --input source.gif --output processed.gif --scale 0.5 --crop "10,10,200,200"

# Combined operations
animately --input source.gif --output final.gif --lossy 50 --colors 128 --scale 0.8
```

### Troubleshooting

**Silent failures (exit code 1/2):**
- **Cause**: Using positional arguments instead of flags
- **Solution**: Always use `--input` and `--output` flags

**"No input file path provided":**
- **Cause**: Missing `--input` flag  
- **Solution**: Use `animately --input file.gif --output out.gif`

**Version management:**
- Multiple versions available in `/Users/lachlants/bin/`
- Recommended: `1.1.20.0` (Sep 2025) with enhanced logging
- Check version: `animately --help` (--version not available in older versions)

---
For in-depth details see the source tree and README inside the Animately repository. 