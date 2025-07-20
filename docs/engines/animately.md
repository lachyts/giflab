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

After building you have a binary called **`animately-gif`** (or simply `animately` depending on install).  All parameters are **optional**; unspecified ones leave the corresponding property unchanged.

| Flag | Value type | Description |
|------|-----------|-------------|
| `--lossy=N` | `0–100` | Lossy compression level (0 = lossless). |
| `--reduce=R` | `0–1` (float) | Remove the given *percentage* of frames. |
| `--colors=C` | `1–256` | Quantize to C colors (per-GIF palette). |
| `--crop x y w h` | ints | Crop rectangle before further ops. |
| `--zoom x y w h` | ints | Zoom into rectangle *before* crop & scale. |
| `--scale sx sy` | floats | Scale factors after cropping. |
| `--delay F` | float | Multiply all frame delays by F (speed up / slow down). |
| `--repeatedFrame` | (flag) | Detect and drop identical frames. |
| `--trimByFrames a b` | ints | Keep frames `[a, b]` inclusive. |
| `--trimByMs a b` | ints | Keep time-range `[a, b]` ms. |
| `--loops N` | int | Set animation loop count (`0` = infinite). |
| `--tone r1 g1 b1 r2 g2 b2` | ints | Apply duotone mapping. |
| `--threads` | (flag) | Enable multithreading (must build with threads). |

**Wrapper mapping** in GifLab:
* `AnimatelyColorReducer` → `--colors {count}`
* `AnimatelyFrameReducer` → `--reduce {ratio}` (or `--trimByFrames`)
* `AnimatelyLossyCompressor` → `--lossy {level}`

The helper function constructs exactly these flags, measures runtime, and returns the metadata dict.

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
| `GIFLAB_ANIMATELY_PATH` | Override auto-detected path to `animately-gif` launcher. |

---
For in-depth details see the source tree and README inside the Animately repository. 