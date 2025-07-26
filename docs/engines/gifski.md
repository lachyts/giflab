# gifski – Engine Reference

> Official project: <https://github.com/ImageOptim/gifski>

`gifski` is a high-quality GIF encoder that takes a sequence of PNG frames and builds an optimised, palette-dithered animation.

Our usage in **GifLab** is intentionally minimal: just lossy compression (quality slider).  Color and frame reduction are delegated to the frame/color wrappers of other engines.

## ✨ PNG Sequence Optimization

**GifLab automatically optimizes gifski pipelines** to avoid the "single frame" error that occurred when frame reduction tools created 1-frame GIFs.

### How It Works
1. **Detection**: Pipeline detects when gifski follows FFmpeg/ImageMagick tools
2. **Direct Export**: Previous tool exports PNG sequence directly (highest quality)
3. **Bypass**: gifski receives PNG frames directly, bypassing GIF→PNG extraction
4. **Result**: No more "Only a single image file was given as input" errors

### Supported Combinations
- `FFmpegFrameReducer` → `GifskiLossyCompressor` ✅
- `FFmpegColorReducer` → `GifskiLossyCompressor` ✅  
- `ImageMagickFrameReducer` → `GifskiLossyCompressor` ✅
- `ImageMagickColorReducer` → `GifskiLossyCompressor` ✅

### Benefits
- **Eliminates single-frame errors**: Resolved 272 failures (100% of gifski issues)
- **Higher quality**: PNG frames exported at maximum quality from previous tool
- **Better performance**: Avoids double-processing of frame extraction

## Key CLI flags
| Flag | Example | Purpose |
|------|---------|---------|
| `--quality N` | `--quality 60` | Overall quality in % (maps to our `lossy_level`).|
| `-o output.gif` | | Output path |
| `input_*.png` | | Input frame sequence – generated on the fly by the helper |

Full flag list: run `gifski --help` or see the upstream [README](https://github.com/ImageOptim/gifski/blob/main/README.md).

## Wrapper strategy
1.  **Optimized Path** (when following FFmpeg/ImageMagick): Receive PNG frames directly from previous pipeline step
2.  **Fallback Path**: Split the source GIF into PNG frames using ImageMagick (small test-only GIF ⇒ cheap)
3.  Call `gifski` with `--quality {lossy_level}` and PNG frame sequence
4.  Measure runtime, compute output size, return metadata dict.

## Environment variables
| Variable | Description |
|----------|-------------|
| `GIFLAB_GIFSKI_PATH` | Override auto-discovery of the `gifski` executable.|

```text
metadata = {
  "render_ms": 123,
  "engine": "gifski",
  "command": "gifski --quality 60 -o …",
  "kilobytes": 42.0,
}
``` 