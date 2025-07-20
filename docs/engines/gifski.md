# gifski – Engine Reference

> Official project: <https://github.com/ImageOptim/gifski>

`gifski` is a high-quality GIF encoder that takes a sequence of PNG frames and builds an optimised, palette-dithered animation.

Our usage in **GifLab** is intentionally minimal: just lossy compression (quality slider).  Color and frame reduction are delegated to the frame/color wrappers of other engines.

## Key CLI flags
| Flag | Example | Purpose |
|------|---------|---------|
| `--quality N` | `--quality 60` | Overall quality in % (maps to our `lossy_level`).|
| `-o output.gif` | | Output path |
| `input_*.png` | | Input frame sequence – generated on the fly by the helper |

Full flag list: run `gifski --help` or see the upstream [README](https://github.com/ImageOptim/gifski/blob/main/README.md).

## Wrapper strategy
1.  Split the source GIF into PNG frames using Pillow (small test-only GIF ⇒ cheap).  _This step relies on the Pillow library._
2.  Call `gifski` with `--quality {lossy_level}`.
3.  Measure runtime, compute output size, return metadata dict.

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