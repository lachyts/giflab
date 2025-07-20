# ImageMagick – Engine Reference

> Official docs: <https://imagemagick.org>  
> Animation basics: <https://usage.imagemagick.org/anim_basics/>

ImageMagick’s `magick` (or legacy `convert`) command provides very flexible frame- and color-level manipulation of GIFs.

We use three operations:

| Action | Representative command | Notes |
|--------|------------------------|-------|
| **Color reduction** | `magick in.gif +dither -colors {colors} out.gif` | `colors` ≤ 256 |
| **Frame reduction**  | `magick in.gif -coalesce -delete '1--2' -layers optimize out.gif` | Wrapper will approximate ratio by deleting every _n_-th frame. |
| **Lossy compression** | `magick in.gif -sampling-factor 4:2:0 -strip -quality {q} out.gif` | `q` derived from `lossy_level`. |

Exact commands are centralised in the helper so tests stay stable.

## Wrapper strategy
* Single-call pipeline (no temp files) wherever possible.
* Run via `subprocess.run(..., check=True)`; capture `stderr` for debugging.

## Environment variables
| Variable | Description |
|----------|-------------|
| `GIFLAB_IMAGEMAGICK_PATH` | Path to the `magick` executable (if not on `$PATH`).|

---
For full CLI reference see `man magick` or the [animation basics guide](https://usage.imagemagick.org/anim_basics/). 