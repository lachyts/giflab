# Gifsicle – Engine Reference

> Official manual: <https://www.lcdf.org/gifsicle/man.html>

`gifsicle` is a fast, versatile command-line tool for GIF creation and optimisation.  GifLab uses it as the **reference implementation** for all three single-variable compression actions:

| Action              | Primary flags used | Notes |
|---------------------|--------------------|-------|
| **Color reduction** | `--colors N` (optionally `--[no-]dither`) | `N` ≤ 256.  Our color-reduction wrapper adds the flag only when `N` is lower than the source palette. |
| **Frame reduction**  | Frame *selection* syntax `#0 #2 #4 …` | Frames to **keep** must come **after** the input file.  The wrapper calculates evenly-spaced indices for the requested ratio. |
| **Lossy compression**| `--lossy=LEVEL` (0-200) | Requires gifsicle built with the lossy patch (most modern binaries include it). |

### Optimization level
All GifLab commands include at least `--optimize` (equivalent to `-O`).  Experimental wrappers may choose stricter levels:

| Level | Flag | Effect |
|-------|------|--------|
| Basic | `--optimize` | Default heuristics |
| -O1   | `-O1` | Additional palette/frame tweaks |
| -O2   | `-O2` | Deeper re-encoding |
| -O3   | `-O3` | Maximum optimisation (slowest) |

### Wrapper mapping
| Wrapper class | GifLab variable | Flags emitted |
|---------------|-----------------|---------------|
| `GifsicleColorReducer` | `colors` | `--colors {N}` (+ dither control) |
| `GifsicleFrameReducer` | `ratio`  | Frame list `#idx` |
| `GifsicleLossyCompressor` | `lossy_level` | `--lossy={LEVEL}` |
| `GifsicleLossyOptim*` (extended) | `lossy_level` + fixed optimisation | `--lossy`, `-O1/-O2/-O3` |

All wrappers share `COMBINE_GROUP = "gifsicle"`; the **combiner** in `combiner_registry.py` assembles a single gifsicle command containing the union of requested flags so that palette, frame and lossy operations happen in one process.

### Environment variables
| Variable | Purpose | Default |
|----------|---------|---------|
| `GIFLAB_GIFSICLE_PATH` | Override path to the gifsicle executable | autodiscovered in `EngineConfig` |

### Exit codes & error handling
Gifsicle exits **0** on success.  Non-zero exit or a timeout raises `RuntimeError` with the captured `stderr`.

### Version detection
`get_gifsicle_version()` runs `gifsicle --version` and caches the first line, e.g. `1.94`.  Displayed via `Gifsicle*Wrapper.version()`.

---
## Quick examples
```bash
# Basic color reduction and optimization
$ gifsicle --optimize --colors 64 input.gif --output out.gif

# Lossy compression @ level 80 + frame drop (keep every 2nd frame)
$ gifsicle --optimize --lossy=80 input.gif #0#2#4#6 --output out.gif

# Maximum optimization with ordered dithering and palette shrink to 32 colors
$ gifsicle -O3 --dither=ordered --colors 32 input.gif -o out.gif
```

---
## Further reading
* Full manual – <https://www.lcdf.org/gifsicle/man.html>
* Lossy patch details – <https://github.com/kohler/gifsicle/blob/master/doc/lossy_compression.txt> 