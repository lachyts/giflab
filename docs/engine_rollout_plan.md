# Multi-Engine Roll-Out Plan

_Rev 1 – generated 2025-07-19_

The goal of this work stream is to replace the **stub** wrappers for ImageMagick, FFmpeg and gifski with fully-functional implementations, align the configuration / discovery layer, and add lightweight yet meaningful test coverage similar to the existing gifsicle / animately integration tests.

The project standardizes on American English spelling (e.g., "color") across all documentation.

---
## 1  Scope & success criteria

| Engine | Actions to support | Success criteria |
|--------|-------------------|------------------|
| ImageMagick (`convert`, `magick`) | • color reduction  
• frame reduction  
• lossy compression (quality / sampling) | • CLI executes without error  
• Output GIF exists  
• Palette / frame-count / size reflect requested change  
• Metadata dict includes `render_ms`, `engine`, `command`, `kilobytes` |
| FFmpeg | • color reduction (palettegen / paletteuse)  
• frame reduction (`fps=` filter)  
• lossy compression (`qscale`, sampling) | *Same as above* |
| gifski | • lossy compression (quality slider) | *Same as above* |

*Note – ImageMagick & FFmpeg color/frame reduction can be indirect (re-encode via PNG sequence) as long as the wrapper returns a valid GIF and meets the functional checks.*

---
## 2  Implementation tasks

Below is the ordered work-breakdown we will track in the project TODO list.

| # | Task ID | Description | Notes |
|---|---------|-------------|-------|
| 1 | **engines-cli-design** | Document exact CLI invocations for each engine / action. | Examples saved to `docs/cli_recipes.md`. |
| 2 | **helpers-impl** | Add helper functions under `giflab.external_engines.*` that wrap each CLI and return the standard metadata dict. | Must populate `render_ms`, `engine`, `command`, `kilobytes`. |
| 3 | **wrappers-update** | Replace placeholder `apply()` methods in ImageMagick/FFmpeg/gifski wrappers with real calls to the helpers. | Ensure `COMBINE_GROUP` values stay consistent. |
| 4 | **config-update** | Extend `DEFAULT_ENGINE_CONFIG` and `system_tools.discover_tool` to auto-detect executables; allow `$GIFLAB_IMAGEMAGICK_PATH`, `$GIFLAB_FFMPEG_PATH`, `$GIFLAB_GIFSKI_PATH`, `$GIFLAB_GIFSICLE_PATH`, `$GIFLAB_FFPROBE_PATH` overrides. | |
| 5 | **fixtures** | Add 1–2 tiny GIF fixtures for palette, frame-count and size assertions. | Store under `tests/fixtures/`. |
| 6 | **tests-integration** | Create `tests/test_engine_integration_extended.py` with one functional test per *(engine × action)*. | Validate functional change + metadata. |
| 7 | **smoke-extend** | Remove skips in `test_engine_smoke.py` and add functional asserts for the new engines. | Ensure existing gifsicle / Animately smoke tests remain green and runs in fast suite. |
| 8 | **ci-update** | Update CI workflow / Docker image to include ImageMagick, FFmpeg, gifski so the tests pass in CI. | |
| 9 | **docs-update** | Refresh README and technical docs to list the new engines, environment variables and usage examples. | |
| 10 | **cleanup-stubs** | Remove obsolete stub wrappers once real implementations are merged. | |

---
## 3  CLI recipes (draft)

### 3.1  ImageMagick
```bash
# color reduction to N colors
magick input.gif +dither -colors 32 output.gif

# frame reduction – keep 50 % frames (delete every 2nd frame)
magick input.gif -coalesce -delete '1--2' -layers optimize output.gif
# (wrapper deletes every second frame; adjust ratio dynamically)

# lossy – sample + strip metadata
magick input.gif -sampling-factor 4:2:0 -strip -quality 85 output.gif
```

### 3.2  FFmpeg
```bash
# color reduction via palette
ffmpeg -i input.gif -filter_complex "fps=15,palettegen" palette.png
ffmpeg -i input.gif -i palette.png -filter_complex "fps=15,paletteuse" output.gif
# (wrapper can run two-pass transparently)

# frame reduction only
ffmpeg -i input.gif -filter_complex "fps=7.5" output.gif

# lossy (quantiser 30)
ffmpeg -i input.gif -lavfi "fps=15" -q:v 30 output.gif
```

### 3.3  gifski
```bash
# lossy quality 60 %
gifski --quality 60 -o output.gif input_frames/*.png
# (wrapper will split GIF to PNGs first; OK for small smoke tests)
```

---
## 4  Testing strategy

1. **Smoke tests (`tests/test_engine_smoke.py`)**  
   – Already in place, will gain functional assertions for the new engines once wrappers are real.

2. **Per-engine integration tests**  
   – For each engine/action: generate toy GIF ➞ run wrapper ➞ assert functional change + metadata.

3. **Fail-fast on CI**  
   – Mark integration tests `@pytest.mark.external_tools` and run them in a dedicated workflow job where the binaries are pre-installed.

---
## 5  Milestones & sequencing

1. ImageMagick helpers & wrapper swap-in → merge.  
2. FFmpeg helpers & wrappers → merge.  
3. gifski lossy helper & wrapper → merge.  
4. Extended smoke + integration tests.  
5. CI and docs update.

Each step should keep the existing gifsicle / animately tests green.

---
## 6  Open questions

* Reliable frame-drop recipe for ImageMagick without giant intermediate files?  
* Acceptable quality / performance trade-off for gifski PNG split in integration tests.

---
**End of document** 