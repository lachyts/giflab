# Multi-Engine Roll-Out Plan

_Rev 1 – generated 2025-07-19_

The goal of this work stream is to replace the **stub** wrappers for ImageMagick, FFmpeg and gifski with fully-functional implementations, align the configuration / discovery layer, and add lightweight yet meaningful test coverage similar to the existing gifsicle / animately integration tests.

---
## 1  Scope & success criteria

| Engine | Actions to support | Success criteria |
|--------|-------------------|------------------|
| ImageMagick (`convert`, `magick`) | • colour reduction  
• frame reduction  
• lossy compression (quality / sampling) | • CLI executes without error  
• Output GIF exists  
• Palette / frame-count / size reflect requested change  
• Metadata dict includes `render_ms`, `engine`, optional command string |
| FFmpeg | • colour reduction (palettegen / paletteuse)  
• frame reduction (`fps=` filter)  
• lossy compression (`qscale`, sampling) | *Same as above* |
| gifski | • lossy compression (quality slider) | *Same as above* |

*Note – ImageMagick & FFmpeg colour/frame reduction can be indirect (re-encode via PNG sequence) as long as the wrapper returns a valid GIF and meets the functional checks.*

---
## 2  Implementation tasks (mirrors TODO list)

| ID | Task | Owner | Notes |
|----|------|-------|-------|
| engines-cli-design | Define exact command lines for each action / engine |  | Put examples in `docs/cli_recipes.md` |
| helpers-impl | Implement helpers under `giflab.external_engines.*` |  | Each returns metadata dict with `render_ms`, `command`, possibly `stderr` |
| wrappers-update | Replace placeholder `apply()` in wrappers |  | Use helper funcs; ensure `COMBINE_GROUP` consistent |
| config-update | Extend `DEFAULT_ENGINE_CONFIG` + `discover_tool` |  | Auto-detect binary path; allow `$IMAGEMAGICK_PATH` override etc. |
| fixtures | Add 1-2 tiny GIFs for palette / frame checks |  | Put under `tests/fixtures/` |
| tests-integration | New module `tests/test_engine_integration_extended.py` |  | One test per (engine, action) with functional asserts |
| smoke-extend | Remove skips in `test_engine_smoke.py`; add asserts |  | After wrappers are live |
| ci-update | Update CI workflow to install the binaries |  | Use brew/apt or docker image |
| docs-update | Refresh README + tech docs |  | Document new capabilities & env vars |

---
## 3  CLI recipes (draft)

### 3.1  ImageMagick
```bash
# colour reduction to N colours
magick input.gif +dither -colors 32 output.gif

# frame reduction – keep 50 % frames (coalesce -> select -> optimize)
magick input.gif -coalesce "-set delay %%[fx:t>1?2*delay:delay]" -layers optimize output.gif
# (exact recipe TBD – frame dropping is less direct in IM)

# lossy – sample + strip metadata
magick input.gif -sampling-factor 4:2:0 -strip -quality 85 output.gif
```

### 3.2  FFmpeg
```bash
# colour reduction via palette
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