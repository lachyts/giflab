# 🎞️ GifLab — Project Scope

---

## 0 Objective
Analyse every GIF in `data/raw/` by generating a grid of compression variants and writing **one CSV row per variant** with:

* **Frame keep ratio** `frame_keep_ratio` ∈ { 1.00 · 0.90 · 0.80 · 0.70 · 0.50 }
* **Palette keep count** `color_keep_count` ∈ { 256 · 128 · 64 }
* **Lossy level** `lossy` ∈ { 0 · 40 · 120 }
* **Engine** `gifsicle`, `animately`

Requirements:

* Parallel execution, resumable after interruption.
* Corrupt/unreadable GIFs moved to `data/bad_gifs/`.
* Works on macOS and Windows/WSL.
* Keeps each GIF's original file-name **and** a content hash for deduplication.

---

## 1 Directory Layout

```
giflab/
├─ data/
│   ├─ raw/              ← originals
│   ├─ renders/          ← rendered variants
│   ├─ csv/              ← results_YYYYMMDD.csv
│   ├─ bad_gifs/         ← corrupt originals (same weird names)
│   └─ tmp/              ← temp files
├─ seed/                 ← lookup_seed_*.json
├─ logs/                 ← run logs
├─ src/giflab/
│   config.py
│   meta.py
│   frame_keep.py
│   color_keep.py
│   lossy.py
│   metrics.py
│   tagger.py            ← optional AI tags
│   io.py                ← atomic_write, CSV append, error log
│   pipeline.py          ← compression orchestrator (resume)
│   tag_pipeline.py      ← tagging pass (adds `tags` column)
│   cli.py               ← `python -m giflab …`
├─ notebooks/
│   01_explore_dataset.ipynb
│   02_build_seed_json.ipynb
├─ tests/
└─ pyproject.toml
```

---

## 2 CSV Column Schema

| column               | dtype | example                    |
|----------------------|-------|----------------------------|
| gif_sha              | str   | `6c54c899e2b0baf7…`        |
| orig_filename        | str   | `very_weird name (2).gif`  |
| engine               | str   | `gifsicle`                 |
| lossy                | int   | `40`                       |
| frame_keep_ratio     | float | `0.80`                     |
| color_keep_count     | int   | `64`                       |
| kilobytes            | float | `413.72`                   |
| ssim                 | float | `0.936`                    |
| render_ms            | int   | `217`                      |
| orig_kilobytes       | float | `2134.55`                  |
| orig_width           | int   | `480`                      |
| orig_height          | int   | `270`                      |
| orig_frames          | int   | `24`                       |
| orig_fps             | float | `24.0`                     |
| orig_n_colors        | int   | `220`                      |
| entropy (optional)   | float | `4.1`                      |
| tags (optional)      | str   | `vector-art;low-fps`       |
| timestamp            | str   | ISO-8601                   |

*Primary key = (`gif_sha`, `engine`, `lossy`, `frame_keep_ratio`, `color_keep_count`).*

---

## 3 Compression Engine Architecture

**Important**: All compression parameters (lossy, frame_keep, color_keep) must be applied **in a single pass** by the same engine. This is critical for efficiency since GIF recompilaton is inherently lossy.

### Engine Responsibilities

| Engine      | Handles                                           |
|-------------|---------------------------------------------------|
| `gifsicle`  | Lossy compression + frame reduction + color reduction |
| `animately` | Lossy compression + frame reduction + color reduction |

### Anti-Pattern (❌ Don't Do This)
```
1. Reduce frames → save temp.gif
2. Apply color reduction → save temp2.gif  
3. Apply lossy compression → save final.gif
```

### Correct Pattern (✅ Do This)
```
1. Single engine call with all parameters:
   - lossy level
   - frame_keep_ratio
   - color_keep_count
   → save final.gif
```

This approach:
- Minimizes quality loss from multiple recompressions
- Reduces I/O overhead
- Allows engines to optimize the full parameter space together

---

## 4 Variant Matrix per GIF

| frame_keep_ratio | color_keep_count | lossy values |
|------------------|------------------|--------------|
| 1.00             | 256              | 0, 40, 120   |
| 1.00             | 64               | 0, 40, 120   |
| 0.80             | 256              | 0, 40, 120   |
| 0.80             | 64               | 0, 40, 120   |

*Total*: 24 renders per GIF.
Add more ratios or palette sizes via `config.py`; the pipeline renders only missing combinations.

---

## 4 CLI

```bash
python -m giflab run    <RAW_DIR> <OUT_DIR> [options]   # compression pass
python -m giflab tag    <CSV_FILE> [options]            # tagging pass
```

Options:

```
  --workers, -j N   number of processes (default: CPU count)
  --resume          skip existing renders (default: true)
  --fail-dir PATH   folder for bad gifs   (default: data/bad_gifs)
  --csv PATH        output CSV            (default: auto-date)
  --dry-run         list work only
```

*SIGINT* finishes active tasks, flushes CSV, exits.
Re-run with `--resume` continues where it left off, using the presence of render files and SHA dedup to skip duplicates.

---

## 5 Implementation Stages (Cursor Tasks)

| Stage   | Deliverable                                                        | Status |
|---------|--------------------------------------------------------------------|--------|
| **S0**  | Repo scaffold, Poetry, black/ruff, pytest.                         | ✅     |
| **S1**  | `meta.py` — extract metadata + SHA + file-name; tests.             | ✅     |
| **S2**  | `lossy.py`.                                                        | ✅     |
| **S3**  | `frame_keep.py`.                                                   | ✅     |
| **S4**  | `color_keep.py`.                                                   | ✅     |
| **S5**  | `metrics.py` — see [QUALITY_METRICS_APPROACH.md](./QUALITY_METRICS_APPROACH.md) for technical details.                                                    | ✅     |
| **S6**  | `pipeline.py` + `io.py` (skip, resume, CSV append, move bad gifs). | ✅     |
| **S7**  | `cli.py` (`run` subcommand).                                       | ✅     |
| **S8**  | `tests/test_resume.py`.                                            | ✅     |
| **S9**  | `tagger.py` + `tag_pipeline.py` + `cli tag`.                       | ⏳     |
| **S10** | notebooks 01 & 02.                                                 | ⏳     |

**Status Legend:**
- ⏳ = Not started
- 🔄 = In progress  
- ✅ = Completed
- ❌ = Blocked/Issues

---

## 6 Cross-Platform Setup

| macOS                                      | Windows / WSL                                  |
|--------------------------------------------|------------------------------------------------|
| `brew install python@3.11 ffmpeg gifsicle` | `choco install python ffmpeg gifsicle` or WSL2 |
| Place `animately-cli` binary on PATH       | Same / WSL                                     |

**Engine Paths:**
- Animately engine: `/Users/lachlants/bin/launcher`
- Gifsicle: `gifsicle`

```bash
git clone https://…/giflab.git
cd giflab
poetry install
poetry run python -m giflab run data/raw data/
```