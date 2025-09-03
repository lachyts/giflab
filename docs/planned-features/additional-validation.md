# GIFLab — Objective Metrics + Timing Grid (Implementation Guide)

## Scope

* **Included metrics:** LPIPS, DISTS, ΔE00 patches, Flicker Excess (LPIPS-T + flat-flicker), Banding Index, Dither Index, OCR Confidence Delta, MTF50 Edge Acuity, SSIMULACRA2, (optional) VIF, Butteraugli, VMAF.
* **Also included:** **Timing-aligned evaluation** using a fixed grid so both versions compare fairly even when frames are dropped/retimed.
* **Explicitly out (for now):** Loop-seam metric.

---

## Core pipeline (high level)

1. **Decode & composite GIFs** → per-frame RGBA images + `delay_ms` + `disposal_method`.
2. **Build a timing grid** (e.g., `grid_ms=10` → 100 fps).
3. **Align both sequences to the grid** (replicate “held” frames to their covered grid slots).
4. **Run metrics** on aligned indices (same timestamps on both sides).
5. **Aggregate** (mean + p95 by default) and log.
6. **Hook into composite score** via z-scores against a frozen golden set.

---

## Timing grid (what to implement)

Goal: fair, reproducible alignment when frame delays differ or frames are dropped.

* **Inputs per original/compressed GIF:**

  * `frames[i].image` (composited RGBA)
  * `frames[i].delay_ms` (integer)
  * `frames[i].disposal_method` (respect during compositing)
* **Choose** `grid_ms` (default `10`). Smaller grid = higher fidelity, more compute.
* **Expand to grid:**

  * Create cumulative timestamps for each frame (`start_ms`, `end_ms` = `start + delay_ms`).
  * For each grid tick `t` in `[0, total_ms)` step `grid_ms`, pick the *currently displayed* frame (hold-last semantics).
  * Build `aligned_imgs = [img_at_t0, img_at_t1, ...]`.
  * Keep a **mapping** back to source frame indices for debugging:

    * `grid_to_src_frame[tick] = i`
* **Shape handling:** ensure both GIFs expand to the same number of grid ticks; if total durations differ, pad the shorter side with its last displayed frame until `min(total_ms)` (or clip both to the min duration—pick one policy and keep it consistent).
* **Optional CLI route:** you can also pre-expand with FFmpeg for both inputs:

  ```
  ffmpeg -ignore_loop 0 -i input.gif -vf "fps=100" -pix_fmt rgb24 frames/%05d.png
  ```

  (100 fps ≈ `grid_ms=10`.) The pure-Python approach keeps it self-contained.

**Add these timing fields to logs:**
`grid_ms, grid_len, total_ms_orig, total_ms_comp, duration_diff_ms`.

---

## Metrics (what they do, when to run, what to log)

### 1) LPIPS — deep perceptual distance

* **What:** Deep-feature “looks different” score (structure/texture).
* **Run:** Always. Downscale long edge to \~512 px for speed; batch if GPU.
* **Aggregate:** mean, p95.
* **Log:** `lpips_mean, lpips_p95`.

### 2) DISTS — deep texture/structure balance

* **What:** Balances structure vs texture; complements LPIPS.
* **Run:** Always (same downscale/batching).
* **Aggregate:** mean, p95.
* **Log:** `dists_mean, dists_p95`.

### 3) ΔE00 (CIEDE2000) — perceptual color difference on patches

* **What:** Perceptual color error (ΔE≈1 is just noticeable).
* **Run:** Always; **native resolution**. Sample patches (grid + any tagged regions like UI greys/brand swatches if available).
* **Aggregate:** mean, p95, and % of patches above 1/2/3/5.
* **Log:** `deltae_mean, deltae_p95, deltae_pct_gt1, deltae_pct_gt2, deltae_pct_gt3, deltae_pct_gt5`.

### 4) Flicker excess — temporal stability penalty

* **What:** Extra shimmer/pumping vs. the original.
* **Run:** Always; sample every 2–3 grid frames if needed.
* **Compute:**

  * **LPIPS-T:** LPIPS between **consecutive aligned frames** for orig and comp.
    `flicker_excess = mean(LPIPS_T_comp) − mean(LPIPS_T_orig)`; also p95.
  * **Flat-region flicker:** in low-variance patches, compare time-series luminance std (comp vs orig) → ratio.
* **Log:** `flicker_lpips_excess_mean, flicker_lpips_excess_p95, flat_flicker_std_ratio`.

### 5) Banding index — gradient contouring in flat areas

* **What:** Posterization in smooth gradients.
* **Run:** Always; operate on aligned frames but only needs the **pair at each tick** (no temporal part).
* **Compute:** On low-variance patches, use gradient-magnitude histogram features → map to a \[0–100] severity (keep mapping fixed).
* **Log:** `banding_score_mean, banding_score_p95`.

### 6) Dither index — noise vs smear in flat areas

* **What:** High-freq energy in smooth zones; flags over-/under-dither.
* **Run:** Always.
* **Compute:** Flat patches → 2D FFT power spectrum → **high-freq / mid-band** ratio.
* **Log:** `dither_energy_ratio_mean, dither_energy_ratio_p95`.

### 7) OCR confidence delta — text legibility (conditional)

* **What:** Readability impact on captions/UI text.
* **Run:** Only when “UI/text” tag or heuristic triggers (edges + small components). Use a few **key grid indices** (e.g., every Nth or max-contrast frames).
* **Compute:** OCR on orig vs comp; `ΔOCR = mean(conf_comp − conf_orig)`.
* **Log:** `ocr_conf_delta_mean, ocr_conf_delta_p95`.

### 8) MTF50 edge acuity — crispness (conditional)

* **What:** Edge sharpness (frequency at 50% contrast).
* **Run:** Same gate as OCR; sample a few long edges per chosen grid frames.
* **Compute:** Canny → Hough → slanted-edge MTF; report comp/orig ratio (lower = softer).
* **Log:** `mtf50_ratio_mean, mtf50_ratio_p10`.

### 9) SSIMULACRA2 (CLI) — fast single-number perceptual check

* **What:** Modern perceptual scalar; good human alignment at low cost.
* **Run:** Always (key frames or all).
* **Aggregate:** mean (or p95 as well if desired).
* **Log:** `ssimulacra2_mean`.

### 10) Optional extras

* **VIF (IQA lib):** general information fidelity → `vif_mean`.
* **Butteraugli (CLI, slow lane):** borderline cases → log `butteraugli_p95`.
* **VMAF (FFmpeg/libvmaf):** longer, photoreal GIFs (after grid expansion to fixed-fps clips) → `vmaf_mean`.

---

## Execution order (fast/conditional/slow)

**Fast (default every run)**
LPIPS, DISTS, ΔE00 patches, Flicker Excess, Banding Index, Dither Index, SSIMULACRA2.

**Conditional (UI/text or graphic-heavy)**
OCR Confidence Delta, MTF50 Edge Acuity.

**Slow lane (only if borderline/disagreeing)**
VIF, Butteraugli p95, VMAF mean.

---

## Aggregation & composite hook

* **Per metric:** compute over **aligned grid**; aggregate **mean + p95** (where relevant).
* **Normalization:** z-score each metric against a frozen **golden set**.
* **Composite (example weights; version them):**

  * Core perceptual: `-lpips_z (0.30)`, `-dists_z (0.20)`, `-ssimulacra2_z (0.10)`
  * Color: `-deltae_mean_z (0.10)`, `-deltae_pct_gt3_z (0.05)`
  * Temporal: `-flicker_lpips_excess_mean_z (0.10)`, `-flat_flicker_std_ratio_z (0.05)`
  * Artifacts: `-banding_score_mean_z (0.07)`, **dither window penalty** from distance to \[lo, hi] target (0.03)
  * Conditional: `ocr_conf_delta_mean_z (+)`, `mtf50_ratio_mean_z (+)` replace some weight when tag=UI/text
* Keep your legacy composite for continuity; report both.

---

## Red-flag heuristics (soft defaults)

* `deltae_pct_gt3 > 0.10` for UI/brand → warn
* `flicker_lpips_excess_mean > 0.02` → warn
* `banding_score_p95 > 60` → warn
* `dither_energy_ratio_mean` outside `[0.8, 1.3]` → warn (content-dependent)
* `ocr_conf_delta_mean < -0.05` (when run) → warn
* `mtf50_ratio_p10 < 0.75` (when run) → warn
* `ssimulacra2_mean` worse than golden-set p75 → warn

---

## Minimal function skeletons (for Claude Code)

```python
# 1) Expand to timing grid
def expand_to_grid(frames, grid_ms=10):
    """
    frames: list of {image: np.ndarray|PIL, delay_ms: int}
    returns: imgs_aligned [T], grid_to_src_idx [T], total_ms
    """
    # compose frames using disposal rules before calling this
    # build cumulative timeline and replicate held frames per grid tick
    ...

# 2) Core metrics (per-tick)
def lpips_pair_batch(imgs_a, imgs_b) -> (mean, p95): ...
def dists_pair_batch(imgs_a, imgs_b) -> (mean, p95): ...
def deltae_patches(orig_imgs, comp_imgs) -> dict: ...
def banding_index(orig_imgs, comp_imgs) -> (mean, p95): ...
def dither_index(orig_imgs, comp_imgs) -> (mean, p95): ...

# 3) Temporal metrics (using aligned imgs)
def flicker_excess(aligned_a, aligned_b) -> dict:
    # LPIPS between consecutive frames (orig vs comp), plus flat-region luminance std ratio
    ...

# 4) Conditional
def ocr_conf_delta(keyframes_a, keyframes_b) -> (mean, p95): ...
def mtf50_ratios(keyframes_a, keyframes_b) -> (mean, p10): ...

# 5) Optional CLIs
def ssimulacra2_mean(keyframes_a, keyframes_b) -> float: ...
def butteraugli_p95(keyframes_a, keyframes_b) -> float: ...
def vmaf_mean(video_a_path, video_b_path) -> float: ...
```

---

## CSV fields to append

`grid_ms, grid_len, total_ms_orig, total_ms_comp, duration_diff_ms,  
lpips_mean, lpips_p95, dists_mean, dists_p95,  
deltae_mean, deltae_p95, deltae_pct_gt1, deltae_pct_gt2, deltae_pct_gt3, deltae_pct_gt5,  
flicker_lpips_excess_mean, flicker_lpips_excess_p95, flat_flicker_std_ratio,  
banding_score_mean, banding_score_p95, dither_energy_ratio_mean, dither_energy_ratio_p95,  
ocr_conf_delta_mean, ocr_conf_delta_p95, mtf50_ratio_mean, mtf50_ratio_p10,  
ssimulacra2_mean, vif_mean, butteraugli_p95, vmaf_mean`

(Include optional fields only if those paths are enabled.)

---

### Implementation notes

* Keep **Lab conversion** consistent (sRGB → Lab with fixed gamma).
* For LPIPS/DISTS, **batch** tensors and use GPU if present; otherwise downscale to keep CPU reasonable.
* For banding/dither, expose thresholds and patch counts via config; return both raw features and your mapped severity.
* Gate OCR/MTF50 with a simple heuristic or tag to control cost.
* On errors, raise with `metric_name`, `tick_or_frame_idx`, and a short message.

That’s everything needed to implement the timing grid + metrics cleanly without over-constraining how it fits your repo.
