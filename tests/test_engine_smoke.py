from pathlib import Path

import pytest
from giflab.tool_wrappers import (
    AnimatelyColorReducer,
    AnimatelyFrameReducer,
    AnimatelyLossyCompressor,
    FFmpegColorReducer,
    FFmpegFrameReducer,
    FFmpegLossyCompressor,
    GifsicleColorReducer,
    GifsicleFrameReducer,
    GifsicleLossyCompressor,
    GifskiLossyCompressor,
    ImageMagickColorReducer,
    ImageMagickFrameReducer,
    ImageMagickLossyCompressor,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_gif(
    tmp: Path, frames: int = 8, size: tuple[int, int] = (48, 48)
) -> Path:
    """Generate a small synthetic GIF for smoke-testing."""
    imgs = []
    for i in range(frames):
        imgs.append(Image.new("RGB", size, color=(i * 30 % 255, 120, 180)))
    dst = tmp / "src.gif"
    imgs[0].save(dst, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    return dst


# ---------------------------------------------------------------------------
# Parameterised wrapper inventories
# ---------------------------------------------------------------------------

_COLOR_WRAPPERS = [
    (GifsicleColorReducer, {"colors": 32}),
    (AnimatelyColorReducer, {"colors": 32}),
    (ImageMagickColorReducer, {"colors": 32}),
    (FFmpegColorReducer, {"fps": 15.0}),  # FFmpeg uses fps for palette generation
]

_FRAME_WRAPPERS = [
    (GifsicleFrameReducer, {"ratio": 0.5}),
    (AnimatelyFrameReducer, {"ratio": 0.5}),
    (ImageMagickFrameReducer, {"ratio": 0.5}),
    (
        FFmpegFrameReducer,
        {"ratio": 0.5},
    ),  # FFmpeg now uses ratio like other frame reducers
]

_LOSSY_WRAPPERS = [
    (GifsicleLossyCompressor, {"lossy_level": 40}),
    (AnimatelyLossyCompressor, {"lossy_level": 40}),
    (ImageMagickLossyCompressor, {"quality": 75}),  # ImageMagick uses quality 1-100
    (FFmpegLossyCompressor, {"lossy_level": 50}),  # FFmpeg now uses lossy_level 0-100
    (GifskiLossyCompressor, {"quality": 70}),  # gifski uses quality 0-100
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("wrapper_cls,params", _COLOR_WRAPPERS)
def test_color_wrapper_smoke(wrapper_cls, params, tmp_path):
    wrapper = wrapper_cls()
    if not wrapper.available():
        pytest.skip(f"{wrapper_cls.NAME} not available on this system")

    src = _make_test_gif(tmp_path)
    out = tmp_path / f"out_{wrapper_cls.NAME}.gif"

    meta = wrapper.apply(src, out, params=params)

    assert out.exists(), "output GIF missing"
    assert "render_ms" in meta and meta["render_ms"] >= 0

    # Performance threshold check - operations should complete within 30 seconds
    assert (
        meta["render_ms"] <= 30000
    ), f"Processing took too long: {meta['render_ms']}ms (>30s limit)"

    # Functional check for real engines – validate actual changes
    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        # Existing engines: strict palette validation
        if "colors" in params:
            after_colors = len(Image.open(out).getpalette()) // 3
            assert (
                after_colors <= params["colors"]
            ), f"Palette has {after_colors} colors, expected ≤ {params['colors']}"
    elif wrapper_cls.COMBINE_GROUP == "imagemagick":
        # ImageMagick: validate output and reasonable size
        assert out.stat().st_size > 0, "ImageMagick output is empty"
        if "colors" in params:
            # ImageMagick color reduction should produce smaller or similar palette
            try:
                after_colors = len(Image.open(out).getpalette()) // 3
                assert (
                    after_colors <= params["colors"] * 2
                ), f"ImageMagick palette has {after_colors} colors, expected ≤ {params['colors'] * 2}"
            except (AttributeError, TypeError):
                # Some ImageMagick outputs may not have palettes, just validate non-empty
                pass
    elif wrapper_cls.COMBINE_GROUP == "ffmpeg":
        # FFmpeg: validate output exists and has reasonable size
        assert out.stat().st_size > 0, "FFmpeg output is empty"
        assert (
            out.stat().st_size <= src.stat().st_size * 5
        ), "FFmpeg output unreasonably large"


@pytest.mark.fast
@pytest.mark.parametrize("wrapper_cls,params", _FRAME_WRAPPERS)
def test_frame_wrapper_smoke(wrapper_cls, params, tmp_path):
    wrapper = wrapper_cls()
    if not wrapper.available():
        pytest.skip(f"{wrapper_cls.NAME} not available")

    src = _make_test_gif(tmp_path, frames=10)
    out = tmp_path / f"out_{wrapper_cls.NAME}.gif"

    meta = wrapper.apply(src, out, params=params)
    assert out.exists()
    assert "render_ms" in meta

    # Performance threshold check
    assert (
        meta["render_ms"] <= 30000
    ), f"Frame reduction took too long: {meta['render_ms']}ms (>30s limit)"

    # Functional check for frame reduction
    src_frames = Image.open(src).n_frames
    out_frames = Image.open(out).n_frames

    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        # Existing engines: strict frame reduction validation
        if "ratio" in params and params["ratio"] < 1.0:
            assert (
                out_frames <= src_frames
            ), f"Frame count should not increase: {out_frames} > {src_frames}"
            assert out_frames >= 1, f"Should have at least 1 frame, got {out_frames}"
    elif wrapper_cls.COMBINE_GROUP == "imagemagick":
        # ImageMagick: validate frame reduction with some tolerance
        if "ratio" in params and params["ratio"] < 1.0:
            assert (
                out_frames <= src_frames
            ), f"ImageMagick frame count should not increase: {out_frames} > {src_frames}"
            assert out_frames >= 1, f"Should have at least 1 frame, got {out_frames}"
        assert out.stat().st_size > 0, "ImageMagick frame reduction output is empty"
    elif wrapper_cls.COMBINE_GROUP == "ffmpeg":
        # FFmpeg: validate output and reasonable frame handling
        assert out.stat().st_size > 0, "FFmpeg frame reduction output is empty"
        assert (
            out_frames >= 1
        ), f"FFmpeg should produce at least 1 frame, got {out_frames}"
        # FFmpeg with ratio parameter should produce reasonable frame count
        if "ratio" in params and params["ratio"] < 1.0:
            assert (
                out_frames <= src_frames
            ), f"FFmpeg frame count should not increase: {out_frames} > {src_frames}"


@pytest.mark.fast
@pytest.mark.parametrize("wrapper_cls,params", _LOSSY_WRAPPERS)
def test_lossy_wrapper_smoke(wrapper_cls, params, tmp_path):
    wrapper = wrapper_cls()
    if not wrapper.available():
        pytest.skip(f"{wrapper_cls.NAME} not available")

    src = _make_test_gif(tmp_path)
    out = tmp_path / f"out_{wrapper_cls.NAME}.gif"

    meta = wrapper.apply(src, out, params=params)
    assert out.exists()
    assert "render_ms" in meta

    # Performance threshold check
    assert (
        meta["render_ms"] <= 30000
    ), f"Lossy compression took too long: {meta['render_ms']}ms (>30s limit)"

    # Functional check for lossy compression
    src_size = src.stat().st_size
    out_size = out.stat().st_size
    size_ratio = out_size / src_size

    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        # Existing engines: expect compression for typical lossy levels
        assert (
            out_size < src_size * 1.1
        ), f"Expected compression, got {size_ratio:.2f}x size ratio"
        assert (
            out_size > src_size * 0.1
        ), f"Over-compressed: {size_ratio:.2f}x size ratio seems too small"
    elif wrapper_cls.COMBINE_GROUP == "imagemagick":
        # ImageMagick: validate reasonable output size and quality parameters
        assert out_size > 0, "ImageMagick output is empty"
        assert (
            size_ratio <= 3.0
        ), f"ImageMagick output too large: {size_ratio:.2f}x original size"
        assert (
            size_ratio >= 0.1
        ), f"ImageMagick over-compressed: {size_ratio:.2f}x size ratio"
        # Validate output is still a valid GIF
        with Image.open(out) as img:
            assert img.format == "GIF", f"ImageMagick output not a GIF: {img.format}"
    elif wrapper_cls.COMBINE_GROUP == "ffmpeg":
        # FFmpeg: validate output and reasonable compression behavior
        # Note: FFmpeg may increase size due to format conversion overhead
        assert out_size > 0, "FFmpeg output is empty"
        assert (
            size_ratio <= 10.0
        ), f"FFmpeg output too large: {size_ratio:.2f}x original size"
        assert (
            size_ratio >= 0.1
        ), f"FFmpeg over-compressed: {size_ratio:.2f}x size ratio"
        # Validate output is still a valid GIF
        with Image.open(out) as img:
            assert img.format == "GIF", f"FFmpeg output not a GIF: {img.format}"
    elif wrapper_cls.COMBINE_GROUP == "gifski":
        # gifski: high-quality encoder, validate output characteristics
        assert out_size > 0, "gifski output is empty"
        assert (
            size_ratio <= 4.0
        ), f"gifski output too large: {size_ratio:.2f}x original size"
        assert (
            size_ratio >= 0.2
        ), f"gifski over-compressed: {size_ratio:.2f}x size ratio"
        # gifski should produce high-quality output
        with Image.open(out) as img:
            assert img.format == "GIF", f"gifski output not a GIF: {img.format}"
            assert img.n_frames >= 1, "gifski should preserve at least 1 frame"
