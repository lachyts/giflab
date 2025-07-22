from pathlib import Path

import pytest
from PIL import Image

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_gif(tmp: Path, frames: int = 8, size: tuple[int, int] = (48, 48)) -> Path:
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
    (FFmpegFrameReducer, {"fps": 5.0}),  # FFmpeg uses target fps
]

_LOSSY_WRAPPERS = [
    (GifsicleLossyCompressor, {"lossy_level": 40}),
    (AnimatelyLossyCompressor, {"lossy_level": 40}),
    (ImageMagickLossyCompressor, {"quality": 75}),  # ImageMagick uses quality 1-100
    (FFmpegLossyCompressor, {"qv": 25, "fps": 15.0}),  # FFmpeg uses qv + fps
    (GifskiLossyCompressor, {"quality": 70}),  # gifski uses quality 0-100
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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

    # Functional check for real engines – validate actual changes
    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately", "imagemagick"}:
        # For gifsicle/animately/imagemagick: check palette size reduction
        if "colors" in params:
            after_colors = len(Image.open(out).getpalette()) // 3
            assert after_colors <= params["colors"], (
                f"Palette has {after_colors} colors, expected ≤ {params['colors']}"
            )
    elif wrapper_cls.COMBINE_GROUP == "ffmpeg":
        # For FFmpeg: just validate output exists and has reasonable size
        assert out.stat().st_size > 0, "FFmpeg output is empty"


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

    # Functional check for frame reduction
    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately", "imagemagick"}:
        # These engines should actually reduce frame count
        if "ratio" in params and params["ratio"] < 1.0:
            assert Image.open(out).n_frames < Image.open(src).n_frames
    elif wrapper_cls.COMBINE_GROUP == "ffmpeg":
        # FFmpeg should produce valid output
        assert out.stat().st_size > 0, "FFmpeg frame reduction output is empty"


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

    # Functional check for lossy compression
    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        # These should definitely compress
        assert out.stat().st_size < src.stat().st_size
    elif wrapper_cls.COMBINE_GROUP in {"imagemagick", "ffmpeg", "gifski"}:
        # New engines should produce valid output, compression depends on settings
        assert out.stat().st_size > 0, f"{wrapper_cls.NAME} output is empty"
        # Some engines may increase file size depending on parameters, just validate it's reasonable
        size_ratio = out.stat().st_size / src.stat().st_size
        assert size_ratio <= 10.0, f"{wrapper_cls.NAME} produced unreasonably large output (ratio: {size_ratio:.1f})"
