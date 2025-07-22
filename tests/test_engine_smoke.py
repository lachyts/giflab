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
    (FFmpegColorReducer, {"colors": 32}),
]

_FRAME_WRAPPERS = [
    (GifsicleFrameReducer, {"ratio": 0.5}),
    (AnimatelyFrameReducer, {"ratio": 0.5}),
    (ImageMagickFrameReducer, {"ratio": 0.5}),
    (FFmpegFrameReducer, {"ratio": 0.5}),
]

_LOSSY_WRAPPERS = [
    (GifsicleLossyCompressor, {"lossy_level": 40}),
    (AnimatelyLossyCompressor, {"lossy_level": 40}),
    (ImageMagickLossyCompressor, {"lossy_level": 40}),
    (FFmpegLossyCompressor, {"lossy_level": 40}),
    (GifskiLossyCompressor, {"lossy_level": 40}),
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

    # Functional check for the two real engines – palette size should drop.
    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        after_colors = len(Image.open(out).getpalette()) // 3
        assert after_colors <= params["colors"], (
            f"Palette has {after_colors} colors, expected ≤ {params['colors']}"
        )


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

    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        assert Image.open(out).n_frames < Image.open(src).n_frames


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

    # Only real engines expected to compress
    if wrapper_cls.COMBINE_GROUP in {"gifsicle", "animately"}:
        assert out.stat().st_size < src.stat().st_size
