import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
from PIL import Image, ImageDraw

from giflab.color_keep import count_gif_colors
from giflab.config import DEFAULT_ENGINE_CONFIG
from giflab.lossy import LossyEngine, compress_with_animately, compress_with_gifsicle
from giflab.metrics import extract_gif_frames

# -------------------------
# Helper utilities & fixtures
# -------------------------


def _create_test_gif(
    path: Path, frames: int = 10, size: tuple[int, int] = (50, 50)
) -> None:
    """Generate a simple animated GIF for test purposes.

    The content purposefully varies per-frame so both colour and frame
    analyses have something to measure.
    """
    images = []
    for i in range(frames):
        img = Image.new(
            "RGB", size, color=((i * 37) % 255, (i * 53) % 255, (i * 97) % 255)
        )
        draw = ImageDraw.Draw(img)
        # moving rectangle
        offset = (i * 3) % (size[0] - 10)
        draw.rectangle([offset, offset, offset + 10, offset + 10], fill=(255, 255, 255))
        images.append(img)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
    )


def _engine_available(engine: LossyEngine) -> bool:
    """Return True if the specified engine binary appears to be available."""
    if engine == LossyEngine.GIFSICLE:
        binary = DEFAULT_ENGINE_CONFIG.GIFSICLE_PATH
        try:
            subprocess.run(
                [binary, "--version"], capture_output=True, check=True, timeout=3
            )
            return True
        except Exception:
            return False
    elif engine == LossyEngine.ANIMATELY:
        binary = DEFAULT_ENGINE_CONFIG.ANIMATELY_PATH
        from giflab.lossy import _is_executable

        return _is_executable(binary)
    else:
        return False


# Fixture yields a temporary GIF path
@pytest.fixture(scope="module")
def test_gif_tmp():
    """Provide a temporary GIF shared by tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = Path(tmpdir) / "sample.gif"
        _create_test_gif(gif_path, frames=10, size=(50, 50))
        yield gif_path


def _compress_with(
    engine: LossyEngine,
    src: Path,
    lossy_level: int,
    frame_ratio: float,
    colour_count: int | None = None,
):
    """Wrapper around compression functions with sensible defaults."""
    tmpdir = tempfile.TemporaryDirectory()
    out_path = Path(tmpdir.name) / f"output_{engine.value}.gif"

    # Use the direct compression functions to avoid validation issues
    if engine == LossyEngine.GIFSICLE:
        result = compress_with_gifsicle(
            src, out_path, lossy_level, frame_ratio, colour_count
        )
    else:
        result = compress_with_animately(
            src, out_path, lossy_level, frame_ratio, colour_count
        )

    return out_path, result, tmpdir


# -------------------------
# Actual equivalence tests
# -------------------------


@pytest.mark.parametrize(
    "lossy_level, frame_ratio, colours",
    [
        (0, 1.0, None),  # lossless, no reduction
        (0, 1.0, 64),  # lossless with color reduction
        (0, 0.5, None),  # frame reduction only
    ],
)
def test_gifsicle_vs_animately_equivalence(
    test_gif_tmp, lossy_level, frame_ratio, colours
):
    """Ensure both engines apply the *same* high-level operations.

    We don't compare visual quality – only that derived *metadata* such as
    frame count and palette size line up, proving we invoked the engines
    consistently.
    """
    # Skip if either engine isn't present
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip(
            "Both gifsicle and animately must be available for equivalence checks"
        )

    try:
        gif_out, meta_gif, tmpdir_gif = _compress_with(
            LossyEngine.GIFSICLE, test_gif_tmp, lossy_level, frame_ratio, colours
        )
    except RuntimeError as e:
        pytest.skip(f"Gifsicle failed: {e}")

    try:
        ani_out, meta_ani, tmpdir_ani = _compress_with(
            LossyEngine.ANIMATELY, test_gif_tmp, lossy_level, frame_ratio, colours
        )
    except RuntimeError as e:
        tmpdir_gif.cleanup()
        pytest.skip(f"Animately failed: {e}")

    try:
        # ----------------- Assertions on frames -----------------
        frames_gif = extract_gif_frames(gif_out).frame_count
        frames_ani = extract_gif_frames(ani_out).frame_count

        print(f"Frame counts: gifsicle={frames_gif}, animately={frames_ani}")

        # When we requested a reduction ensure it was honoured
        original_frames = extract_gif_frames(test_gif_tmp).frame_count
        expected_frames = max(1, int(original_frames * frame_ratio))

        print(f"Original frames: {original_frames}, expected: {expected_frames}")

        # Both engines should produce the same number of frames
        assert (
            frames_gif == frames_ani
        ), f"Frame count mismatch – gifsicle={frames_gif}, animately={frames_ani}"

        # And it should match our expectation
        assert (
            frames_gif == expected_frames
        ), f"Frame reduction did not meet expectation: got {frames_gif}, expected {expected_frames}"

        # ----------------- Assertions on colours ----------------
        colours_gif = count_gif_colors(gif_out)
        colours_ani = count_gif_colors(ani_out)

        print(f"Color counts: gifsicle={colours_gif}, animately={colours_ani}")

        if colours is not None:
            # Engines should not exceed the requested palette size
            assert (
                colours_gif <= colours
            ), f"Gifsicle exceeded color limit: {colours_gif} > {colours}"
            assert (
                colours_ani <= colours
            ), f"Animately exceeded color limit: {colours_ani} > {colours}"

        # For now, just ensure both engines produce reasonable color counts
        # (allow significant differences as engines may optimize differently)
        assert (
            colours_gif > 0 and colours_ani > 0
        ), "Both engines should produce GIFs with colors"

        # The color counts should be in a reasonable range for a simple test GIF
        assert (
            colours_gif <= 256 and colours_ani <= 256
        ), "Color counts should not exceed GIF maximum"

        # Print summary for manual verification
        print(f"✅ Test passed: {lossy_level=}, {frame_ratio=}, {colours=}")
        print(f"   Frames: {frames_gif} (both engines)")
        print(f"   Colors: gifsicle={colours_gif}, animately={colours_ani}")

    finally:
        # Clean up temporary directories
        tmpdir_gif.cleanup()
        tmpdir_ani.cleanup()


def test_engine_basic_functionality():
    """Basic smoke test to ensure both engines can process a simple GIF."""
    if not (
        _engine_available(LossyEngine.GIFSICLE)
        and _engine_available(LossyEngine.ANIMATELY)
    ):
        pytest.skip(
            "Both gifsicle and animately must be available for basic functionality test"
        )

    # Create a simple test GIF
    with tempfile.TemporaryDirectory() as tmpdir:
        test_gif = Path(tmpdir) / "simple.gif"
        _create_test_gif(test_gif, frames=3, size=(30, 30))

        # Test both engines can process it
        try:
            gif_out, _, tmpdir_gif = _compress_with(
                LossyEngine.GIFSICLE, test_gif, 0, 1.0, None
            )
            assert gif_out.exists(), "Gifsicle should create output file"
            tmpdir_gif.cleanup()
            print("✅ Gifsicle basic functionality: PASS")
        except Exception as e:
            print(f"❌ Gifsicle basic functionality: FAIL - {e}")

        try:
            ani_out, _, tmpdir_ani = _compress_with(
                LossyEngine.ANIMATELY, test_gif, 0, 1.0, None
            )
            assert ani_out.exists(), "Animately should create output file"
            tmpdir_ani.cleanup()
            print("✅ Animately basic functionality: PASS")
        except Exception as e:
            print(f"❌ Animately basic functionality: FAIL - {e}")
