"""Quick unit-test for the new advanced lossy pipeline.

Runs under the fast test-suite: it patches out the real *animately* binary so
no external dependency is required.  It checks that the helper returns sane
metadata and writes the expected output file.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

import giflab.lossy as lossy


@pytest.fixture()
def tiny_gif(tmp_path: Path) -> Path:  # noqa: D401 – simple factory
    """Create a 2-frame 10×10 GIF for quick tests."""
    frames = [
        Image.new("RGB", (10, 10), (255, 0, 0)),
        Image.new("RGB", (10, 10), (0, 0, 255)),
    ]
    gif_path = tmp_path / "tiny.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return gif_path


def _fake_animately(*args, **kwargs):  # noqa: D401 – signature irrelevant
    """Stub that mimics the Animately binary CLI.

    Simply copies the PNG sequence referenced in the JSON config into the GIF
    *output* path so the caller sees a file.  No real compression done.
    """
    import json
    import shutil

    cmd = list(args[0])  # first positional arg is command list
    json_idx = cmd.index("--advanced-lossy") + 1
    json_path = Path(cmd[json_idx])
    out_idx = cmd.index("--output") + 1
    out_path = Path(cmd[out_idx])

    with open(json_path) as fh:
        cfg = json.load(fh)
    # Pick first PNG as dummy; in the test we don't care about contents
    first_png = Path(cfg["frames"][0]["png"])
    shutil.copy(first_png, out_path)
    return lossy.subprocess.CompletedProcess(cmd, 0, "", "")


@patch("giflab.lossy._execute_animately_advanced")
def test_advanced_lossy_path(fake_exec, tiny_gif, tmp_path):  # noqa: D401
    """Ensure *compress_with_animately_advanced_lossy* returns expected keys."""
    output = tmp_path / "out.gif"

    # Fake out the low-level executor so we don't need the real binary
    def _stub_exec(animately_path, json_config_path, out_path):
        out_path.touch()
        return (123, None)

    fake_exec.side_effect = _stub_exec

    result = lossy.compress_with_animately_advanced_lossy(
        tiny_gif, output, lossy_level=40, color_keep_count=16
    )

    assert result["engine"] == "animately-advanced"
    assert result["lossy_level"] == 40
    assert result["color_keep_count"] == 16
    assert result["render_ms"] == 123
    assert output.exists()
