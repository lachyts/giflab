from pathlib import Path
from unittest.mock import patch

import pytest

from giflab.dynamic_pipeline import Pipeline, PipelineStep
from giflab.core.runner import (
    ExperimentalConfig,
    ExperimentalPipeline,
    ExperimentJob,
)
from giflab.meta import GifMetadata
from giflab.tool_wrappers import (
    AnimatelyColorReducer,
    AnimatelyFrameReducer,
    GifsicleLossyBasic,
)


@pytest.fixture
def dummy_gif(tmp_path: Path) -> Path:
    p = tmp_path / "dummy.gif"
    p.write_bytes(
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04"
        b"\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00;"
    )
    return p


def _metadata_for(path: Path) -> GifMetadata:  # minimal helper
    return GifMetadata(
        gif_sha="sha",
        orig_filename=path.name,
        orig_kilobytes=1.0,
        orig_width=1,
        orig_height=1,
        orig_frames=1,
        orig_fps=1.0,
        orig_n_colors=2,
    )


def test_mixed_tool_collapsing(tmp_path: Path, dummy_gif: Path):
    """Animately+Animately+Gifsicle should collapse into *two* CLI calls."""
    # Build pipeline: Animately frame → AnimatelyColorReducer → Gifsicle lossy
    pipeline = Pipeline(
        [
            PipelineStep("frame_reduction", AnimatelyFrameReducer),
            PipelineStep("color_reduction", AnimatelyColorReducer),
            PipelineStep("lossy_compression", GifsicleLossyBasic),
        ]
    )

    # Minimal ExperimentalPipeline
    cfg = ExperimentalConfig(ENABLE_DETAILED_ANALYSIS=False)
    runner = ExperimentalPipeline(cfg, workers=1)

    job = ExperimentJob(
        gif_path=dummy_gif,
        metadata=_metadata_for(dummy_gif),
        strategy=pipeline.identifier(),
        engine="dynamic",
        optimization_level="n/a",
        dithering_option="n/a",
        lossy=40,
        frame_keep_ratio=0.8,
        color_keep_count=64,
        output_path=tmp_path / "out.gif",
        pipeline=pipeline,
    )

    # Patch combiners dict so collapsing uses mocks
    from giflab import combiner_registry as _cr

    with patch.dict(
        _cr._COMBINERS,
        {
            "animately": lambda *a, **k: {"render_ms": 1},
            "gifsicle": lambda *a, **k: {"render_ms": 1},
        },
    ):
        res = runner._execute_dynamic_pipeline(job)

    # result should contain two steps (animately group + gifsicle group)
    assert len(res["steps"]) == 2


@pytest.mark.parametrize(
    "group,color_cls,lossy_cls",
    [
        ("imagemagick", "ImageMagickColorReducer", "ImageMagickLossyCompressor"),
        ("ffmpeg", "FFmpegColorReducer", "FFmpegLossyCompressor"),
    ],
)
def test_placeholder_combiner_groups(
    tmp_path: Path, dummy_gif: Path, group, color_cls, lossy_cls
):
    import importlib

    tw_mod = importlib.import_module("giflab.tool_wrappers")
    color_cls_obj = getattr(tw_mod, color_cls)
    lossy_cls_obj = getattr(tw_mod, lossy_cls)

    pipeline = Pipeline(
        [
            PipelineStep("color_reduction", color_cls_obj),
            PipelineStep("lossy_compression", lossy_cls_obj),
        ]
    )

    cfg = ExperimentalConfig(ENABLE_DETAILED_ANALYSIS=False)
    runner = ExperimentalPipeline(cfg, workers=1)

    job = ExperimentJob(
        gif_path=dummy_gif,
        metadata=_metadata_for(dummy_gif),
        strategy=pipeline.identifier(),
        engine="dynamic",
        optimization_level="n/a",
        dithering_option="n/a",
        lossy=10,
        frame_keep_ratio=1.0,
        color_keep_count=128,
        output_path=tmp_path / "out.gif",
        pipeline=pipeline,
    )

    from giflab import combiner_registry as _cr

    with patch.dict(_cr._COMBINERS, {group: lambda *a, **k: {"render_ms": 1}}):
        res = runner._execute_dynamic_pipeline(job)

    assert len(res["steps"]) == 1  # collapsed into single call
