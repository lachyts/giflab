from pathlib import Path

from PIL import Image

from giflab.metrics import calculate_comprehensive_metrics


def _make_gifs(tmp_path: Path):
    original = tmp_path / "orig.gif"
    compressed = tmp_path / "comp.gif"

    img1 = Image.new("RGB", (64, 64), (100, 100, 100))
    img2 = Image.new("RGB", (64, 64), (110, 110, 110))

    # Create 3-frame GIFs with slight changes to yield temporal variance
    img1.save(original, save_all=True, append_images=[img1] * 2, duration=100, loop=0)
    img2.save(compressed, save_all=True, append_images=[img2] * 2, duration=100, loop=0)

    return original, compressed


def test_temporal_delta_keys(tmp_path):
    orig, comp = _make_gifs(tmp_path)

    metrics = calculate_comprehensive_metrics(orig, comp)

    assert "temporal_consistency_pre" in metrics
    assert "temporal_consistency_post" in metrics
    assert "temporal_consistency_delta" in metrics

    # Delta should be non-negative and <= 1
    delta = metrics["temporal_consistency_delta"]
    assert 0.0 <= delta <= 1.0


def test_temporal_delta_raw(tmp_path):
    from giflab.config import MetricsConfig

    orig, comp = _make_gifs(tmp_path)

    cfg = MetricsConfig(RAW_METRICS=True)
    metrics = calculate_comprehensive_metrics(orig, comp, cfg)

    # Raw variants should exist
    for key in [
        "temporal_consistency_pre_raw",
        "temporal_consistency_post_raw",
        "temporal_consistency_delta_raw",
    ]:
        assert key in metrics
    
    # Raw delta should equal delta
    assert metrics["temporal_consistency_delta_raw"] == metrics["temporal_consistency_delta"] 