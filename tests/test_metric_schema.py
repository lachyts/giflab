from pathlib import Path

import pytest
from giflab.metrics import calculate_comprehensive_metrics
from giflab.schema import MetricRecordV1, validate_metric_record, is_valid_record
from PIL import Image


class TestMetricRecordSchema:
    """Validate MetricRecordV1 against real metric outputs."""

    @staticmethod
    def _create_test_gifs(tmp_path: Path) -> tuple[Path, Path]:
        original_path = tmp_path / "original.gif"
        compressed_path = tmp_path / "compressed.gif"

        img1 = Image.new("RGB", (64, 64), (120, 120, 120))
        img2 = Image.new("RGB", (64, 64), (130, 130, 130))

        img1.save(original_path)
        img2.save(compressed_path)

        return original_path, compressed_path

    def test_validation_passes_on_real_record(self, tmp_path):
        original, compressed = self._create_test_gifs(tmp_path)

        metrics = calculate_comprehensive_metrics(original, compressed)

        # Should not raise
        model = validate_metric_record(metrics)
        assert isinstance(model, MetricRecordV1)
        assert 0.0 <= model.composite_quality <= 1.0

    def test_validation_fails_on_negative_values(self):
        bad_record = {
            "composite_quality": 0.5,
            "kilobytes": -10,  # invalid
            "render_ms": 100,
            "ssim": 0.8,
            "ssim_std": 0.0,
            "ssim_min": 0.8,
            "ssim_max": 0.8,
        }

        assert is_valid_record(bad_record) is False

        with pytest.raises(Exception):
            validate_metric_record(bad_record) 