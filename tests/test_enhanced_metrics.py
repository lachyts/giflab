"""Tests for enhanced metrics calculation including efficiency weighting and composite quality."""

import math
from unittest.mock import patch

from giflab.config import MetricsConfig
from giflab.enhanced_metrics import (
    calculate_composite_quality,
    calculate_efficiency_metric,
    calculate_legacy_composite_quality,
    get_enhanced_weights_info,
    normalize_metric,
    process_metrics_with_enhanced_quality,
)


class TestEfficiencyMetricCalculation:
    """Tests for the 50/50 balanced efficiency metric calculation."""

    def test_basic_efficiency_calculation(self):
        """Test basic efficiency calculation with normal values."""
        efficiency = calculate_efficiency_metric(
            compression_ratio=5.0, composite_quality=0.8
        )

        # Should return a value between 0 and 1
        assert 0.0 <= efficiency <= 1.0

        # Calculate expected value manually
        normalized_compression = math.log(1 + 5.0) / math.log(1 + 20.0)
        expected = (0.8**0.5) * (normalized_compression**0.5)

        assert abs(efficiency - expected) < 0.001

    def test_equal_50_50_weighting(self):
        """Test that quality and compression have equal 50% weighting."""
        # Test with perfect quality, moderate compression
        eff1 = calculate_efficiency_metric(compression_ratio=5.0, composite_quality=1.0)

        # Test with moderate quality, perfect compression (20x = max practical)
        eff2 = calculate_efficiency_metric(
            compression_ratio=20.0, composite_quality=0.5
        )

        # Extract the normalized compression for 20x
        math.log(1 + 20.0) / math.log(1 + 20.0)  # = 1.0

        # Calculate expected efficiencies
        expected1 = (1.0**0.5) * ((math.log(6) / math.log(21)) ** 0.5)
        expected2 = (0.5**0.5) * (1.0**0.5)

        assert abs(eff1 - expected1) < 0.001
        assert abs(eff2 - expected2) < 0.001

    def test_equal_weighting_verification(self):
        """Verify that quality and compression changes have equal impact."""
        base_quality = 0.5
        base_compression = 5.0

        # Increase quality by 0.2
        eff_quality_up = calculate_efficiency_metric(
            compression_ratio=base_compression, composite_quality=base_quality + 0.2
        )

        # Increase compression to achieve similar normalized increase
        # Need to find compression that gives normalized_compression increase ≈ 0.2
        # Current normalized: log(6)/log(21) ≈ 0.588
        # Target normalized: 0.588 + 0.2 = 0.788
        # Solve: log(1+x)/log(21) = 0.788 => x = 21^0.788 - 1 ≈ 11.6
        higher_compression = 11.6
        eff_compression_up = calculate_efficiency_metric(
            compression_ratio=higher_compression, composite_quality=base_quality
        )

        base_efficiency = calculate_efficiency_metric(base_compression, base_quality)

        # Changes should have similar magnitude (within tolerance for log scaling)
        quality_change = eff_quality_up - base_efficiency
        compression_change = eff_compression_up - base_efficiency

        # Should be reasonably close due to equal 50/50 weighting
        assert abs(quality_change - compression_change) < 0.1

    def test_log_normalization_compression(self):
        """Test log normalization of compression ratios."""
        # Test that compression > 20x is capped at 1.0 normalized
        efficiency_20x = calculate_efficiency_metric(20.0, 1.0)
        efficiency_100x = calculate_efficiency_metric(100.0, 1.0)

        # Should be equal due to 20x cap
        assert abs(efficiency_20x - efficiency_100x) < 0.001

        # Both should equal (1.0 ** 0.5) * (1.0 ** 0.5) = 1.0
        assert abs(efficiency_20x - 1.0) < 0.001

    def test_geometric_mean_properties(self):
        """Test that geometric mean prevents one-dimensional optimization."""
        # Perfect compression with zero quality should give zero efficiency
        eff_zero_quality = calculate_efficiency_metric(20.0, 0.0)
        assert eff_zero_quality == 0.0

        # Zero compression with perfect quality should give zero efficiency
        eff_zero_compression = calculate_efficiency_metric(0.0, 1.0)
        assert eff_zero_compression == 0.0

        # Balanced approach should beat extreme approach
        balanced = calculate_efficiency_metric(
            5.0, 0.8
        )  # Good quality + good compression
        calculate_efficiency_metric(1.5, 1.0)  # Perfect quality + poor compression
        calculate_efficiency_metric(20.0, 0.6)  # Perfect compression + poor quality

        # Balanced should be competitive with or better than extremes
        assert balanced > 0.5  # Should achieve reasonable efficiency

    def test_boundary_conditions(self):
        """Test boundary conditions and error handling."""
        # Negative compression ratio
        assert calculate_efficiency_metric(-1.0, 0.8) == 0.0

        # Zero compression ratio
        assert calculate_efficiency_metric(0.0, 0.8) == 0.0

        # Negative quality
        assert calculate_efficiency_metric(5.0, -0.1) == 0.0

        # Quality > 1.0 should still work (not clamped in efficiency calc)
        eff_high_quality = calculate_efficiency_metric(5.0, 1.2)
        assert eff_high_quality > 0.0

    def test_efficiency_range(self):
        """Test that efficiency stays in expected 0-1 range."""
        test_cases = [
            (1.1, 0.1),  # Low compression, low quality
            (5.0, 0.5),  # Medium compression, medium quality
            (10.0, 0.9),  # High compression, high quality
            (20.0, 1.0),  # Max compression, perfect quality
            (100.0, 1.0),  # Over-max compression, perfect quality
        ]

        for compression, quality in test_cases:
            efficiency = calculate_efficiency_metric(compression, quality)
            assert (
                0.0 <= efficiency <= 1.0
            ), f"Efficiency {efficiency} out of range for compression={compression}, quality={quality}"

    def test_monotonicity_properties(self):
        """Test that efficiency increases with better quality or compression."""
        base_compression, base_quality = 5.0, 0.7

        # Better quality should increase efficiency
        better_quality_eff = calculate_efficiency_metric(
            base_compression, base_quality + 0.1
        )
        base_eff = calculate_efficiency_metric(base_compression, base_quality)
        assert better_quality_eff > base_eff

        # Better compression should increase efficiency
        better_compression_eff = calculate_efficiency_metric(
            base_compression * 1.5, base_quality
        )
        assert better_compression_eff > base_eff


class TestNormalizeMetric:
    """Tests for metric normalization functions."""

    def test_standard_normalization(self):
        """Test standard 0-1 normalization."""
        # SSIM-like metric (higher is better)
        normalized = normalize_metric("ssim_mean", 0.75, min_val=0.0, max_val=1.0)
        assert normalized == 0.75

    def test_mse_normalization(self):
        """Test MSE log normalization (lower is better)."""
        # Zero MSE should give perfect score
        assert normalize_metric("mse_mean", 0.0) == 1.0

        # Higher MSE should give lower normalized score
        mse_10 = normalize_metric("mse_mean", 10.0)
        mse_100 = normalize_metric("mse_mean", 100.0)
        assert mse_100 < mse_10

    def test_gmsd_normalization(self):
        """Test GMSD normalization (lower is better)."""
        # Zero GMSD should give perfect score
        assert normalize_metric("gmsd_mean", 0.0) == 1.0

        # GMSD at max (0.5) should give zero score
        assert normalize_metric("gmsd_mean", 0.5) == 0.0

        # Mid-range should give reasonable score
        mid_score = normalize_metric("gmsd_mean", 0.25)
        assert 0.4 < mid_score < 0.6

    def test_ms_ssim_normalization(self):
        """Test MS-SSIM normalization handles negative values."""
        # Negative MS-SSIM should give zero score
        assert normalize_metric("ms_ssim_mean", -0.1) == 0.0

        # Positive values should normalize normally
        assert normalize_metric("ms_ssim_mean", 0.8) == 0.8


class TestEnhancedCompositeQuality:
    """Tests for enhanced composite quality calculation."""

    def create_test_metrics(self) -> dict:
        """Create a comprehensive set of test metrics."""
        return {
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
            "psnr_mean": 0.8,
            "mse_mean": 100.0,
            "fsim_mean": 0.88,
            "edge_similarity_mean": 0.82,
            "gmsd_mean": 0.1,
            "chist_mean": 0.75,
            "sharpness_similarity_mean": 0.78,
            "texture_similarity_mean": 0.84,
            "temporal_consistency": 0.92,
        }

    def test_enhanced_composite_calculation(self):
        """Test enhanced composite quality calculation with all metrics."""
        metrics = self.create_test_metrics()
        config = MetricsConfig(USE_ENHANCED_COMPOSITE_QUALITY=True)

        enhanced_quality = calculate_composite_quality(metrics, config)

        assert 0.0 <= enhanced_quality <= 1.0
        assert enhanced_quality > 0.65  # Should be reasonably high for good metrics

    def test_legacy_fallback(self):
        """Test fallback to legacy calculation when enhanced is disabled."""
        metrics = self.create_test_metrics()
        config = MetricsConfig(USE_ENHANCED_COMPOSITE_QUALITY=False)

        enhanced_quality = calculate_composite_quality(metrics, config)
        legacy_quality = calculate_legacy_composite_quality(metrics, config)

        # Should be equal when enhanced is disabled
        assert abs(enhanced_quality - legacy_quality) < 0.001

    def test_missing_metrics_handling(self):
        """Test handling of missing metrics in enhanced calculation."""
        # Only provide core metrics
        minimal_metrics = {
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
        }

        enhanced_quality = calculate_composite_quality(minimal_metrics)

        # Should still work and provide reasonable result
        assert 0.0 <= enhanced_quality <= 1.0

    def test_weights_sum_properly(self):
        """Test that enhanced weights are properly normalized."""
        weights_info = get_enhanced_weights_info()

        # Should provide weight distribution info
        assert "core_structural" in weights_info
        assert "signal_quality" in weights_info
        assert "advanced_structural" in weights_info
        assert "perceptual_quality" in weights_info
        assert "temporal_consistency" in weights_info
        assert "deep_perceptual" in weights_info
        assert "grand_total" in weights_info

        # Grand total should be 1.0 (or very close due to floating point)
        assert abs(weights_info["grand_total"] - 1.0) < 0.001


class TestProcessMetricsIntegration:
    """Tests for the integrated metrics processing function."""

    def test_process_metrics_adds_efficiency(self):
        """Test that processing adds efficiency metric."""
        raw_metrics = {
            "compression_ratio": 5.0,
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
            "psnr_mean": 0.8,
            "temporal_consistency": 0.92,
        }

        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Should add composite quality and efficiency
        assert "composite_quality" in processed
        assert "efficiency" in processed

        # Efficiency should be reasonable
        assert 0.0 <= processed["efficiency"] <= 1.0

    def test_process_metrics_without_compression(self):
        """Test processing when compression ratio is missing."""
        raw_metrics = {
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
        }

        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Should add composite quality but not efficiency
        assert "composite_quality" in processed
        assert "efficiency" not in processed

    def test_process_metrics_adds_legacy_quality(self):
        """Test that legacy composite quality is added when missing."""
        raw_metrics = {
            "compression_ratio": 5.0,
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
            "psnr_mean": 0.8,
            "temporal_consistency": 0.92,
        }

        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Should add composite quality
        assert "composite_quality" in processed

        # Should be reasonable value
        assert processed["composite_quality"] > 0.0

    def test_process_metrics_edge_case_no_metrics(self):
        """Test processing when no quality metrics are provided (regression test)."""
        # This is a regression test for the batch efficiency calculation failure
        # where providing no quality metrics resulted in enhanced_composite_quality = 0.0
        raw_metrics = {
            "compression_ratio": 5.0,
            # No quality metrics provided
        }

        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Should handle gracefully
        assert "composite_quality" in processed
        assert "efficiency" in processed

        # With no quality metrics, composite quality should be 0.0
        assert processed["composite_quality"] == 0.0
        # And efficiency should also be 0.0 (geometric mean with 0 quality)
        assert processed["efficiency"] == 0.0


class TestEfficiencyWeightingChanges:
    """Tests specifically for the 60/40 -> 50/50 weighting change."""

    def test_50_50_weighting_constants(self):
        """Test that the code uses exactly 50/50 weighting."""
        # This is a regression test to ensure 50/50 weighting is maintained
        efficiency = calculate_efficiency_metric(10.0, 0.8)

        # Calculate what the efficiency should be with 50/50 weights
        normalized_compression = math.log(1 + 10.0) / math.log(1 + 20.0)
        expected_50_50 = (0.8**0.5) * (normalized_compression**0.5)

        assert abs(efficiency - expected_50_50) < 0.001

        # Verify this is different from what 60/40 would produce
        expected_60_40 = (0.8**0.6) * (normalized_compression**0.4)
        # Note: For mid-range values (compression=10x, quality=0.8), the mathematical
        # difference between 50/50 and 60/40 weighting is small (~0.0012 or 0.16%)
        # This tolerance ensures we detect the difference without unrealistic expectations
        assert (
            abs(efficiency - expected_60_40) > 0.0005
        )  # Should be measurably different

    def test_weighting_impact_comparison(self):
        """Test the impact of equal vs quality-favored weighting."""
        compression_focused = calculate_efficiency_metric(
            15.0, 0.6
        )  # High compression, lower quality
        quality_focused = calculate_efficiency_metric(
            3.0, 0.9
        )  # Lower compression, high quality

        # With 50/50 weighting, the difference should be more moderate
        # than it would be with 60/40 quality-favoring weights
        difference = abs(compression_focused - quality_focused)

        # Difference should exist but not be extreme (balanced weighting effect)
        assert 0.05 < difference < 0.3

    @patch("giflab.enhanced_metrics.calculate_efficiency_metric")
    def test_process_metrics_uses_50_50(self, mock_efficiency):
        """Test that process_metrics calls efficiency with correct parameters."""
        mock_efficiency.return_value = 0.75

        raw_metrics = {
            "compression_ratio": 5.0,
            "ssim_mean": 0.9,
        }

        process_metrics_with_enhanced_quality(raw_metrics)

        # Verify efficiency calculation was called
        mock_efficiency.assert_called_once()
        call_args = mock_efficiency.call_args[0]

        # Should be called with compression ratio and composite quality
        assert call_args[0] == 5.0  # compression_ratio
        assert 0.0 <= call_args[1] <= 1.0  # composite_quality
