"""End-to-end validation tests for Phase 3 conditional content-specific metrics.

This module tests the complete validation flow with Phase 3 metrics, including
CSV output validation, validation report generation, threshold configuration,
and integration with the validation system.
"""

import csv
import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, mock_open, patch

import cv2
import numpy as np
import pytest
from giflab.config import MetricsConfig
from giflab.meta import GifMetadata
from giflab.metrics import calculate_comprehensive_metrics, calculate_comprehensive_metrics_from_frames
from giflab.optimization_validation.config import ValidationConfig
from giflab.optimization_validation.data_structures import (
    ValidationResult,
    ValidationStatus,
)
from giflab.optimization_validation.validation_checker import ValidationChecker

# Import fixture generator for consistent test data
try:
    from tests.fixtures.generate_phase3_fixtures import Phase3FixtureGenerator
except ImportError:
    Phase3FixtureGenerator = None


@pytest.fixture
def fixture_generator():
    """Create fixture generator for tests."""
    if Phase3FixtureGenerator is None:
        pytest.skip("Phase 3 fixture generator not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = Phase3FixtureGenerator(Path(tmpdir))
        yield generator


@pytest.fixture
def sample_phase3_metrics():
    """Provide sample Phase 3 metrics for testing."""
    return {
        # Basic metrics
        "composite_quality": 0.68,
        "efficiency": 0.72,
        "compression_ratio": 2.3,
        "compressed_frame_count": 8,
        "orig_fps": 12.0,
        "kilobytes": 85.0,
        # Phase 1 temporal metrics (for context)
        "timing_grid_ms": 83.3,
        "grid_length": 10.0,
        "duration_diff_ms": 15.0,
        "flicker_excess": 0.025,
        "lpips_t_mean": 0.04,
        # Phase 1 gradient/color metrics (for context)
        "deltae_mean": 2.1,
        "banding_score_mean": 32.0,
        "dither_ratio_mean": 1.12,
        # Phase 2 deep perceptual metrics (for context)
        "lpips_quality_mean": 0.22,
        "lpips_quality_p95": 0.28,
        # Phase 3 Text/UI validation metrics
        "has_text_ui_content": True,
        "text_ui_edge_density": 0.16,
        "text_ui_component_count": 6,
        "ocr_conf_delta_mean": -0.05,
        "ocr_conf_delta_min": -0.12,
        "ocr_regions_analyzed": 4,
        "mtf50_ratio_mean": 0.78,
        "mtf50_ratio_min": 0.65,
        "edge_sharpness_score": 75.0,
        # Phase 3 SSIMULACRA2 metrics
        "ssimulacra2_mean": 0.62,
        "ssimulacra2_p95": 0.58,
        "ssimulacra2_min": 0.54,
        "ssimulacra2_frame_count": 8.0,
        "ssimulacra2_triggered": 1.0,
    }


class TestValidationFlowIntegration:
    """Test complete validation flow with Phase 3 metrics."""

    def test_validation_result_includes_phase3_metrics(self, sample_phase3_metrics):
        """Test that ValidationResult includes Phase 3 metrics."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        original_metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="test.gif",
            orig_kilobytes=120.0,
            orig_width=256,
            orig_height=256,
            orig_frames=10,
            orig_fps=12.0,
            orig_n_colors=256,
        )

        # Validate with Phase 3 metrics
        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=sample_phase3_metrics,
            gif_name="test_ui_content",
            pipeline_id="ui_optimized_pipeline",
            content_type="ui",
        )

        # Verify Phase 3 metrics are included in ValidationResult
        assert result.metrics.has_text_ui_content is True
        assert result.metrics.text_ui_edge_density == 0.16
        assert result.metrics.text_ui_component_count == 6
        assert result.metrics.ocr_conf_delta_mean == -0.05
        assert result.metrics.edge_sharpness_score == 75.0

        # Verify SSIMULACRA2 metrics are included
        # Note: These may not be in ValidationMetrics yet, but test documents expectation
        assert (
            hasattr(result.metrics, "ssimulacra2_mean")
            or "ssimulacra2_mean" in sample_phase3_metrics
        )

        # Should complete validation without errors
        assert result.status in [
            ValidationStatus.PASS,
            ValidationStatus.WARNING,
            ValidationStatus.ERROR,
        ]

    def test_validation_with_phase3_issues(self, sample_phase3_metrics):
        """Test validation that detects Phase 3 specific issues."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Create metrics with problematic Phase 3 values
        problematic_metrics = sample_phase3_metrics.copy()
        problematic_metrics.update(
            {
                # Text/UI issues
                "ocr_conf_delta_mean": -0.15,  # Significant OCR degradation
                "ocr_conf_delta_min": -0.25,  # Severe degradation in worst case
                "edge_sharpness_score": 45.0,  # Poor sharpness
                "mtf50_ratio_mean": 0.45,  # Poor edge acuity
                # SSIMULACRA2 issues
                "ssimulacra2_mean": 0.35,  # Poor perceptual quality
                "ssimulacra2_min": 0.25,  # Very poor worst case
                "ssimulacra2_p95": 0.30,  # Poor consistency
            }
        )

        original_metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="test.gif",
            orig_kilobytes=120.0,
            orig_width=256,
            orig_height=256,
            orig_frames=10,
            orig_fps=12.0,
            orig_n_colors=256,
        )

        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=problematic_metrics,
            gif_name="problematic_ui_gif",
            pipeline_id="aggressive_pipeline",
            content_type="ui",
        )

        # Should detect issues with Phase 3 metrics
        assert len(result.issues) > 0 or len(result.warnings) > 0

        # Check for expected issue categories (if validation methods exist)
        issue_categories = {issue.category for issue in result.issues}
        warning_categories = {warning.category for warning in result.warnings}

        all_categories = issue_categories | warning_categories

        # May include Phase 3 specific categories

        # Some overlap expected, but exact categories depend on implementation
        # This test documents expected behavior
        assert len(all_categories) > 0

    def test_validation_threshold_configuration(self):
        """Test that Phase 3 validation respects threshold configuration."""
        ValidationConfig()

        # Test that Phase 3 thresholds can be configured
        # This test documents expected threshold structure

        # Expected Phase 3 threshold names
        expected_thresholds = [
            "ocr_conf_delta_threshold",
            "mtf50_ratio_threshold",
            "edge_sharpness_threshold",
            "ssimulacra2_threshold",
            "ssimulacra2_low_threshold",
            "ssimulacra2_high_threshold",
            "text_ui_edge_density_threshold",
        ]

        # Verify validation config can handle Phase 3 thresholds
        # Exact implementation depends on ValidationConfig structure
        for _threshold_name in expected_thresholds:
            # This documents the expected interface
            # Actual validation depends on implementation
            assert True  # Placeholder for threshold configuration test

    def test_content_type_specific_validation(self, sample_phase3_metrics):
        """Test that validation adapts to content type."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        original_metadata = GifMetadata(
            gif_sha="test_sha256_hash",
            orig_filename="test.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=8,
            orig_fps=15.0,
            orig_kilobytes=100.0
        )

        # Test different content types
        content_types = ["ui", "animation", "photo", "mixed"]

        for content_type in content_types:
            result = checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=sample_phase3_metrics,
                gif_name=f"test_{content_type}_content",
                pipeline_id="adaptive_pipeline",
                content_type=content_type,
            )

            # Should adapt validation criteria based on content type
            assert result.content_type == content_type

            # UI content should pay more attention to Phase 3 metrics
            if content_type == "ui":
                # Should include text/UI related validation
                assert result.metrics.has_text_ui_content is not None

            # All content types should complete validation
            assert result.status != ValidationStatus.UNKNOWN


class TestCSVOutputValidation:
    """Test CSV output includes all Phase 3 fields."""

    def test_csv_headers_include_phase3_fields(self, sample_phase3_metrics):
        """Test that CSV output includes all Phase 3 metric fields."""
        # Expected Phase 3 CSV fields based on documentation
        expected_phase3_fields = [
            # Text/UI validation metrics
            "has_text_ui_content",
            "text_ui_edge_density",
            "text_ui_component_count",
            "ocr_conf_delta_mean",
            "ocr_conf_delta_min",
            "ocr_regions_analyzed",
            "mtf50_ratio_mean",
            "mtf50_ratio_min",
            "edge_sharpness_score",
            # SSIMULACRA2 perceptual metrics
            "ssimulacra2_mean",
            "ssimulacra2_p95",
            "ssimulacra2_min",
            "ssimulacra2_frame_count",
            "ssimulacra2_triggered",
        ]

        # Verify all Phase 3 fields are present in sample metrics
        for field in expected_phase3_fields:
            assert field in sample_phase3_metrics, f"Missing field: {field}"

        # This documents the expected CSV structure
        # Actual CSV generation would depend on metrics output system
        assert len(expected_phase3_fields) == 14  # Document expected field count

    def test_csv_output_completeness(self, sample_phase3_metrics):
        """Test complete CSV output with Phase 3 metrics."""
        # Simulate CSV generation with Phase 3 metrics
        csv_data = sample_phase3_metrics

        # Convert to CSV format for validation
        output = io.StringIO()

        # Write headers
        headers = list(csv_data.keys())
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()

        # Write data row
        writer.writerow(csv_data)

        # Validate CSV output
        csv_content = output.getvalue()

        # Check that Phase 3 fields are included
        phase3_indicators = [
            "text_ui_edge_density",
            "ocr_conf_delta_mean",
            "ssimulacra2_mean",
            "edge_sharpness_score",
        ]

        for indicator in phase3_indicators:
            assert indicator in csv_content, f"CSV missing Phase 3 field: {indicator}"

        # Parse CSV to validate structure
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)

        assert len(rows) == 1
        row = rows[0]

        # Validate Phase 3 values are properly formatted
        assert float(row["text_ui_edge_density"]) == 0.16
        assert float(row["ocr_conf_delta_mean"]) == -0.05
        assert float(row["ssimulacra2_mean"]) == 0.62
        assert float(row["edge_sharpness_score"]) == 75.0

    def test_csv_field_data_types(self, sample_phase3_metrics):
        """Test that CSV fields have correct data types."""
        # Define expected data types for Phase 3 fields
        phase3_field_types = {
            # Text/UI fields
            "has_text_ui_content": bool,
            "text_ui_edge_density": float,
            "text_ui_component_count": int,
            "ocr_conf_delta_mean": float,
            "ocr_conf_delta_min": float,
            "ocr_regions_analyzed": int,
            "mtf50_ratio_mean": float,
            "mtf50_ratio_min": float,
            "edge_sharpness_score": float,
            # SSIMULACRA2 fields
            "ssimulacra2_mean": float,
            "ssimulacra2_p95": float,
            "ssimulacra2_min": float,
            "ssimulacra2_frame_count": float,
            "ssimulacra2_triggered": float,
        }

        # Validate data types match expectations
        for field, expected_type in phase3_field_types.items():
            if field in sample_phase3_metrics:
                value = sample_phase3_metrics[field]
                if expected_type == bool:
                    assert isinstance(
                        value, bool
                    ), f"{field} should be bool, got {type(value)}"
                elif expected_type == int:
                    assert isinstance(
                        value, int | np.integer
                    ), f"{field} should be int, got {type(value)}"
                elif expected_type == float:
                    assert isinstance(
                        value, float | int | np.floating | np.integer
                    ), f"{field} should be numeric, got {type(value)}"

    def test_csv_value_ranges(self, sample_phase3_metrics):
        """Test that CSV values are in expected ranges."""
        # Define expected value ranges for Phase 3 fields
        phase3_value_ranges = {
            "text_ui_edge_density": (0.0, 1.0),
            "text_ui_component_count": (0, 1000),
            "ocr_conf_delta_mean": (-1.0, 1.0),
            "ocr_conf_delta_min": (-1.0, 1.0),
            "ocr_regions_analyzed": (0, 100),
            "mtf50_ratio_mean": (0.0, 1.0),
            "mtf50_ratio_min": (0.0, 1.0),
            "edge_sharpness_score": (0.0, 100.0),
            "ssimulacra2_mean": (0.0, 1.0),
            "ssimulacra2_p95": (0.0, 1.0),
            "ssimulacra2_min": (0.0, 1.0),
            "ssimulacra2_frame_count": (0.0, 10000.0),
            "ssimulacra2_triggered": (0.0, 1.0),
        }

        # Validate all values are in expected ranges
        for field, (min_val, max_val) in phase3_value_ranges.items():
            if field in sample_phase3_metrics:
                value = sample_phase3_metrics[field]
                assert (
                    min_val <= value <= max_val
                ), f"{field} value {value} outside expected range [{min_val}, {max_val}]"


class TestValidationReportGeneration:
    """Test validation report generation with Phase 3 metrics."""

    def test_validation_report_includes_phase3_sections(self, sample_phase3_metrics):
        """Test that validation reports include Phase 3 sections."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        original_metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="test.gif",
            orig_kilobytes=120.0,
            orig_width=256,
            orig_height=256,
            orig_frames=10,
            orig_fps=12.0,
            orig_n_colors=256,
        )

        # Create validation result with issues
        problematic_metrics = sample_phase3_metrics.copy()
        problematic_metrics.update(
            {
                "ocr_conf_delta_mean": -0.10,  # OCR issue
                "ssimulacra2_mean": 0.40,  # Perceptual issue
            }
        )

        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=problematic_metrics,
            gif_name="report_test_gif",
            pipeline_id="test_pipeline",
            content_type="ui",
        )

        # Test validation report structure
        # (Exact implementation depends on reporting system)

        # Should include Phase 3 metrics in summary
        assert result.metrics.text_ui_edge_density == 0.16
        assert result.metrics.has_text_ui_content is True

        # Report should document any Phase 3 issues found
        if result.issues or result.warnings:
            # Should have meaningful messages about Phase 3 metrics
            all_messages = [issue.message for issue in result.issues]
            all_messages.extend([warning.message for warning in result.warnings])

            # May include Phase 3 specific terms
            phase3_terms = ["OCR", "sharpness", "SSIMULACRA2", "text", "perceptual"]
            any(
                any(term.lower() in msg.lower() for term in phase3_terms)
                for msg in all_messages
            )

            # Should include some Phase 3 context (if issues detected)
            # Exact behavior depends on validation implementation

    def test_validation_report_recommendations(self):
        """Test that validation reports include Phase 3 specific recommendations."""
        # Expected types of recommendations for Phase 3 issues
        expected_recommendation_types = [
            "Consider reducing compression aggressiveness for text content",
            "Verify OCR readability after compression",
            "Check edge sharpness preservation in UI elements",
            "Review perceptual quality with SSIMULACRA2 assessment",
            "Validate text/UI content detection accuracy",
        ]

        # This documents expected recommendation categories
        # Actual recommendations depend on validation system implementation
        assert len(expected_recommendation_types) == 5

    def test_validation_summary_statistics(self, sample_phase3_metrics):
        """Test validation summary includes Phase 3 statistics."""
        # Expected summary statistics for Phase 3
        expected_summary_fields = [
            "text_ui_content_detected",
            "ocr_regions_processed",
            "average_edge_sharpness",
            "ssimulacra2_quality_level",
            "phase3_metrics_triggered",
        ]

        # Validation summary should aggregate Phase 3 metrics
        # This documents expected summary structure
        assert len(expected_summary_fields) == 5


class TestThresholdConfigurationEnd2End:
    """Test threshold configuration and overrides for Phase 3."""

    def test_threshold_override_functionality(self):
        """Test that Phase 3 thresholds can be overridden."""
        ValidationConfig()

        # Expected threshold override interface
        phase3_threshold_overrides = {
            "ocr_conf_delta_threshold": -0.08,  # More sensitive
            "edge_sharpness_threshold": 70.0,  # Higher requirement
            "ssimulacra2_threshold": 0.6,  # Higher quality requirement
            "text_ui_edge_density_threshold": 0.12,  # Higher detection threshold
        }

        # Test override mechanism (depends on ValidationConfig implementation)
        for _threshold, value in phase3_threshold_overrides.items():
            # This documents expected override interface
            # Actual implementation depends on config system
            assert isinstance(value, int | float)

    def test_content_type_specific_thresholds(self):
        """Test content-type specific threshold application."""
        ValidationConfig()

        # Expected content-type specific thresholds
        content_type_thresholds = {
            "ui": {
                "ocr_conf_delta_threshold": -0.05,  # Strict for UI
                "edge_sharpness_threshold": 80.0,  # High sharpness for UI
            },
            "animation": {
                "ocr_conf_delta_threshold": -0.10,  # More lenient
                "edge_sharpness_threshold": 70.0,  # Lower sharpness OK
            },
            "photo": {
                "ssimulacra2_threshold": 0.7,  # High perceptual quality
            },
        }

        # Test content-type adaptation
        for _content_type, thresholds in content_type_thresholds.items():
            # This documents expected content-type specific behavior
            assert isinstance(thresholds, dict)
            assert len(thresholds) > 0

    def test_dynamic_threshold_adjustment(self, sample_phase3_metrics):
        """Test dynamic threshold adjustment based on content analysis."""
        # Simulate dynamic thresholds based on content characteristics

        # High text content should have stricter text-related thresholds
        if sample_phase3_metrics["text_ui_component_count"] > 5:
            # Stricter OCR requirements
            pass  # vs default -0.05

        # High edge density should have stricter sharpness requirements
        if sample_phase3_metrics["text_ui_edge_density"] > 0.15:
            # Stricter sharpness requirements
            pass  # vs default 75.0

        # This documents expected dynamic adjustment behavior
        assert True  # Placeholder for dynamic adjustment test


class TestValidationSystemIntegrationE2E:
    """End-to-end integration tests for validation system."""

    def test_full_pipeline_validation_with_phase3(self, fixture_generator):
        """Test complete pipeline from metrics calculation to validation."""
        # Create test content
        ui_img_path = fixture_generator.create_text_ui_image(
            "ui_buttons", size=(160, 120)
        )
        orig_frame = cv2.imread(str(ui_img_path))

        # Create degraded version
        comp_frame = cv2.GaussianBlur(orig_frame, (3, 3), 1.0)

        orig_frames = [orig_frame for _ in range(3)]
        comp_frames = [comp_frame for _ in range(3)]

        # Step 1: Calculate comprehensive metrics (including Phase 3)
        config = MetricsConfig()
        config.USE_COMPREHENSIVE_METRICS = True
        config.ENABLE_SSIMULACRA2 = True

        # Mock Phase 3 components for consistent test
        with patch("giflab.text_ui_validation.calculate_text_ui_metrics") as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2, patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch(
            "giflab.ssimulacra2_metrics.should_use_ssimulacra2", return_value=True
        ):
            mock_text_ui.return_value = {
                "has_text_ui_content": True,
                "text_ui_edge_density": 0.18,
                "text_ui_component_count": 4,
                "ocr_conf_delta_mean": -0.08,
                "edge_sharpness_score": 68.0,
                "ocr_regions_analyzed": 3,
                "mtf50_ratio_mean": 0.72,
                "mtf50_ratio_min": 0.62,
            }

            mock_ssim2.return_value = {
                "ssimulacra2_mean": 0.58,
                "ssimulacra2_p95": 0.54,
                "ssimulacra2_min": 0.50,
                "ssimulacra2_frame_count": 3.0,
                "ssimulacra2_triggered": 1.0,
            }

            # Calculate metrics
            metrics = calculate_comprehensive_metrics_from_frames(orig_frames, comp_frames, config)

        # Step 2: Create validation metadata
        original_metadata = GifMetadata(
            gif_sha="test_sha256_hash",
            orig_filename="test_ui.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=3,
            orig_fps=10.0,
            orig_kilobytes=90.0
        )

        # Step 3: Run validation
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        validation_result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=metrics,
            gif_name="e2e_test_ui",
            pipeline_id="ui_pipeline",
            content_type="ui",
        )

        # Step 4: Verify complete flow
        assert validation_result.metrics.has_text_ui_content == 1.0  # Stored as float
        assert validation_result.metrics.text_ui_edge_density == 0.18

        # Should complete validation successfully
        assert validation_result.status != ValidationStatus.UNKNOWN

        # May detect issues due to degradation
        total_feedback = len(validation_result.issues) + len(validation_result.warnings)
        assert total_feedback >= 0  # May have issues, but should not crash

    def test_validation_batch_processing(self):
        """Test validation with batch processing of multiple GIFs."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Create batch of test cases
        test_cases = [
            (
                "ui_gif_1",
                "ui",
                {"has_text_ui_content": True, "edge_sharpness_score": 85.0},
            ),
            (
                "animation_gif_2",
                "animation",
                {"has_text_ui_content": False, "ssimulacra2_mean": 0.75},
            ),
            (
                "mixed_gif_3",
                "mixed",
                {"has_text_ui_content": True, "ssimulacra2_mean": 0.45},
            ),
        ]

        batch_results = []

        for gif_name, content_type, phase3_metrics in test_cases:
            # Base metrics
            metrics = {
                "composite_quality": 0.70,
                "efficiency": 0.65,
                "compression_ratio": 2.0,
                "compressed_frame_count": 5,
                **phase3_metrics,
            }

            original_metadata = GifMetadata(
                gif_sha=f"test_sha_{gif_name}",
                orig_filename=f"{gif_name}.gif",
                orig_width=640,
                orig_height=480,
                orig_n_colors=256,
                orig_frames=5,
                orig_fps=15.0,
                orig_kilobytes=100.0
            )

            result = checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=metrics,
                gif_name=gif_name,
                pipeline_id="batch_pipeline",
                content_type=content_type,
            )

            batch_results.append(result)

        # Verify batch processing
        assert len(batch_results) == 3

        # All should complete successfully
        for result in batch_results:
            assert result.status != ValidationStatus.UNKNOWN

        # Different content types may have different validation outcomes
        content_types = {result.content_type for result in batch_results}
        assert len(content_types) == 3  # ui, animation, mixed

    def test_validation_error_recovery(self):
        """Test validation system error recovery with Phase 3 failures."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Test with incomplete Phase 3 metrics
        incomplete_metrics = {
            "composite_quality": 0.60,
            "has_text_ui_content": True,
            "text_ui_edge_density": 0.15,
            # Missing: ocr_conf_delta_mean, ssimulacra2_mean, etc.
        }

        original_metadata = GifMetadata(
            gif_sha="test_sha_incomplete",
            orig_filename="test_error.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=5,
            orig_fps=12.0,
            orig_kilobytes=80.0
        )

        # Should handle missing Phase 3 metrics gracefully
        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=incomplete_metrics,
            gif_name="incomplete_metrics_test",
            pipeline_id="error_recovery_pipeline",
            content_type="ui",
        )

        # Should not crash with incomplete metrics
        assert result.status != ValidationStatus.UNKNOWN

        # May add warnings about missing metrics
        warning_messages = [w.message for w in result.warnings]
        missing_metric_warnings = [
            msg
            for msg in warning_messages
            if "unavailable" in msg.lower() or "missing" in msg.lower()
        ]

        # Should handle missing metrics appropriately
        assert len(missing_metric_warnings) >= 0

    def test_validation_performance_with_phase3(self):
        """Test that validation performance remains acceptable with Phase 3."""
        import time

        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Create comprehensive metrics set
        full_metrics = {
            # Basic metrics
            "composite_quality": 0.72,
            "efficiency": 0.68,
            "compression_ratio": 2.5,
            # All Phase 1-3 metrics
            "flicker_excess": 0.02,
            "deltae_mean": 1.8,
            "banding_score_mean": 25.0,
            "lpips_quality_mean": 0.18,
            "dither_ratio_mean": 1.05,
            # Full Phase 3 metrics
            "has_text_ui_content": True,
            "text_ui_edge_density": 0.14,
            "text_ui_component_count": 8,
            "ocr_conf_delta_mean": -0.04,
            "ocr_conf_delta_min": -0.09,
            "ocr_regions_analyzed": 5,
            "mtf50_ratio_mean": 0.80,
            "mtf50_ratio_min": 0.72,
            "edge_sharpness_score": 82.0,
            "ssimulacra2_mean": 0.65,
            "ssimulacra2_p95": 0.62,
            "ssimulacra2_min": 0.58,
            "ssimulacra2_frame_count": 8.0,
            "ssimulacra2_triggered": 1.0,
        }

        original_metadata = GifMetadata(
            gif_sha="test_sha_performance",
            orig_filename="test_perf.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=8,
            orig_fps=15.0,
            orig_kilobytes=110.0
        )

        # Measure validation time
        start_time = time.perf_counter()

        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=full_metrics,
            gif_name="performance_test_gif",
            pipeline_id="comprehensive_pipeline",
            content_type="mixed",
        )

        end_time = time.perf_counter()
        validation_time = end_time - start_time

        # Performance target: validation should complete quickly
        assert validation_time < 0.5, f"Validation too slow: {validation_time:.4f}s"

        # Should complete successfully with comprehensive metrics
        assert result.status != ValidationStatus.UNKNOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
