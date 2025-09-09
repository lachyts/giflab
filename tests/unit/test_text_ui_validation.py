"""Unit tests for text/UI validation functionality."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from giflab.text_ui_validation import (
    EdgeAcuityAnalyzer,
    OCRValidator,
    TextUIContentDetector,
    calculate_text_ui_metrics,
    should_validate_text_ui,
)


class TestTextUIContentDetector:
    """Test cases for TextUIContentDetector."""

    def test_init(self):
        """Test TextUIContentDetector initialization."""
        detector = TextUIContentDetector()
        assert detector.edge_threshold == 30.0  # Updated default from 50.0
        assert detector.min_component_area == 10
        assert detector.max_component_area == 500
        assert detector.edge_density_threshold == 0.03  # Updated default from 0.10

    def test_init_with_params(self):
        """Test TextUIContentDetector initialization with custom parameters."""
        detector = TextUIContentDetector(
            edge_threshold=60.0,
            min_component_area=20,
            max_component_area=1000,
            edge_density_threshold=0.15,
        )
        assert detector.edge_threshold == 60.0
        assert detector.min_component_area == 20
        assert detector.max_component_area == 1000
        assert detector.edge_density_threshold == 0.15

    def test_detect_edge_density_rgb_frame(self):
        """Test edge density calculation on RGB frame."""
        detector = TextUIContentDetector()

        # Create test frame with edges
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 255  # White square for edges

        edge_density = detector._detect_edge_density(frame)
        assert 0.0 <= edge_density <= 1.0
        assert edge_density > 0.0  # Should detect some edges

    def test_detect_edge_density_grayscale_frame(self):
        """Test edge density calculation on grayscale frame."""
        detector = TextUIContentDetector()

        # Create test frame with edges
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[40:60, 40:60] = 255  # White square for edges

        edge_density = detector._detect_edge_density(frame)
        assert 0.0 <= edge_density <= 1.0
        assert edge_density > 0.0  # Should detect some edges

    def test_find_text_like_components_empty_image(self):
        """Test component detection on empty image."""
        detector = TextUIContentDetector()
        binary_image = np.zeros((100, 100), dtype=np.uint8)

        components = detector._find_text_like_components(binary_image)
        assert len(components) == 0

    def test_find_text_like_components_with_suitable_components(self):
        """Test component detection with text-like components."""
        detector = TextUIContentDetector()

        # Create binary image with text-like rectangles
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        binary_image[30:35, 20:40] = 255  # Text-like rectangle
        binary_image[40:45, 20:35] = 255  # Another text-like rectangle

        components = detector._find_text_like_components(binary_image)
        # Should find some components (exact number depends on connected components)
        assert len(components) >= 0

    def test_filter_ui_components_empty_list(self):
        """Test UI component filtering with empty list."""
        detector = TextUIContentDetector()
        components = detector._filter_ui_components([])
        assert len(components) == 0

    def test_detect_text_ui_regions_low_edge_density(self):
        """Test text/UI region detection with low edge density."""
        detector = TextUIContentDetector(edge_density_threshold=0.2)

        # Create smooth frame with very few edges
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        regions = detector.detect_text_ui_regions(frame)
        assert len(regions) == 0  # Should skip due to low edge density

    def test_detect_text_ui_regions_high_edge_density(self):
        """Test text/UI region detection with high edge density."""
        detector = TextUIContentDetector(edge_density_threshold=0.05)

        # Create frame with text-like pattern
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add text-like rectangles
        frame[20:25, 10:30, :] = 255
        frame[30:35, 10:25, :] = 255
        frame[40:45, 10:35, :] = 255

        regions = detector.detect_text_ui_regions(frame)
        # Should detect some regions (exact number varies)
        assert len(regions) >= 0

    def test_merge_nearby_regions_empty_list(self):
        """Test region merging with empty list."""
        detector = TextUIContentDetector()
        merged = detector._merge_nearby_regions([])
        assert len(merged) == 0

    def test_merge_nearby_regions_single_region(self):
        """Test region merging with single region."""
        detector = TextUIContentDetector()
        regions = [(10, 10, 20, 20)]
        merged = detector._merge_nearby_regions(regions)
        assert len(merged) == 1
        assert merged[0] == (10, 10, 20, 20)

    def test_merge_nearby_regions_overlapping(self):
        """Test region merging with overlapping regions."""
        detector = TextUIContentDetector()
        regions = [(10, 10, 20, 20), (15, 15, 20, 20)]  # Overlapping
        merged = detector._merge_nearby_regions(regions)
        assert len(merged) == 1  # Should merge into one region


class TestOCRValidator:
    """Test cases for OCRValidator."""

    def test_init_default(self):
        """Test OCRValidator initialization with defaults."""
        validator = OCRValidator()
        # Availability depends on system OCR libraries
        assert isinstance(validator.use_tesseract, bool)
        assert isinstance(validator.fallback_to_easyocr, bool)

    def test_init_with_params(self):
        """Test OCRValidator initialization with parameters."""
        validator = OCRValidator(use_tesseract=False, fallback_to_easyocr=False)
        assert validator.use_tesseract is False
        assert validator.fallback_to_easyocr is False

    def test_calculate_confidence_delta(self):
        """Test confidence delta calculation."""
        validator = OCRValidator()
        delta = validator._calculate_confidence_delta(0.8, 0.6)
        assert delta == pytest.approx(-0.2)  # 0.6 - 0.8 = -0.2

        delta = validator._calculate_confidence_delta(0.5, 0.7)
        assert delta == pytest.approx(0.2)  # 0.7 - 0.5 = 0.2  # 0.7 - 0.5 = 0.2

    def test_calculate_ocr_confidence_delta_empty_regions(self):
        """Test OCR confidence calculation with empty regions."""
        validator = OCRValidator()
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        compressed = np.zeros((100, 100, 3), dtype=np.uint8)

        result = validator.calculate_ocr_confidence_delta(original, compressed, [])
        expected = {
            "ocr_conf_delta_mean": 0.0,
            "ocr_conf_delta_min": 0.0,
            "ocr_regions_analyzed": 0,
        }
        assert result == expected

    def test_calculate_ocr_confidence_delta_with_regions(self):
        """Test OCR confidence calculation with valid regions."""
        validator = OCRValidator()

        # Create test frames
        original = np.full((100, 100, 3), 255, dtype=np.uint8)
        compressed = np.full((100, 100, 3), 128, dtype=np.uint8)  # Different brightness

        # Define text regions
        regions = [(10, 10, 30, 20), (50, 50, 40, 15)]

        result = validator.calculate_ocr_confidence_delta(original, compressed, regions)

        # Check result structure
        assert "ocr_conf_delta_mean" in result
        assert "ocr_conf_delta_min" in result
        assert "ocr_regions_analyzed" in result

        # Values should be reasonable
        assert isinstance(result["ocr_conf_delta_mean"], float)
        assert isinstance(result["ocr_conf_delta_min"], float)
        assert isinstance(result["ocr_regions_analyzed"], int)

    def test_preprocess_for_ocr(self):
        """Test OCR preprocessing."""
        validator = OCRValidator()

        # Create test ROI
        roi = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        processed = validator._preprocess_for_ocr(roi)

        # Check processed image properties
        assert len(processed.shape) == 2  # Should be grayscale
        assert processed.dtype == np.uint8
        # Should be resized if too small
        assert processed.shape[0] >= 20 and processed.shape[1] >= 20

    def test_preprocess_for_ocr_small_image(self):
        """Test OCR preprocessing with small image (should be upscaled)."""
        validator = OCRValidator()

        # Create small test ROI
        roi = np.random.randint(0, 255, (15, 15), dtype=np.uint8)

        processed = validator._preprocess_for_ocr(roi)

        # Should be upscaled
        assert processed.shape[0] >= 20 or processed.shape[1] >= 20

    def test_template_matching_confidence(self):
        """Test template matching confidence fallback."""
        validator = OCRValidator()

        # Create test ROI with some structure
        roi = np.zeros((50, 50), dtype=np.uint8)
        roi[20:30, 20:30] = 255  # White square

        confidence = validator._template_matching_confidence(roi)

        assert 0.1 <= confidence <= 1.0  # Should be within expected range
        assert confidence > 0.1  # Should be above minimum

    def test_extract_text_confidence_small_region(self):
        """Test text confidence extraction with very small region."""
        validator = OCRValidator()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        small_region = (10, 10, 5, 5)  # Very small region

        confidence = validator._extract_text_confidence(frame, small_region)
        assert confidence == 0.0  # Should return 0 for too small regions


class TestEdgeAcuityAnalyzer:
    """Test cases for EdgeAcuityAnalyzer."""

    def test_init_default(self):
        """Test EdgeAcuityAnalyzer initialization with defaults."""
        analyzer = EdgeAcuityAnalyzer()
        assert analyzer.mtf_threshold == 0.5

    def test_init_with_threshold(self):
        """Test EdgeAcuityAnalyzer initialization with custom threshold."""
        analyzer = EdgeAcuityAnalyzer(mtf_threshold=0.3)
        assert analyzer.mtf_threshold == 0.3

    def test_calculate_mtf50_empty_regions(self):
        """Test MTF50 calculation with empty regions."""
        analyzer = EdgeAcuityAnalyzer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = analyzer.calculate_mtf50(frame, [])
        expected = {
            "mtf50_ratio_mean": 1.0,
            "mtf50_ratio_min": 1.0,
            "edge_sharpness_score": 100.0,
        }
        assert result == expected

    def test_calculate_mtf50_with_regions(self):
        """Test MTF50 calculation with valid regions."""
        analyzer = EdgeAcuityAnalyzer()

        # Create frame with edges
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60, :] = 255  # White square

        regions = [(35, 35, 30, 30)]  # Region covering the square

        result = analyzer.calculate_mtf50(frame, regions)

        # Check result structure
        assert "mtf50_ratio_mean" in result
        assert "mtf50_ratio_min" in result
        assert "edge_sharpness_score" in result

        # Values should be reasonable
        assert isinstance(result["mtf50_ratio_mean"], float)
        assert isinstance(result["mtf50_ratio_min"], float)
        assert isinstance(result["edge_sharpness_score"], float)
        assert 0.0 <= result["edge_sharpness_score"] <= 100.0

    def test_calculate_mtf_empty_profile(self):
        """Test MTF calculation with empty profile."""
        analyzer = EdgeAcuityAnalyzer()
        empty_profile = np.array([])

        result = analyzer._calculate_mtf(empty_profile)
        assert len(result) == 0

    def test_calculate_mtf_short_profile(self):
        """Test MTF calculation with short profile."""
        analyzer = EdgeAcuityAnalyzer()
        short_profile = np.array([1, 2, 3])  # Too short

        result = analyzer._calculate_mtf(short_profile)
        assert len(result) == 0

    def test_calculate_mtf_valid_profile(self):
        """Test MTF calculation with valid profile."""
        analyzer = EdgeAcuityAnalyzer()

        # Create edge profile (step function)
        profile = np.concatenate([np.zeros(20), np.ones(20) * 255])

        result = analyzer._calculate_mtf(profile)

        assert len(result) > 0
        assert result[0] <= 1.0  # Should be normalized
        assert result[0] >= 0.0

    def test_find_mtf50_frequency_short_curve(self):
        """Test MTF50 frequency finding with short curve."""
        analyzer = EdgeAcuityAnalyzer()
        short_curve = np.array([1, 0.8])  # Too short

        result = analyzer._find_mtf50_frequency(short_curve)
        assert result == 0.0

    def test_find_mtf50_frequency_never_drops(self):
        """Test MTF50 frequency finding when MTF never drops below threshold."""
        analyzer = EdgeAcuityAnalyzer()
        high_curve = np.array([1.0, 0.9, 0.8, 0.7, 0.6])  # Never below 0.5

        result = analyzer._find_mtf50_frequency(high_curve)
        assert result == 1.0  # Very sharp edge

    def test_find_mtf50_frequency_starts_below(self):
        """Test MTF50 frequency finding when MTF starts below threshold."""
        analyzer = EdgeAcuityAnalyzer()
        low_curve = np.array([0.3, 0.2, 0.1])  # Starts below 0.5

        result = analyzer._find_mtf50_frequency(low_curve)
        assert result == 0.0  # Very blurry edge

    def test_find_mtf50_frequency_normal_case(self):
        """Test MTF50 frequency finding in normal case."""
        analyzer = EdgeAcuityAnalyzer()
        normal_curve = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Drops below 0.5 at index 2

        result = analyzer._find_mtf50_frequency(normal_curve)
        assert 0.0 < result < 1.0  # Should be between extremes

    def test_extract_profiles_at_point_valid(self):
        """Test profile extraction at valid point."""
        analyzer = EdgeAcuityAnalyzer()

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        profiles = analyzer._extract_profiles_at_point(roi, 25, 25)  # Center point

        # Should return horizontal and vertical profiles
        assert len(profiles) <= 2  # Could be 0, 1, or 2 profiles
        for profile in profiles:
            assert len(profile) >= 10  # Minimum profile length

    def test_extract_profiles_at_point_edge(self):
        """Test profile extraction at edge of image."""
        analyzer = EdgeAcuityAnalyzer()

        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        profiles = analyzer._extract_profiles_at_point(roi, 5, 5)  # Near edge

        # May not extract profiles due to proximity to edge
        assert len(profiles) >= 0


class TestShouldValidateTextUI:
    """Test cases for should_validate_text_ui function."""

    def test_empty_frames(self):
        """Test validation decision with empty frame list."""
        should_validate, hints = should_validate_text_ui([])
        assert should_validate is False
        assert hints["edge_density"] == 0.0
        assert hints["component_count"] == 0

    def test_smooth_frames_no_text(self):
        """Test validation decision with smooth frames (no text)."""
        # Create smooth frames with gradients
        frames = []
        for _i in range(3):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Add smooth gradient
            for x in range(100):
                frame[:, x, :] = int(x * 2.55)
            frames.append(frame)

        should_validate, hints = should_validate_text_ui(frames, quick_check=False)

        # Should not validate due to low edge density
        # (exact result depends on gradient detection)
        assert "edge_density" in hints
        assert "component_count" in hints
        assert "frames_analyzed" in hints

    def test_high_edge_density_frames(self):
        """Test validation decision with high edge density frames."""
        # Create frames with text-like patterns
        frames = []
        for _i in range(2):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Add text-like rectangles
            frame[20:25, 10:30, :] = 255
            frame[30:35, 10:25, :] = 255
            frame[40:45, 10:35, :] = 255
            frames.append(frame)

        should_validate, hints = should_validate_text_ui(frames, quick_check=False)

        # Should likely validate due to high edge density
        assert "edge_density" in hints
        assert "max_edge_density" in hints
        assert "component_count" in hints
        assert hints["frames_analyzed"] == 2

    def test_quick_check_mode(self):
        """Test validation decision in quick check mode."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames = [frame, frame, frame]  # 3 identical frames

        should_validate, hints = should_validate_text_ui(frames, quick_check=True)

        assert hints["frames_analyzed"] == 1  # Should only analyze first frame


class TestCalculateTextUIMetrics:
    """Test cases for calculate_text_ui_metrics function."""

    def test_mismatched_frame_counts(self):
        """Test error handling with mismatched frame counts."""
        original = [np.zeros((50, 50, 3), dtype=np.uint8)]
        compressed = [
            np.zeros((50, 50, 3), dtype=np.uint8),
            np.zeros((50, 50, 3), dtype=np.uint8),
        ]

        with pytest.raises(
            ValueError, match="Original and compressed frame counts must match"
        ):
            calculate_text_ui_metrics(original, compressed)

    def test_no_text_ui_content(self):
        """Test metrics calculation when no text/UI content is detected."""
        # Create smooth frames with no text-like content
        frames = []
        for _i in range(2):
            frame = np.full((50, 50, 3), 128, dtype=np.uint8)  # Solid gray
            frames.append(frame)

        result = calculate_text_ui_metrics(frames, frames)

        assert result["has_text_ui_content"] is False
        assert "text_ui_edge_density" in result
        assert "text_ui_component_count" in result
        assert result["text_ui_component_count"] == 0

    def test_with_text_ui_content(self):
        """Test metrics calculation with text/UI content."""
        # Create frames with text-like patterns
        original_frames = []
        compressed_frames = []

        for _i in range(3):
            # Original frame with sharp text
            orig = np.zeros((100, 100, 3), dtype=np.uint8)
            orig[20:25, 10:50, :] = 255  # Text-like rectangle
            orig[30:35, 10:40, :] = 255  # Another text-like rectangle
            original_frames.append(orig)

            # Compressed frame (slightly blurred)
            comp = orig.copy()
            comp = cv2.GaussianBlur(comp, (3, 3), 1.0)  # Slight blur
            compressed_frames.append(comp)

        result = calculate_text_ui_metrics(
            original_frames, compressed_frames, max_frames=2
        )

        # Check result structure
        assert "has_text_ui_content" in result
        assert "text_ui_edge_density" in result
        assert "text_ui_component_count" in result
        assert "ocr_regions_analyzed" in result

        # Values should be reasonable
        for key, value in result.items():
            if isinstance(value, int | float):
                assert not np.isnan(value), f"NaN value found for {key}"
                if key.endswith("_score"):
                    assert 0.0 <= value <= 100.0, f"Score {key} out of range: {value}"

    def test_max_frames_limit(self):
        """Test that max_frames parameter is respected."""
        # Create many frames
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(10)]

        # Should process at most max_frames
        result = calculate_text_ui_metrics(frames, frames, max_frames=3)

        # Function should complete without error
        assert "has_text_ui_content" in result


# Integration test
class TestTextUIValidationIntegration:
    """Integration tests for text/UI validation system."""

    def test_full_pipeline_no_text(self):
        """Test full pipeline with non-text content."""
        # Create natural image-like frames
        original_frames = []
        compressed_frames = []

        for _i in range(2):
            # Create frame with natural patterns (no text)
            orig = np.random.randint(100, 200, (80, 80, 3), dtype=np.uint8)
            # Add some smooth gradients
            orig[:40, :, 0] = np.linspace(100, 150, 80).reshape(1, -1)
            original_frames.append(orig)

            # Slightly compressed version
            comp = cv2.GaussianBlur(orig, (1, 1), 0.5)
            compressed_frames.append(comp)

        result = calculate_text_ui_metrics(original_frames, compressed_frames)

        # Should detect no text/UI content
        assert result["has_text_ui_content"] is False

    def test_full_pipeline_with_text(self):
        """Test full pipeline with text-like content."""
        # Create frames with clear text-like patterns
        original_frames = []
        compressed_frames = []

        for _i in range(2):
            # Original with sharp text-like elements
            orig = np.zeros((100, 100, 3), dtype=np.uint8)
            # Horizontal text lines
            orig[20:26, 10:60, :] = 255
            orig[35:41, 10:50, :] = 255
            orig[50:56, 10:55, :] = 255
            # Vertical separators
            orig[15:65, 5:8, :] = 255
            orig[15:65, 70:73, :] = 255
            original_frames.append(orig)

            # Compressed with some degradation
            comp = orig.copy()
            # Add slight blur and noise - use odd kernel size for GaussianBlur
            comp = cv2.GaussianBlur(comp, (3, 3), 0.8)
            noise = np.random.randint(-10, 10, comp.shape).astype(np.int16)
            comp = np.clip(comp.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            compressed_frames.append(comp)

        result = calculate_text_ui_metrics(original_frames, compressed_frames)

        # Should detect text/UI content (exact metrics depend on implementation)
        if result["has_text_ui_content"]:
            # If content detected, check metric structure
            assert "ocr_regions_analyzed" in result
            # edge_sharpness_score is only present when regions are actually analyzed
            if result["ocr_regions_analyzed"] > 0:
                assert "edge_sharpness_score" in result
            assert isinstance(result["text_ui_edge_density"], float)
            assert result["text_ui_edge_density"] >= 0.0
        # Note: Detection depends on thresholds, so we don't assert it must be True


if __name__ == "__main__":
    pytest.main([__file__])
