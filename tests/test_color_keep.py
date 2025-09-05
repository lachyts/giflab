"""Tests for giflab.color_keep module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from giflab.color_keep import (
    analyze_gif_palette,
    build_animately_color_args,
    build_gifsicle_color_args,
    count_gif_colors,
    extract_dominant_colors,
    get_color_reduction_info,
    get_optimal_color_count,
    validate_color_keep_count,
)


class TestValidateColorKeepCount:
    """Tests for validate_color_keep_count function."""

    def test_valid_counts(self):
        """Test that configured valid color counts pass validation."""
        valid_counts = [256, 128, 64]
        for count in valid_counts:
            # Should not raise any exception
            validate_color_keep_count(count)

    def test_invalid_count_not_configured(self):
        """Test count not in configured valid counts."""
        with pytest.raises(ValueError, match="not in supported counts"):
            validate_color_keep_count(4)  # 4 is not in [256, 128, 64, 32, 16, 8]

    def test_invalid_count_negative(self):
        """Test negative color count."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_color_keep_count(-1)

    def test_invalid_count_zero(self):
        """Test zero color count."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_color_keep_count(0)

    def test_non_integer_count(self):
        """Test non-integer color count."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_color_keep_count(64.5)  # type: ignore


class TestBuildGifsicleColorArgs:
    """Tests for build_gifsicle_color_args function."""

    def test_no_reduction_needed_target_higher(self):
        """Test no arguments when target >= original colors."""
        args = build_gifsicle_color_args(128, 64)  # Target higher than original
        assert args == []

    def test_no_reduction_needed_target_equal(self):
        """Test no arguments when target equals original colors."""
        args = build_gifsicle_color_args(128, 128)  # Target equals original
        assert args == []

    def test_no_reduction_needed_max_colors(self):
        """Test no arguments when target is max (256)."""
        args = build_gifsicle_color_args(256, 128)  # Target is max
        assert args == []

    def test_color_reduction_needed(self):
        """Test color reduction arguments when needed."""
        args = build_gifsicle_color_args(64, 256)
        assert args == ["--colors", "64", "--no-dither"]

    def test_various_color_counts(self):
        """Test different color reduction scenarios."""
        # 256 -> 128
        args = build_gifsicle_color_args(128, 256)
        assert args == ["--colors", "128", "--no-dither"]

        # 200 -> 64
        args = build_gifsicle_color_args(64, 200)
        assert args == ["--colors", "64", "--no-dither"]


class TestBuildAnimatelyColorArgs:
    """Tests for build_animately_color_args function."""

    def test_no_reduction_needed(self):
        """Test no arguments when no reduction needed."""
        args = build_animately_color_args(128, 64)
        assert args == []

    def test_color_reduction_needed(self):
        """Test color reduction arguments when needed."""
        args = build_animately_color_args(64, 256)
        assert args == ["--colors", "64"]

    def test_max_color_handling(self):
        """Test handling of maximum color count."""
        args = build_animately_color_args(256, 128)
        assert args == []  # No reduction for max count


class TestCountGifColors:
    """Tests for count_gif_colors function."""

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_palette_mode_gif(self, mock_open, mock_exists):
        """Test color counting for palette mode GIF."""
        mock_exists.return_value = True

        # Mock PIL Image in palette mode
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.mode = "P"
        # Create a simple palette with 5 unique colors
        mock_palette = [
            255,
            0,
            0,  # Red
            0,
            255,
            0,  # Green
            0,
            0,
            255,  # Blue
            255,
            255,
            0,  # Yellow
            0,
            0,
            0,
        ]  # Black
        mock_palette.extend([0] * (256 * 3 - len(mock_palette)))  # Pad to full palette
        mock_img.getpalette.return_value = mock_palette
        mock_open.return_value.__enter__.return_value = mock_img

        color_count = count_gif_colors(Path("test.gif"))

        # Should detect the unique colors in the palette
        assert color_count > 0
        assert color_count <= 256

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_rgb_mode_gif(self, mock_open, mock_exists):
        """Test color counting for RGB mode GIF."""
        mock_exists.return_value = True

        # Mock PIL Image in RGB mode
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.mode = "RGB"

        # Mock quantization
        mock_quantized = MagicMock()
        mock_quantized.getpalette.return_value = [
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            0,
            255,
        ]  # 3 colors
        mock_img.quantize.return_value = mock_quantized

        mock_open.return_value.__enter__.return_value = mock_img

        color_count = count_gif_colors(Path("test.gif"))

        assert color_count > 0
        assert color_count <= 256
        mock_img.quantize.assert_called_once_with(colors=256)

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="File not found"):
            count_gif_colors(Path("missing.gif"))

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_non_gif_file(self, mock_open, mock_exists):
        """Test error when file is not a GIF."""
        mock_exists.return_value = True

        mock_img = MagicMock()
        mock_img.format = "PNG"  # Not a GIF
        mock_open.return_value.__enter__.return_value = mock_img

        with pytest.raises(ValueError, match="File is not a GIF"):
            count_gif_colors(Path("test.png"))

    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_pil_error_handling(self, mock_open, mock_exists):
        """Test handling of PIL errors."""
        mock_exists.return_value = True
        mock_open.side_effect = Exception("PIL error")

        with pytest.raises(IOError, match="Error reading GIF"):
            count_gif_colors(Path("test.gif"))


class TestGetColorReductionInfo:
    """Tests for get_color_reduction_info function."""

    @patch("giflab.color_keep.count_gif_colors")
    @patch("pathlib.Path.exists")
    def test_valid_color_analysis(self, mock_exists, mock_count):
        """Test color reduction analysis for valid GIF."""
        mock_exists.return_value = True
        mock_count.return_value = 256  # Original has 256 colors

        info = get_color_reduction_info(Path("test.gif"), 128)

        assert info["original_colors"] == 256
        assert info["target_colors"] == 128
        assert info["color_keep_count"] == 128
        assert info["reduction_needed"] is True
        assert info["reduction_percent"] == 50.0
        assert info["compression_ratio"] == 2.0

    @patch("giflab.color_keep.count_gif_colors")
    @patch("pathlib.Path.exists")
    def test_no_reduction_needed(self, mock_exists, mock_count):
        """Test when no color reduction is needed."""
        mock_exists.return_value = True
        mock_count.return_value = 64  # Original has fewer colors than target

        info = get_color_reduction_info(Path("test.gif"), 128)

        assert info["original_colors"] == 64
        assert info["target_colors"] == 64
        assert info["color_keep_count"] == 128
        assert info["reduction_needed"] is False
        assert info["reduction_percent"] == 0.0
        assert info["compression_ratio"] == 1.0

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when input file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="Input file not found"):
            get_color_reduction_info(Path("missing.gif"), 128)

    def test_invalid_color_count(self):
        """Test error with invalid color count."""
        with pytest.raises(ValueError, match="not in supported counts"):
            get_color_reduction_info(
                Path("test.gif"), 4
            )  # 4 is not in supported counts


class TestExtractDominantColors:
    """Tests for extract_dominant_colors function."""

    def test_rgb_image_dominant_colors(self):
        """Test dominant color extraction from RGB image."""
        # Create a mock PIL Image
        mock_img = MagicMock()
        mock_img.mode = "RGB"

        # Mock numpy array conversion
        with patch("numpy.array") as mock_array, patch(
            "giflab.color_keep.Counter"
        ) as mock_counter:
            # Mock pixel data
            mock_array.return_value.reshape.return_value = [
                [255, 0, 0],  # Red
                [255, 0, 0],  # Red (duplicate)
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
            ]

            # Mock counter results
            mock_counter.return_value.most_common.return_value = [
                ((255, 0, 0), 2),  # Red appears twice
                ((0, 255, 0), 1),  # Green appears once
                ((0, 0, 255), 1),  # Blue appears once
            ]

            colors = extract_dominant_colors(mock_img, 3)

            assert colors == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            mock_counter.assert_called_once()

    def test_palette_mode_conversion(self):
        """Test color extraction with palette mode conversion."""
        mock_img = MagicMock()
        mock_img.mode = "P"

        # Mock conversion to RGB
        mock_rgb = MagicMock()
        mock_rgb.mode = "RGB"
        mock_img.convert.return_value = mock_rgb

        with patch("numpy.array") as mock_array, patch(
            "giflab.color_keep.Counter"
        ) as mock_counter:
            mock_array.return_value.reshape.return_value = [[128, 128, 128]]
            mock_counter.return_value.most_common.return_value = [((128, 128, 128), 1)]

            colors = extract_dominant_colors(mock_img, 1)

            mock_img.convert.assert_called_once_with("RGB")
            assert colors == [(128, 128, 128)]

    def test_invalid_n_colors(self):
        """Test error with invalid n_colors."""
        mock_img = MagicMock()

        with pytest.raises(ValueError, match="n_colors must be positive"):
            extract_dominant_colors(mock_img, 0)

        with pytest.raises(ValueError, match="n_colors must be positive"):
            extract_dominant_colors(mock_img, -1)


class TestAnalyzeGifPalette:
    """Tests for analyze_gif_palette function."""

    @patch("giflab.color_keep.extract_dominant_colors")
    @patch("giflab.color_keep.count_gif_colors")
    @patch("pathlib.Path.exists")
    @patch("PIL.Image.open")
    def test_palette_mode_analysis(
        self, mock_open, mock_exists, mock_count, mock_extract
    ):
        """Test palette analysis for palette mode GIF."""
        mock_exists.return_value = True
        mock_count.return_value = 128
        mock_extract.return_value = [(255, 0, 0), (0, 255, 0)]

        # Mock PIL Image in palette mode
        mock_img = MagicMock()
        mock_img.format = "GIF"
        mock_img.mode = "P"
        mock_img.getpalette.return_value = [255, 0, 0] * 128  # 128 colors
        mock_img.info = {}
        mock_open.return_value.__enter__.return_value = mock_img

        analysis = analyze_gif_palette(Path("test.gif"))

        assert analysis["total_colors"] == 128
        assert analysis["dominant_colors"] == [(255, 0, 0), (0, 255, 0)]
        assert analysis["palette_info"]["mode"] == "palette"
        assert analysis["palette_info"]["palette_size"] == 128
        assert analysis["palette_info"]["has_transparency"] is False

        # Check reduction candidates
        assert 256 in analysis["reduction_candidates"]
        assert 128 in analysis["reduction_candidates"]
        assert 64 in analysis["reduction_candidates"]

    @patch("pathlib.Path.exists")
    def test_missing_file(self, mock_exists):
        """Test error when file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(IOError, match="File not found"):
            analyze_gif_palette(Path("missing.gif"))


class TestGetOptimalColorCount:
    """Tests for get_optimal_color_count function."""

    @patch("giflab.color_keep.analyze_gif_palette")
    def test_optimal_count_high_quality(self, mock_analyze):
        """Test optimal color count with high quality threshold."""
        mock_analyze.return_value = {"total_colors": 256}

        optimal = get_optimal_color_count(Path("test.gif"), 0.9)

        # Should suggest a color count that retains 90% of colors
        assert optimal in [256, 128, 64, 32, 16, 8]

    @patch("giflab.color_keep.analyze_gif_palette")
    def test_optimal_count_low_quality(self, mock_analyze):
        """Test optimal color count with low quality threshold."""
        mock_analyze.return_value = {"total_colors": 256}

        optimal = get_optimal_color_count(Path("test.gif"), 0.2)

        # Should suggest aggressive reduction
        assert optimal in [256, 128, 64]

    def test_invalid_quality_threshold(self):
        """Test error with invalid quality threshold."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            get_optimal_color_count(Path("test.gif"), 1.5)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            get_optimal_color_count(Path("test.gif"), -0.1)
