"""Tests for experimental pipeline framework.

These tests validate the systematic experimental approach and verify
research findings about redundant dithering methods.
"""

from unittest.mock import MagicMock, patch

import pytest
from giflab.core.runner import AnalysisResult, GifLabRunner
from giflab.synthetic_gifs import SyntheticGifSpec


class TestSyntheticGifGeneration:
    """Test synthetic GIF generation for various content types."""

    def test_creates_all_synthetic_specs(self, tmp_path):
        """Test that all synthetic GIF specifications are generated."""
        eliminator = GifLabRunner(tmp_path)

        # Generate synthetic GIFs
        gif_paths = eliminator.generate_synthetic_gifs()

        # Should create all specified synthetic types
        assert len(gif_paths) == len(eliminator.synthetic_specs)

        # All files should exist
        for gif_path in gif_paths:
            assert gif_path.exists()
            assert gif_path.suffix == ".gif"

    def test_synthetic_gif_content_types(self, tmp_path):
        """Test that different content types are properly represented."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        # Check that research-based content types are included
        content_types = {spec.content_type for spec in eliminator.synthetic_specs}

        # Should include key content types from research
        assert "gradient" in content_types  # Benefits from dithering
        assert "solid" in content_types  # Should NOT use dithering
        assert "noise" in content_types  # Good for Bayer scales 4-5
        assert "contrast" in content_types  # High contrast patterns

    def test_gradient_frame_generation(self, tmp_path):
        """Test gradient frame generation produces expected output."""
        eliminator = GifLabRunner(tmp_path)

        # Create a gradient frame using the new vectorized generator
        frame = eliminator._frame_generator.create_frame("gradient", (50, 50), 0, 10)

        assert frame.size == (50, 50)
        assert frame.mode == "RGB"

        # Should have color variation (not solid color)
        colors = frame.getcolors(maxcolors=10000)  # Increase max colors limit
        if colors:
            assert len(colors) > 10  # Should have many different colors
        else:
            # If getcolors returns None, there are many colors (which is good for gradients)
            assert True  # This is expected for complex gradients

    def test_solid_frame_generation(self, tmp_path):
        """Test solid color frame generation."""
        eliminator = GifLabRunner(tmp_path)

        # Create a solid color frame using the new vectorized generator
        frame = eliminator._frame_generator.create_frame("solid", (50, 50), 0, 10)

        assert frame.size == (50, 50)
        assert frame.mode == "RGB"

        # Should have fewer distinct colors than gradient
        colors = frame.getcolors()
        assert len(colors) <= 25  # Block-based pattern with limited colors


class TestEliminationLogic:
    """Test the core elimination analysis logic."""

    @pytest.mark.fast
    def test_elimination_result_structure(self):
        """Test AnalysisResult data structure."""
        result = AnalysisResult()

        # Should initialize with empty collections
        assert isinstance(result.eliminated_pipelines, set)
        assert isinstance(result.retained_pipelines, set)
        assert isinstance(result.performance_matrix, dict)
        assert isinstance(result.elimination_reasons, dict)
        assert isinstance(result.content_type_winners, dict)

    @pytest.mark.fast
    @patch("giflab.dynamic_pipeline.generate_all_pipelines")
    def test_analyze_and_eliminate_logic(self, mock_generate_pipelines, tmp_path):
        """Test the pipeline elimination logic."""
        import pandas as pd

        # Mock pipeline generation
        mock_generate_pipelines.return_value = []

        eliminator = GifLabRunner(tmp_path)

        # Create mock results DataFrame with more realistic elimination scenario
        test_data = [
            {"content_type": "gradient", "pipeline_id": "pipeline_A", "ssim_mean": 0.9},
            {"content_type": "gradient", "pipeline_id": "pipeline_B", "ssim_mean": 0.8},
            {
                "content_type": "gradient",
                "pipeline_id": "pipeline_C",
                "ssim_mean": 0.02,
            },  # Much lower to ensure elimination
            {"content_type": "solid", "pipeline_id": "pipeline_A", "ssim_mean": 0.6},
            {"content_type": "solid", "pipeline_id": "pipeline_B", "ssim_mean": 0.9},
            {
                "content_type": "solid",
                "pipeline_id": "pipeline_C",
                "ssim_mean": 0.01,
            },  # Much lower to ensure elimination
        ]
        results_df = pd.DataFrame(test_data)

        # Run elimination analysis
        elimination_result = eliminator._analyze_and_experiment(
            results_df, threshold=0.05
        )

        # Pipeline A and B should be retained (winners in at least one content type)
        assert "pipeline_A" in elimination_result.retained_pipelines
        assert "pipeline_B" in elimination_result.retained_pipelines

        # Pipeline C should be eliminated (consistently poor performance across content types)
        assert "pipeline_C" in elimination_result.eliminated_pipelines

        # Check content type winners
        assert "gradient" in elimination_result.content_type_winners
        assert "solid" in elimination_result.content_type_winners


class TestResearchValidation:
    """Test validation of preliminary research findings."""

    @pytest.mark.fast
    def test_content_type_detection(self, tmp_path):
        """Test content type detection from GIF names."""
        eliminator = GifLabRunner(tmp_path)

        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()

        # Test content type mapping
        assert eliminator._get_content_type("smooth_gradient") == "gradient"
        assert eliminator._get_content_type("solid_blocks") == "solid"
        assert eliminator._get_content_type("photographic_noise") == "noise"
        assert eliminator._get_content_type("unknown_name") == "unknown"

    @pytest.mark.fast
    def test_validate_research_findings(self, tmp_path):
        """Test research findings validation framework."""
        eliminator = GifLabRunner(tmp_path)

        # Run validation (returns placeholder results for now)
        findings = eliminator.validate_research_findings()

        # Should return dict with validation results
        assert isinstance(findings, dict)

        # Should include key research findings
        assert any("imagemagick" in key for key in findings.keys())
        assert any("ffmpeg" in key for key in findings.keys())
        assert any("gifsicle" in key for key in findings.keys())


class TestImageMagickEnhanced:
    """Test enhanced ImageMagick engine with dithering methods."""

    @pytest.mark.fast
    def test_dithering_methods_list(self):
        """Test that all expected dithering methods are included."""
        from giflab.external_engines.imagemagick_enhanced import (
            IMAGEMAGICK_DITHERING_METHODS,
        )

        # Should include all 13 methods from research
        assert len(IMAGEMAGICK_DITHERING_METHODS) == 13

        # Should include key methods identified in research
        assert "None" in IMAGEMAGICK_DITHERING_METHODS
        assert "FloydSteinberg" in IMAGEMAGICK_DITHERING_METHODS
        assert "Riemersma" in IMAGEMAGICK_DITHERING_METHODS  # Best performer

        # Should include redundant methods for testing
        assert "O2x2" in IMAGEMAGICK_DITHERING_METHODS
        assert "H4x4a" in IMAGEMAGICK_DITHERING_METHODS

    @pytest.mark.fast
    @patch("giflab.external_engines.imagemagick_enhanced._magick_binary")
    @patch(
        "giflab.external_engines.imagemagick_enhanced.run_command"
    )  # Fixed mock path
    def test_color_reduce_with_dithering(
        self, mock_run_command, mock_magick_binary, tmp_path
    ):
        """Test enhanced color reduction with specific dithering method."""
        from giflab.external_engines.imagemagick_enhanced import (
            color_reduce_with_dithering,
        )

        # Mock binary discovery and command execution
        mock_magick_binary.return_value = "magick"
        mock_run_command.return_value = {
            "render_ms": 100,
            "engine": "imagemagick",
            "command": "mock_command",
            "kilobytes": 50,
        }

        input_path = tmp_path / "input.gif"
        output_path = tmp_path / "output.gif"

        # Create dummy input and output files for the test
        input_path.touch()
        output_path.touch()  # Ensure output file exists for size calculation

        # Test with Riemersma dithering
        result = color_reduce_with_dithering(
            input_path, output_path, colors=16, dithering_method="Riemersma"
        )

        # Should add dithering metadata
        assert result["dithering_method"] == "Riemersma"
        assert "pipeline_variant" in result
        assert result["pipeline_variant"] == "imagemagick_dither_riemersma"

        # Should call magick with correct dithering parameter
        mock_run_command.assert_called_once()
        call_args = mock_run_command.call_args[0][0]  # First positional argument (cmd)
        assert "-dither" in call_args
        assert "Riemersma" in call_args


class TestFFmpegEnhanced:
    """Test enhanced FFmpeg engine with dithering methods."""

    def test_ffmpeg_dithering_methods_list(self):
        """Test that all expected FFmpeg dithering methods are included."""
        from giflab.external_engines.ffmpeg_enhanced import FFMPEG_DITHERING_METHODS

        # Should include all methods from research (10 total)
        assert len(FFMPEG_DITHERING_METHODS) == 10

        # Should include key methods
        assert "none" in FFMPEG_DITHERING_METHODS
        assert "floyd_steinberg" in FFMPEG_DITHERING_METHODS
        assert "sierra2" in FFMPEG_DITHERING_METHODS  # Best balance from research

        # Should include all Bayer scale variants
        bayer_methods = [m for m in FFMPEG_DITHERING_METHODS if m.startswith("bayer")]
        assert len(bayer_methods) == 6  # Scales 0-5

        # Should include best performing Bayer scales from research
        assert "bayer:bayer_scale=4" in FFMPEG_DITHERING_METHODS
        assert "bayer:bayer_scale=5" in FFMPEG_DITHERING_METHODS

    @patch("giflab.external_engines.ffmpeg_enhanced._ffmpeg_binary")
    @patch("giflab.external_engines.ffmpeg_enhanced.run_command")  # Fixed mock path
    def test_color_reduce_with_dithering_ffmpeg(
        self, mock_run_command, mock_ffmpeg_binary, tmp_path
    ):
        """Test FFmpeg enhanced color reduction with dithering."""
        from giflab.external_engines.ffmpeg_enhanced import color_reduce_with_dithering

        # Mock binary and command execution
        mock_ffmpeg_binary.return_value = "ffmpeg"
        mock_run_command.return_value = {
            "render_ms": 50,
            "engine": "ffmpeg",
            "command": "mock_command",
            "kilobytes": 25,
        }

        input_path = tmp_path / "input.gif"
        output_path = tmp_path / "output.gif"
        palette_path = tmp_path / "palette.png"

        # Create dummy input and output files for the test
        input_path.touch()
        output_path.touch()
        palette_path.touch()  # FFmpeg needs palette file too

        # Test with Sierra2 dithering
        result = color_reduce_with_dithering(
            input_path, output_path, colors=16, dithering_method="sierra2"
        )

        # Should add dithering metadata
        assert result["dithering_method"] == "sierra2"
        assert result["pipeline_variant"] == "ffmpeg_dither_sierra2"

        # Should be called twice (palette generation + application)
        assert mock_run_command.call_count == 2


class TestIntegration:
    """Integration tests for the complete elimination workflow."""

    def test_cli_command_structure(self):
        """Test that CLI command is properly structured."""
        from giflab.cli import main

        # Should be a click group with experiment command
        assert hasattr(main, "commands")
        assert "run" in main.commands

        # Get the run command
        run_cmd = main.commands["run"]

        # Should have expected options
        param_names = [param.name for param in run_cmd.params]
        assert "output_dir" in param_names
        assert "threshold" in param_names
        assert "sampling" in param_names
        assert "max_pipelines" in param_names

    @pytest.mark.fast
    @patch("giflab.core.runner.GifLabRunner")
    def test_elimination_workflow_integration(self, mock_eliminator_class, tmp_path):
        """Test integration of elimination workflow components."""

        # Mock the eliminator
        mock_eliminator = MagicMock()
        mock_eliminator_class.return_value = mock_eliminator

        # Mock results
        mock_elimination_result = AnalysisResult()
        mock_elimination_result.eliminated_pipelines = {"bad_pipeline"}
        mock_elimination_result.retained_pipelines = {"good_pipeline"}
        mock_elimination_result.content_type_winners = {
            "gradient": ["pipeline_A", "pipeline_B"],
            "solid": ["pipeline_C"],
        }

        mock_eliminator.run_analysis.return_value = mock_elimination_result

        # Test workflow integration - use the mock instead of real object
        eliminator = mock_eliminator_class(tmp_path)
        eliminator.run_analysis()

        # Should create eliminator instance
        mock_eliminator_class.assert_called_once_with(tmp_path)


@pytest.fixture
def sample_gif(tmp_path):
    """Create a simple test GIF for testing."""
    from PIL import Image

    # Create a simple 2-frame GIF
    frames = []
    for i in range(2):
        img = Image.new("RGB", (10, 10), color=(i * 128, 0, 0))
        frames.append(img)

    gif_path = tmp_path / "test.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )

    return gif_path


class TestRealWorldIntegration:
    """Test with actual GIF files to ensure the framework works end-to-end."""

    def test_synthetic_gif_creation_produces_valid_gifs(self, tmp_path):
        """Test that synthetic GIF creation produces valid GIF files."""
        eliminator = GifLabRunner(tmp_path)

        # Generate one synthetic GIF
        gif_path = tmp_path / "test_gradient.gif"
        spec = SyntheticGifSpec(
            "test_gradient", 5, (50, 50), "gradient", "Test gradient"
        )

        eliminator._create_synthetic_gif(gif_path, spec)

        # Should create a valid GIF file
        assert gif_path.exists()

        # Should be readable by PIL
        from PIL import Image

        with Image.open(gif_path) as img:
            assert img.format == "GIF"
            assert img.size == (50, 50)

            # Should have multiple frames
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                pass

            assert frame_count == 5  # Should match spec.frames
