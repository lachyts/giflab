"""Tests for new synthetic expansion functionality.

Tests all new frame generation methods, targeted expansion strategy,
and bug fixes added for the expanded synthetic dataset.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from giflab.core import (
    GifLabRunner, 
    SyntheticGifSpec,
    AnalysisResult
)


class TestNewFrameGenerationMethods:
    """Test all new frame generation methods added for expansion."""
    
    @pytest.mark.fast
    def test_mixed_content_frame_generation(self, tmp_path):
        """Test mixed content frame generation."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test different sizes and frames
        test_cases = [
            (50, 50),
            (200, 150),  # Original spec size
            (10, 10)     # Edge case
        ]
        
        for width, height in test_cases:
            frame = eliminator._frame_generator.create_frame("mixed", (width, height), 0, 10)
            assert frame.size == (width, height)
            assert frame.mode == "RGB"
            
            # Should have varied colors (not solid)
            colors = frame.getcolors(maxcolors=256*256)
            if colors:  # getcolors returns None if too many colors
                assert len(colors) > 1
    
    @pytest.mark.fast
    def test_data_visualization_frame_generation(self, tmp_path):
        """Test data visualization (charts) frame generation."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test with different frame indices for animation
        for frame_idx in [0, 5, 9]:
            frame = eliminator._frame_generator.create_frame("charts", (300, 200), frame_idx, 10)
            assert frame.size == (300, 200)
            assert frame.mode == "RGB"
            
    @pytest.mark.fast
    def test_transitions_frame_generation(self, tmp_path):
        """Test transitions (morphing) frame generation."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test both halves of the morphing animation
        total_frames = 15
        for frame_idx in [0, 7, 14]:  # Beginning, middle, end
            frame = eliminator._frame_generator.create_frame("morph", (150, 150), frame_idx, total_frames)
            assert frame.size == (150, 150)
            assert frame.mode == "RGB"
    
    @pytest.mark.fast
    def test_single_pixel_anim_frame_generation(self, tmp_path):
        """Test single pixel animation frame generation."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test minimal motion detection
        frame1 = eliminator._frame_generator.create_frame("micro_detail", (100, 100), 0, 10)
        frame2 = eliminator._frame_generator.create_frame("micro_detail", (100, 100), 1, 10)
        
        assert frame1.size == (100, 100)
        assert frame2.size == (100, 100)
        assert frame1.mode == "RGB"
        assert frame2.mode == "RGB"
        
        # Frames should be different (pixel changes)
        assert list(frame1.getdata()) != list(frame2.getdata())
    
    @pytest.mark.fast
    def test_static_minimal_change_frame_generation(self, tmp_path):
        """Test static minimal change frame generation."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test edge cases that previously caused modulo by zero
        test_sizes = [
            (5, 5),    # Very small - should not crash
            (10, 10),  # Exactly at threshold
            (25, 25),  # Large enough for all elements
        ]
        
        for size in test_sizes:
            for frame_idx in [0, 4, 5, 8]:  # Test trigger frames
                frame = eliminator._frame_generator.create_frame("static_plus", size, frame_idx, 20)
                assert frame.size == size
                assert frame.mode == "RGB"
    
    @pytest.mark.fast
    def test_high_frequency_detail_frame_generation(self, tmp_path):
        """Test high frequency detail frame generation."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test aliasing patterns
        frame = eliminator._frame_generator.create_frame("detail", (200, 200), 0, 12)
        assert frame.size == (200, 200)
        assert frame.mode == "RGB"
        
        # Should have high frequency patterns (multiple colors)
        colors = frame.getcolors(maxcolors=1000)
        if colors:
            assert len(colors) > 1  # Should have varied patterns, not solid color


class TestExpandedSyntheticSpecs:
    """Test the expanded synthetic GIF specifications."""
    
    @pytest.mark.fast
    def test_all_new_content_types_present(self, tmp_path):
        """Test that all new content types are included in specs."""
        eliminator = GifLabRunner(tmp_path)
        
        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()
        
        content_types = {spec.content_type for spec in eliminator.synthetic_specs}
        
        # New content types should be present
        assert "mixed" in content_types
        assert "charts" in content_types  
        assert "morph" in content_types
        assert "micro_detail" in content_types
        assert "static_plus" in content_types
        assert "detail" in content_types
    
    @pytest.mark.fast
    def test_size_variations_present(self, tmp_path):
        """Test that size variations are properly included."""
        eliminator = GifLabRunner(tmp_path)
        
        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()
        
        spec_names = {spec.name for spec in eliminator.synthetic_specs}
        
        # Size variation specs should be present
        assert "gradient_small" in spec_names
        assert "gradient_medium" in spec_names
        assert "gradient_large" in spec_names
        assert "gradient_xlarge" in spec_names
        assert "noise_small" in spec_names
        assert "noise_large" in spec_names
    
    @pytest.mark.fast
    def test_frame_variations_present(self, tmp_path):
        """Test that frame count variations are included."""
        eliminator = GifLabRunner(tmp_path)
        
        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()
        
        spec_names = {spec.name for spec in eliminator.synthetic_specs}
        
        # Frame variation specs should be present
        assert "minimal_frames" in spec_names
        assert "long_animation" in spec_names
        assert "extended_animation" in spec_names
    
    @pytest.mark.fast
    def test_expanded_spec_count(self, tmp_path):
        """Test that we have the expected number of specs."""
        eliminator = GifLabRunner(tmp_path)
        
        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()
        
        # Should have expanded from 10 to 25 total specs
        assert len(eliminator.synthetic_specs) == 25
    
    def test_all_specs_generate_successfully(self, tmp_path):
        """Test that all expanded specs can generate GIFs without errors."""
        eliminator = GifLabRunner(tmp_path)
        
        # Initialize experiment directory to populate synthetic_specs
        eliminator._initialize_experiment_directory()
        
        # Test each spec individually
        for spec in eliminator.synthetic_specs:
            gif_path = tmp_path / f"{spec.name}.gif"
            
            # Should not raise an exception
            eliminator._create_synthetic_gif(gif_path, spec)
            
            # Should create a valid file
            assert gif_path.exists()
            assert gif_path.stat().st_size > 0


class TestTargetedExpansionStrategy:
    """Test the new targeted expansion sampling strategy."""
    
    @pytest.mark.fast
    def test_targeted_strategy_in_sampling_strategies(self, tmp_path):
        """Test that targeted strategy is available."""
        eliminator = GifLabRunner(tmp_path)
        
        assert "targeted" in eliminator.SAMPLING_STRATEGIES
        
        strategy = eliminator.SAMPLING_STRATEGIES["targeted"]
        assert strategy.name == "Targeted Expansion"
        assert strategy.sample_ratio == 0.12
        assert strategy.min_samples_per_tool == 4
    
    @pytest.mark.fast
    def test_targeted_sampling_method(self, tmp_path):
        """Test the targeted sampling method."""
        eliminator = GifLabRunner(tmp_path)
        
        # Create mock pipelines
        mock_pipelines = [MagicMock() for _ in range(100)]
        
        # Should not crash with targeted sampling
        result = eliminator.select_pipelines_intelligently(
            mock_pipelines, 
            strategy="targeted"
        )
        
        # Should return some pipelines
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= len(mock_pipelines)
    
    def test_get_targeted_synthetic_gifs(self, tmp_path):
        """Test targeted GIF generation."""
        eliminator = GifLabRunner(tmp_path)
        
        targeted_gifs = eliminator.get_targeted_synthetic_gifs()
        
        # Should generate exactly 17 GIFs
        assert len(targeted_gifs) == 17
        
        # All should be valid paths
        for gif_path in targeted_gifs:
            assert isinstance(gif_path, Path)
            assert gif_path.suffix == ".gif"
            # Note: Files are not immediately generated, just paths are returned
            # Actual GIF generation happens during the testing phase
    
    def test_targeted_gifs_content_selection(self, tmp_path):
        """Test that targeted GIFs include the right content."""
        eliminator = GifLabRunner(tmp_path)
        
        targeted_gifs = eliminator.get_targeted_synthetic_gifs()
        targeted_names = {gif.stem for gif in targeted_gifs}
        
        # Should include all original research-based content (10 GIFs)
        original_names = [
            'smooth_gradient', 'complex_gradient', 'solid_blocks', 'high_contrast',
            'photographic_noise', 'texture_complex', 'geometric_patterns', 
            'few_colors', 'many_colors', 'animation_heavy'
        ]
        for name in original_names:
            assert name in targeted_names
        
        # Should include strategic size variations (4 GIFs)
        size_names = ['gradient_small', 'gradient_large', 'gradient_xlarge', 'noise_large']
        for name in size_names:
            assert name in targeted_names
        
        # Should include key frame variations (2 GIFs)
        frame_names = ['minimal_frames', 'long_animation']
        for name in frame_names:
            assert name in targeted_names
        
        # Should include essential new content (1 GIF)
        assert 'mixed_content' in targeted_names
    
    def test_select_pipelines_intelligently_targeted(self, tmp_path):
        """Test intelligent pipeline selection with targeted strategy."""
        eliminator = GifLabRunner(tmp_path)
        
        # Create mock pipelines
        mock_pipelines = [MagicMock() for _ in range(100)]
        
        # Test targeted strategy specifically
        result = eliminator.select_pipelines_intelligently(mock_pipelines, 'targeted')
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) < len(mock_pipelines)  # Should be a subset


class TestEdgeCaseFixes:
    """Test the bug fixes for edge cases."""
    
    def test_empty_pipeline_list_handling(self, tmp_path):
        """Test that empty pipeline lists don't cause division by zero."""
        eliminator = GifLabRunner(tmp_path)
        
        # Should not crash with empty list
        result = eliminator.select_pipelines_intelligently(
            [], 
            strategy="representative"
        )
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_small_gif_size_handling(self, tmp_path):
        """Test that very small GIF sizes don't cause modulo by zero."""
        eliminator = GifLabRunner(tmp_path)
        
        # Test problematic sizes that previously crashed
        test_cases = [
            (1, 1),
            (5, 5), 
            (10, 10)
        ]
        
        for size in test_cases:
            # Should not crash
            frame = eliminator._frame_generator.create_frame("static_plus", size, 0, 10)
            assert frame.size == size
            assert frame.mode == "RGB"
    
    def test_single_frame_generation(self, tmp_path):
        """Test that single frame GIFs can be generated."""
        eliminator = GifLabRunner(tmp_path)
        
        # Create a single-frame spec
        spec = SyntheticGifSpec("test_single", 1, (50, 50), "gradient", "Single frame test")
        
        gif_path = tmp_path / "test_single.gif"
        
        # Should not crash with single frame
        eliminator._create_synthetic_gif(gif_path, spec)
        
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0


class TestIntegrationWithCLI:
    """Test integration of new functionality with CLI."""
    
    def test_targeted_strategy_cli_integration(self, tmp_path):
        """Test that CLI correctly handles targeted strategy."""
        eliminator = GifLabRunner(tmp_path)
        
        # Simulate CLI logic
        sampling_strategy = 'targeted'
        use_targeted_gifs = (sampling_strategy == 'targeted')
        
        assert use_targeted_gifs == True
        
        # Should be able to get targeted GIFs
        if use_targeted_gifs:
            synthetic_gifs = eliminator.get_targeted_synthetic_gifs()
        else:
            synthetic_gifs = eliminator.generate_synthetic_gifs()
            
        assert len(synthetic_gifs) == 17  # Targeted count
    
    def test_run_elimination_analysis_with_targeted_gifs(self, tmp_path):
        """Test that elimination analysis works with targeted GIF flag."""
        eliminator = GifLabRunner(tmp_path)
        
        # Mock dependencies to avoid full integration issues
        with patch('giflab.dynamic_pipeline.generate_all_pipelines') as mock_gen:
            mock_gen.return_value = []
            
            with patch.object(eliminator, '_run_comprehensive_testing') as mock_test:
                import pandas as pd
                mock_test.return_value = pd.DataFrame()
                
                with patch.object(eliminator, '_analyze_and_experiment') as mock_analyze:
                    mock_analyze.return_value = AnalysisResult()
                    
                    # Should work with targeted GIFs enabled
                    result = eliminator.run_analysis(use_targeted_gifs=True)
                    
                    assert isinstance(result, AnalysisResult)


# Fixtures
@pytest.fixture
def tmp_path():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) 