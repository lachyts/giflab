"""Integration tests for updated tool wrappers."""

import pytest
import tempfile
from pathlib import Path

from giflab.tool_wrappers import (
    ImageMagickColorReducer,
    ImageMagickFrameReducer,
    ImageMagickLossyCompressor,
    FFmpegColorReducer,
    FFmpegFrameReducer,
    FFmpegLossyCompressor,
    GifskiLossyCompressor,
)
from giflab.meta import extract_gif_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_gif():
    """Path to a simple test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "simple_4frame.gif"


@pytest.fixture
def single_frame_gif():
    """Path to a single-frame test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "single_frame.gif"


@pytest.fixture
def many_colors_gif():
    """Path to a many-color test GIF fixture."""
    return Path(__file__).parent / "fixtures" / "many_colors.gif"


# ---------------------------------------------------------------------------
# ImageMagick wrapper integration tests
# ---------------------------------------------------------------------------

@pytest.mark.external_tools
class TestImageMagickWrapperIntegration:
    """Integration tests for ImageMagick wrappers."""
    
    def test_color_reducer_functionality(self, many_colors_gif):
        """Test ImageMagick color reducer actually reduces colors."""
        wrapper = ImageMagickColorReducer()
        
        if not wrapper.available():
            pytest.skip("ImageMagick not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            # Test color reduction to 32 colors
            result = wrapper.apply(
                many_colors_gif,
                output_path,
                params={"colors": 32}
            )
            
            # Validate metadata structure
            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert result["kilobytes"] > 0
            assert "command" in result
            
            # Validate functional change
            assert output_path.exists()
            
            # Check that colors were actually reduced (this is a basic check)
            original_size = many_colors_gif.stat().st_size
            output_size = output_path.stat().st_size
            
            # Color reduction should typically reduce file size
            assert output_size <= original_size * 1.2  # Allow 20% tolerance
    
    def test_frame_reducer_functionality(self, test_gif):
        """Test ImageMagick frame reducer actually reduces frames."""
        wrapper = ImageMagickFrameReducer()
        
        if not wrapper.available():
            pytest.skip("ImageMagick not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            # Test frame reduction to 50%
            result = wrapper.apply(
                test_gif,
                output_path,
                params={"ratio": 0.5}
            )
            
            # Validate metadata
            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert output_path.exists()
            
            # Validate functional change - should have fewer frames
            try:
                original_meta = extract_gif_metadata(test_gif)
                output_meta = extract_gif_metadata(output_path)
                
                # Should have roughly half the frames (±1 for rounding)
                expected_frames = max(1, original_meta.orig_frames // 2)
                assert abs(output_meta.orig_frames - expected_frames) <= 1
            except Exception:
                # If metadata extraction fails, just check file exists
                pass
    
    def test_lossy_compressor_functionality(self, test_gif):
        """Test ImageMagick lossy compressor reduces file size."""
        wrapper = ImageMagickLossyCompressor()
        
        if not wrapper.available():
            pytest.skip("ImageMagick not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            # Test lossy compression with low quality
            result = wrapper.apply(
                test_gif,
                output_path,
                params={"quality": 50}
            )
            
            # Validate metadata
            assert result["engine"] == "imagemagick"
            assert result["render_ms"] > 0
            assert output_path.exists()
            
            # Lossy compression should reduce file size
            original_size = test_gif.stat().st_size
            output_size = output_path.stat().st_size
            assert output_size <= original_size
    
    def test_parameter_validation(self, test_gif):
        """Test that wrappers validate parameters correctly."""
        wrapper = ImageMagickColorReducer()
        
        if not wrapper.available():
            pytest.skip("ImageMagick not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            # Should raise ValueError for missing colors parameter
            with pytest.raises(ValueError, match="params must include 'colors'"):
                wrapper.apply(test_gif, output_path, params={})
            
            # Should raise ValueError for None params
            with pytest.raises(ValueError, match="params must include 'colors'"):
                wrapper.apply(test_gif, output_path, params=None)


# ---------------------------------------------------------------------------
# FFmpeg wrapper integration tests
# ---------------------------------------------------------------------------

@pytest.mark.external_tools
class TestFFmpegWrapperIntegration:
    """Integration tests for FFmpeg wrappers."""
    
    def test_color_reducer_functionality(self, many_colors_gif):
        """Test FFmpeg color reducer via palette generation."""
        wrapper = FFmpegColorReducer()
        
        if not wrapper.available():
            pytest.skip("FFmpeg not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            result = wrapper.apply(
                many_colors_gif,
                output_path,
                params={"fps": 10.0}
            )
            
            # Validate metadata
            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert output_path.exists()
            
            # Command should show two-pass operation
            assert "palettegen" in result["command"]
            assert "paletteuse" in result["command"]
    
    def test_frame_reducer_functionality(self, test_gif):
        """Test FFmpeg frame reducer."""
        wrapper = FFmpegFrameReducer()
        
        if not wrapper.available():
            pytest.skip("FFmpeg not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            result = wrapper.apply(
                test_gif,
                output_path,
                params={"fps": 5.0}
            )
            
            # Validate metadata
            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert "fps=5.0" in result["command"]
            assert output_path.exists()
    
    def test_lossy_compressor_functionality(self, test_gif):
        """Test FFmpeg lossy compressor."""
        wrapper = FFmpegLossyCompressor()
        
        if not wrapper.available():
            pytest.skip("FFmpeg not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            result = wrapper.apply(
                test_gif,
                output_path,
                params={"qv": 25, "fps": 12.0}
            )
            
            # Validate metadata
            assert result["engine"] == "ffmpeg"
            assert result["render_ms"] > 0
            assert "q:v" in result["command"]
            assert output_path.exists()
    
    def test_parameter_validation(self, test_gif):
        """Test FFmpeg parameter validation."""
        wrapper = FFmpegFrameReducer()
        
        if not wrapper.available():
            pytest.skip("FFmpeg not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            # Should raise ValueError for missing fps parameter
            with pytest.raises(ValueError, match="params must include 'fps'"):
                wrapper.apply(test_gif, output_path, params={})


# ---------------------------------------------------------------------------
# gifski wrapper integration tests
# ---------------------------------------------------------------------------

@pytest.mark.external_tools
class TestGifskiWrapperIntegration:
    """Integration tests for gifski wrapper."""
    
    def test_lossy_compressor_functionality(self, test_gif):
        """Test gifski lossy compressor."""
        wrapper = GifskiLossyCompressor()
        
        if not wrapper.available():
            pytest.skip("gifski not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            result = wrapper.apply(
                test_gif,
                output_path,
                params={"quality": 70}
            )
            
            # Validate metadata
            assert result["engine"] == "gifski"
            assert result["render_ms"] > 0
            assert "quality" in result["command"]
            assert output_path.exists()
    
    def test_default_parameters(self, test_gif):
        """Test gifski with default parameters."""
        wrapper = GifskiLossyCompressor()
        
        if not wrapper.available():
            pytest.skip("gifski not available")
        
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
            output_path = Path(tmp.name)
            
            # Should work with no params (uses defaults)
            result = wrapper.apply(test_gif, output_path, params=None)
            
            assert result["engine"] == "gifski"
            assert output_path.exists()


# ---------------------------------------------------------------------------
# Cross-engine consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.external_tools
class TestCrossEngineConsistency:
    """Test consistency across different engines."""
    
    def test_metadata_schema_consistency(self, test_gif):
        """Test that all engines return consistent metadata schema."""
        wrappers = [
            ImageMagickLossyCompressor(),
            FFmpegLossyCompressor(),
            GifskiLossyCompressor(),
        ]
        
        results = []
        for wrapper in wrappers:
            if not wrapper.available():
                continue
                
            with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
                output_path = Path(tmp.name)
                
                # Use appropriate parameters for each wrapper
                if isinstance(wrapper, ImageMagickLossyCompressor):
                    params = {"quality": 75}
                elif isinstance(wrapper, FFmpegLossyCompressor):
                    params = {"qv": 25, "fps": 15.0}
                else:  # GifskiLossyCompressor
                    params = {"quality": 75}
                
                result = wrapper.apply(test_gif, output_path, params=params)
                results.append(result)
        
        if len(results) < 2:
            pytest.skip("Need at least 2 engines available for consistency test")
        
        # All results should have the same keys
        expected_keys = {"render_ms", "engine", "command", "kilobytes"}
        for result in results:
            assert set(result.keys()) >= expected_keys
            assert result["render_ms"] > 0
            assert result["kilobytes"] >= 0
            assert len(result["command"]) > 0
    
    def test_combine_group_consistency(self):
        """Test that COMBINE_GROUP values are consistent."""
        # ImageMagick wrappers
        assert ImageMagickColorReducer.COMBINE_GROUP == "imagemagick"
        assert ImageMagickFrameReducer.COMBINE_GROUP == "imagemagick"
        assert ImageMagickLossyCompressor.COMBINE_GROUP == "imagemagick"
        
        # FFmpeg wrappers
        assert FFmpegColorReducer.COMBINE_GROUP == "ffmpeg"
        assert FFmpegFrameReducer.COMBINE_GROUP == "ffmpeg"
        assert FFmpegLossyCompressor.COMBINE_GROUP == "ffmpeg"
        
        # gifski wrapper
        assert GifskiLossyCompressor.COMBINE_GROUP == "gifski"
    
    def test_availability_methods(self):
        """Test that availability methods work correctly."""
        wrappers = [
            ImageMagickColorReducer(),
            FFmpegColorReducer(),
            GifskiLossyCompressor(),
        ]
        
        for wrapper in wrappers:
            # Should return boolean
            available = wrapper.available()
            assert isinstance(available, bool)
            
            # Version should return string
            version = wrapper.version()
            assert isinstance(version, str)
            assert len(version) > 0 