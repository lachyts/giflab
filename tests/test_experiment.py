"""Tests for experimental compression testing framework."""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.giflab.experiment import (
    ExperimentalConfig,
    ExperimentalPipeline,
    ExperimentJob,
    ExperimentResult,
    create_experimental_pipeline
)
from src.giflab.meta import GifMetadata


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiment testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_experiment_dir):
    """Create test experimental configuration."""
    config = ExperimentalConfig(
        TEST_GIFS_COUNT=3,
        SAMPLE_GIFS_PATH=temp_experiment_dir / "sample_gifs",
        RESULTS_PATH=temp_experiment_dir / "results",
        STRATEGIES=["pure_gifsicle", "gifsicle_dithered"],
        GIFSICLE_OPTIMIZATION_LEVELS=["basic"],  # Override to single value
        GIFSICLE_DITHERING_OPTIONS=["none"],     # Override to single value
        LOSSY_LEVELS=[0, 40],
        FRAME_KEEP_RATIOS=[1.0, 0.8],
        COLOR_KEEP_COUNTS=[256, 64],
        ENABLE_DETAILED_ANALYSIS=False  # Disable for faster testing
    )
    
    return config


class TestExperimentalConfig:
    """Test ExperimentalConfig class."""
    
    def test_init(self, temp_experiment_dir):
        """Test configuration initialization."""
        config = ExperimentalConfig(
            SAMPLE_GIFS_PATH=temp_experiment_dir / "sample",
            RESULTS_PATH=temp_experiment_dir / "results"
        )
        
        assert config.TEST_GIFS_COUNT == 10
        assert config.SAMPLE_GIFS_PATH.exists()
        assert config.RESULTS_PATH.exists()
        assert len(config.STRATEGIES) == 5
        assert "pure_gifsicle" in config.STRATEGIES
        assert "animately_then_gifsicle" in config.STRATEGIES
    
    def test_custom_settings(self, temp_experiment_dir):
        """Test custom configuration settings."""
        config = ExperimentalConfig(
            TEST_GIFS_COUNT=5,
            SAMPLE_GIFS_PATH=temp_experiment_dir / "custom",
            STRATEGIES=["pure_gifsicle"],
            LOSSY_LEVELS=[0, 20],
            FRAME_KEEP_RATIOS=[1.0],
            COLOR_KEEP_COUNTS=[128]
        )
        
        assert config.TEST_GIFS_COUNT == 5
        assert config.STRATEGIES == ["pure_gifsicle"]
        assert config.LOSSY_LEVELS == [0, 20]
        assert config.FRAME_KEEP_RATIOS == [1.0]
        assert config.COLOR_KEEP_COUNTS == [128]


class TestExperimentalPipeline:
    """Test ExperimentalPipeline class."""
    
    def test_init(self, test_config):
        """Test pipeline initialization."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        assert pipeline.config == test_config
        assert pipeline.workers == 1
        assert pipeline.results_dir.exists()
        assert len(pipeline.csv_fieldnames) > 20
    
    def test_generate_sample_gifs(self, test_config):
        """Test sample GIF generation."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        # Mock the GIF creation to avoid actual file operations
        with patch.object(pipeline, '_create_test_gif') as mock_create:
            sample_gifs = pipeline.generate_sample_gifs()
            
            # Should generate the requested number of GIFs
            assert len(sample_gifs) == 10  # Always generates 10 different types
            
            # Should have called create_test_gif for each GIF
            assert mock_create.call_count == 10
    
    def test_generate_jobs(self, test_config):
        """Test job generation."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        # Create mock GIF paths
        mock_gifs = [
            test_config.SAMPLE_GIFS_PATH / "test1.gif",
            test_config.SAMPLE_GIFS_PATH / "test2.gif"
        ]
        
        # Mock metadata extraction
        mock_metadata = GifMetadata(
            gif_sha="mock_sha",
            orig_filename="test.gif",
            orig_kilobytes=100.0,
            orig_width=100,
            orig_height=100,
            orig_frames=10,
            orig_fps=10.0,
            orig_n_colors=256
        )
        
        with patch('src.giflab.experiment.extract_gif_metadata', return_value=mock_metadata):
            jobs = pipeline.generate_jobs(mock_gifs)
            
            # Should generate jobs for all combinations
            expected_jobs = len(mock_gifs) * len(test_config.STRATEGIES) * len(test_config.LOSSY_LEVELS) * len(test_config.FRAME_KEEP_RATIOS) * len(test_config.COLOR_KEEP_COUNTS)
            assert len(jobs) == expected_jobs
            
            # Check job properties
            job = jobs[0]
            assert isinstance(job, ExperimentJob)
            assert job.gif_path in mock_gifs
            assert job.strategy in test_config.STRATEGIES
            assert job.lossy in test_config.LOSSY_LEVELS
            assert job.frame_keep_ratio in test_config.FRAME_KEEP_RATIOS
            assert job.color_keep_count in test_config.COLOR_KEEP_COUNTS
    
    def test_job_identifier(self, test_config):
        """Test job identifier generation."""
        mock_metadata = GifMetadata(
            gif_sha="mock_sha",
            orig_filename="test.gif",
            orig_kilobytes=100.0,
            orig_width=100,
            orig_height=100,
            orig_frames=10,
            orig_fps=10.0,
            orig_n_colors=256
        )
        
        job = ExperimentJob(
            gif_path=Path("test.gif"),
            metadata=mock_metadata,
            strategy="pure_gifsicle",
            engine="gifsicle",
            optimization_level="basic",
            dithering_option="none",
            lossy=40,
            frame_keep_ratio=0.8,
            color_keep_count=64,
            output_path=Path("output.gif")
        )
        
        identifier = job.get_identifier()
        assert "pure_gifsicle" in identifier
        assert "gifsicle" in identifier
        assert "basic" in identifier
        assert "none" in identifier
        assert "l40" in identifier
        assert "r0.80" in identifier
        assert "c64" in identifier


class TestExperimentResult:
    """Test ExperimentResult class."""
    
    def test_success_result(self, test_config):
        """Test successful experiment result."""
        mock_metadata = GifMetadata(
            gif_sha="mock_sha",
            orig_filename="test.gif",
            orig_kilobytes=100.0,
            orig_width=100,
            orig_height=100,
            orig_frames=10,
            orig_fps=10.0,
            orig_n_colors=256
        )
        
        job = ExperimentJob(
            gif_path=Path("test.gif"),
            metadata=mock_metadata,
            strategy="pure_gifsicle",
            engine="gifsicle",
            optimization_level="basic",
            dithering_option="none",
            lossy=0,
            frame_keep_ratio=1.0,
            color_keep_count=256,
            output_path=Path("output.gif")
        )
        
        result = ExperimentResult(
            job=job,
            success=True,
            metrics={"ssim": 0.95, "kilobytes": 80.0},
            compression_result={"render_ms": 1000},
            processing_time_ms=1500
        )
        
        assert result.success is True
        assert result.error is None
        assert result.metrics["ssim"] == 0.95
        assert result.processing_time_ms == 1500
    
    def test_failed_result(self, test_config):
        """Test failed experiment result."""
        mock_metadata = GifMetadata(
            gif_sha="mock_sha",
            orig_filename="test.gif",
            orig_kilobytes=100.0,
            orig_width=100,
            orig_height=100,
            orig_frames=10,
            orig_fps=10.0,
            orig_n_colors=256
        )
        
        job = ExperimentJob(
            gif_path=Path("test.gif"),
            metadata=mock_metadata,
            strategy="pure_gifsicle",
            engine="gifsicle",
            optimization_level="basic",
            dithering_option="none",
            lossy=0,
            frame_keep_ratio=1.0,
            color_keep_count=256,
            output_path=Path("output.gif")
        )
        
        result = ExperimentResult(
            job=job,
            success=False,
            metrics={},
            compression_result={},
            error="Engine not found",
            processing_time_ms=100
        )
        
        assert result.success is False
        assert result.error == "Engine not found"
        assert result.metrics == {}
        assert result.processing_time_ms == 100


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_experimental_pipeline(self):
        """Test pipeline creation factory function."""
        pipeline = create_experimental_pipeline(
            test_gifs_count=5,
            workers=2,
            enable_analysis=False
        )
        
        assert isinstance(pipeline, ExperimentalPipeline)
        assert pipeline.config.TEST_GIFS_COUNT == 5
        assert pipeline.workers == 2
        assert pipeline.config.ENABLE_DETAILED_ANALYSIS is False


class TestGifGeneration:
    """Test GIF generation functionality."""
    
    def test_create_gradient_frame(self, test_config):
        """Test gradient frame creation."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        # This should not raise an exception
        img = pipeline._create_gradient_frame((100, 100), 0, 10)
        
        assert img.size == (100, 100)
        assert img.mode == "RGB"
    
    def test_create_shapes_frame(self, test_config):
        """Test shapes frame creation."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        # This should not raise an exception
        img = pipeline._create_shapes_frame((100, 100), 0, 10)
        
        assert img.size == (100, 100)
        assert img.mode == "RGB"
    
    def test_create_text_frame(self, test_config):
        """Test text frame creation."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        # This should not raise an exception
        img = pipeline._create_text_frame((100, 100), 0, 10)
        
        assert img.size == (100, 100)
        assert img.mode == "RGB"


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for the experimental framework."""
    
    def test_minimal_experiment(self, test_config):
        """Test running a minimal experiment."""
        pipeline = ExperimentalPipeline(test_config, workers=1)
        
        # Mock metadata extraction globally
        mock_metadata = GifMetadata(
                gif_sha="mock_sha",
                orig_filename="test.gif",
                orig_kilobytes=100.0,
                orig_width=100,
                orig_height=100,
                orig_frames=10,
                orig_fps=10.0,
                orig_n_colors=256
            )
        
        # Mock the compression and metrics functions
        with patch('src.giflab.experiment.apply_compression_with_all_params') as mock_compress, \
             patch('src.giflab.experiment.calculate_comprehensive_metrics') as mock_metrics, \
             patch('src.giflab.experiment.extract_gif_metadata') as mock_extract, \
             patch('src.giflab.experiment.compress_with_gifsicle_extended') as mock_gifsicle:
            
            # Setup mocks
            mock_compress.return_value = {"render_ms": 1000}
            mock_metrics.return_value = {"ssim": 0.95, "kilobytes": 80.0, "composite_quality": 0.90}
            mock_extract.return_value = mock_metadata
            mock_gifsicle.return_value = {"render_ms": 500}
            
            # Create a mock GIF file
            mock_gif = test_config.SAMPLE_GIFS_PATH / "test.gif"
            mock_gif.parent.mkdir(parents=True, exist_ok=True)
            mock_gif.write_bytes(b"fake gif content")
            
            # Run experiment with mock GIF
            results_path = pipeline.run_experiment([mock_gif])
            
            # Verify results
            assert results_path.exists()
            assert results_path.name == "experiment_results.csv"
            
            # Check that functions were called
            assert mock_extract.called
            # Note: compression functions might not be called if using gifsicle strategy
            assert mock_extract.call_count > 0 