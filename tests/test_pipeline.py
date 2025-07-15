"""Tests for compression pipeline functionality."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.giflab.config import CompressionConfig, PathConfig
from src.giflab.meta import GifMetadata
from src.giflab.pipeline import (
    CompressionJob,
    CompressionPipeline,
    CompressionResult,
    create_pipeline,
    execute_single_job,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    dirs = {
        "raw": temp_dir / "raw",
        "renders": temp_dir / "renders",
        "csv": temp_dir / "csv",
        "bad_gifs": temp_dir / "bad_gifs",
        "logs": temp_dir / "logs"
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    yield dirs

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dirs):
    """Create test configuration."""
    compression_config = CompressionConfig(
        FRAME_KEEP_RATIOS=[1.0, 0.8],
        COLOR_KEEP_COUNTS=[256, 64],
        LOSSY_LEVELS=[0, 40],
        ENGINES=["gifsicle"]
    )

    path_config = PathConfig(
        RAW_DIR=temp_dirs["raw"],
        RENDERS_DIR=temp_dirs["renders"],
        CSV_DIR=temp_dirs["csv"],
        BAD_GIFS_DIR=temp_dirs["bad_gifs"],
        LOGS_DIR=temp_dirs["logs"]
    )

    return compression_config, path_config


@pytest.fixture
def sample_gif_metadata():
    """Create sample GIF metadata for testing."""
    return GifMetadata(
        gif_sha="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abc",
        orig_filename="test.gif",
        orig_kilobytes=100.5,
        orig_width=480,
        orig_height=270,
        orig_frames=24,
        orig_fps=24.0,
        orig_n_colors=128,
        entropy=4.2
    )


@pytest.fixture
def sample_job(temp_dirs, sample_gif_metadata):
    """Create a sample compression job."""
    gif_path = temp_dirs["raw"] / "test.gif"
    gif_path.touch()  # Create empty test file

    # Create path that matches new per-GIF folder structure
    gif_folder = temp_dirs["renders"] / "test_abc123def456"
    output_path = gif_folder / "gifsicle_l0_r1.00_c256.gif"

    return CompressionJob(
        gif_path=gif_path,
        metadata=sample_gif_metadata,
        engine="gifsicle",
        lossy=0,
        frame_keep_ratio=1.0,
        color_keep_count=256,
        output_path=output_path
    )


class TestCompressionPipeline:
    """Test cases for CompressionPipeline class."""

    def test_init(self, test_config):
        """Test pipeline initialization."""
        compression_config, path_config = test_config

        pipeline = CompressionPipeline(
            compression_config=compression_config,
            path_config=path_config,
            workers=2,
            resume=True
        )

        assert pipeline.compression_config == compression_config
        assert pipeline.path_config == path_config
        assert pipeline.workers == 2
        assert pipeline.resume is True
        assert pipeline._shutdown_requested is False

    def test_discover_gifs(self, test_config, temp_dirs):
        """Test GIF file discovery."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Create test GIF files
        (temp_dirs["raw"] / "test1.gif").touch()
        (temp_dirs["raw"] / "test2.GIF").touch()
        (temp_dirs["raw"] / "test.png").touch()  # Should be ignored

        gifs = pipeline.discover_gifs(temp_dirs["raw"])

        assert len(gifs) == 2
        assert any(gif.name == "test1.gif" for gif in gifs)
        assert any(gif.name == "test2.GIF" for gif in gifs)

    @patch('src.giflab.pipeline.extract_gif_metadata')
    def test_generate_jobs(self, mock_extract_metadata, test_config, temp_dirs, sample_gif_metadata):
        """Test job generation."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Setup mock
        mock_extract_metadata.return_value = sample_gif_metadata

        # Create test GIF file
        gif_path = temp_dirs["raw"] / "test.gif"
        gif_path.touch()

        jobs = pipeline.generate_jobs([gif_path])

        # Should create 8 jobs (2 frame ratios × 2 color counts × 2 lossy levels × 1 engine)
        assert len(jobs) == 8

        # Check first job
        first_job = jobs[0]
        assert first_job.gif_path == gif_path
        assert first_job.metadata == sample_gif_metadata
        assert first_job.engine in compression_config.ENGINES
        assert first_job.lossy in compression_config.LOSSY_LEVELS
        assert first_job.frame_keep_ratio in compression_config.FRAME_KEEP_RATIOS
        assert first_job.color_keep_count in compression_config.COLOR_KEEP_COUNTS

        # Check that output path uses per-GIF folder structure
        expected_folder = f"test_{sample_gif_metadata.gif_sha}"
        assert expected_folder in str(first_job.output_path)
        assert first_job.output_path.name.startswith("gifsicle_l")
        assert first_job.output_path.name.endswith(".gif")

    def test_create_gif_folder_name(self, test_config):
        """Test GIF folder name creation."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Test normal filename
        folder_name = pipeline._create_gif_folder_name("test.gif", "abcdef123456789")
        assert folder_name == "test_abcdef123456789"

        # Test filename with special characters
        folder_name = pipeline._create_gif_folder_name("weird name (2)!@#.gif", "xyz789abc123def")
        assert folder_name == "weird_name__2_____xyz789abc123def"

        # Test very long filename
        long_name = "a" * 250 + ".gif"
        folder_name = pipeline._create_gif_folder_name(long_name, "123456789abc")
        assert len(folder_name) <= 264  # 200 + 64 + 1 underscore (full SHA)
        assert folder_name.endswith("_123456789abc")

    @patch('src.giflab.pipeline.move_bad_gif')
    @patch('src.giflab.pipeline.extract_gif_metadata')
    def test_generate_jobs_with_bad_gif(self, mock_extract_metadata, mock_move_bad_gif,
                                       test_config, temp_dirs):
        """Test job generation with a bad GIF file."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Setup mock to raise exception
        mock_extract_metadata.side_effect = Exception("Bad GIF")

        # Create test GIF file
        gif_path = temp_dirs["raw"] / "bad.gif"
        gif_path.touch()

        jobs = pipeline.generate_jobs([gif_path])

        assert len(jobs) == 0
        mock_move_bad_gif.assert_called_once()

    def test_load_existing_csv_records(self, test_config, temp_dirs):
        """Test loading existing CSV records."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Create test CSV file with valid 64-character SHA hashes
        csv_path = temp_dirs["csv"] / "test.csv"
        sha1 = "abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abc"
        sha2 = "def4567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        csv_content = f"""gif_sha,orig_filename,engine,lossy,frame_keep_ratio,color_keep_count,kilobytes,ssim,render_ms,orig_kilobytes,orig_width,orig_height,orig_frames,orig_fps,orig_n_colors,entropy,timestamp
{sha1},test1.gif,gifsicle,0,1.0,256,50.5,0.95,1200,100.0,480,270,24,24.0,128,4.2,2024-01-01T10:00:00
{sha2},test2.gif,gifsicle,40,0.8,64,25.5,0.85,800,80.0,320,240,16,12.0,64,3.8,2024-01-01T10:00:00"""

        csv_path.write_text(csv_content)

        records = pipeline._load_existing_csv_records(csv_path)

        assert len(records) == 2
        assert (sha1, "gifsicle", 0, 1.0, 256) in records
        assert (sha2, "gifsicle", 40, 0.8, 64) in records

    def test_filter_existing_jobs(self, test_config, temp_dirs, sample_job):
        """Test filtering of existing jobs."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)

        # Create CSV with existing record
        csv_path = temp_dirs["csv"] / "test.csv"
        csv_content = f"""gif_sha,orig_filename,engine,lossy,frame_keep_ratio,color_keep_count,kilobytes,ssim,render_ms,orig_kilobytes,orig_width,orig_height,orig_frames,orig_fps,orig_n_colors,entropy,timestamp
{sample_job.metadata.gif_sha},{sample_job.metadata.orig_filename},{sample_job.engine},{sample_job.lossy},{sample_job.frame_keep_ratio},{sample_job.color_keep_count},50.5,0.95,1200,100.0,480,270,24,24.0,128,4.2,2024-01-01T10:00:00"""

        csv_path.write_text(csv_content)

        # Create output file in per-GIF folder to simulate completion
        sample_job.output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_job.output_path.touch()

        filtered_jobs = pipeline.filter_existing_jobs([sample_job], csv_path)

        # Job should be filtered out (already completed)
        assert len(filtered_jobs) == 0

    @patch('src.giflab.pipeline.calculate_comprehensive_metrics')
    @patch('src.giflab.pipeline.apply_compression_with_all_params')
    def test_execute_job(self, mock_compress, mock_metrics, test_config, sample_job):
        """Test job execution."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Setup mocks
        mock_compress.return_value = {"render_ms": 1200}
        mock_metrics.return_value = {
            "kilobytes": 50.5,
            "ssim": 0.95,
            "render_ms": 1200
        }

        result = pipeline.execute_job(sample_job)

        assert isinstance(result, CompressionResult)
        assert result.gif_sha == sample_job.metadata.gif_sha
        assert result.engine == sample_job.engine
        assert result.kilobytes == 50.5
        assert result.ssim == 0.95
        assert result.render_ms == 1200

    @patch('src.giflab.pipeline.calculate_comprehensive_metrics')
    @patch('src.giflab.pipeline.apply_compression_with_all_params')
    def test_execute_job_failure(self, mock_compress, mock_metrics, test_config, sample_job):
        """Test job execution failure handling."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Setup mock to fail
        mock_compress.side_effect = Exception("Compression failed")

        with pytest.raises(RuntimeError, match="Job execution failed"):
            pipeline.execute_job(sample_job)

    @patch('src.giflab.pipeline.ProcessPoolExecutor')
    @patch('src.giflab.pipeline.extract_gif_metadata')
    def test_run_success(self, mock_extract_metadata, mock_executor_class,
                        test_config, temp_dirs, sample_gif_metadata):
        """Test successful pipeline run."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, workers=1)

        # Setup metadata mock
        mock_extract_metadata.return_value = sample_gif_metadata

        # Create mock result
        mock_result = CompressionResult(
            gif_sha="abc123def456",
            orig_filename="test.gif",
            engine="gifsicle",
            lossy=0,
            frame_keep_ratio=1.0,
            color_keep_count=256,
            kilobytes=50.5,
            ssim=0.95,
            render_ms=1200,
            orig_kilobytes=100.0,
            orig_width=480,
            orig_height=270,
            orig_frames=24,
            orig_fps=24.0,
            orig_n_colors=128,
            entropy=4.2,
            timestamp=datetime.now().isoformat()
        )

        # Mock ProcessPoolExecutor to avoid multiprocessing
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Mock future object
        mock_future = MagicMock()
        mock_future.result.return_value = mock_result
        mock_executor.submit.return_value = mock_future

        # Mock as_completed to return our future
        with patch('src.giflab.pipeline.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future]

            # Create test GIF file
            gif_path = temp_dirs["raw"] / "test.gif"
            gif_path.touch()

            # Run pipeline with limited jobs (1 engine, 1 lossy, 1 ratio, 1 color)
            compression_config.ENGINES = ["gifsicle"]
            compression_config.LOSSY_LEVELS = [0]
            compression_config.FRAME_KEEP_RATIOS = [1.0]
            compression_config.COLOR_KEEP_COUNTS = [256]

            result = pipeline.run(temp_dirs["raw"])

            assert result["status"] == "completed"
            assert result["processed"] == 1
            assert result["failed"] == 0

            # Check CSV was created
            csv_files = list(temp_dirs["csv"].glob("results_*.csv"))
            assert len(csv_files) == 1

    @patch('src.giflab.pipeline.extract_gif_metadata')
    def test_find_original_gif_by_sha(self, mock_extract_metadata, test_config, temp_dirs, sample_gif_metadata):
        """Test finding original GIF by SHA hash."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Create test GIF file
        gif_path = temp_dirs["raw"] / "test.gif"
        gif_path.touch()

        # Setup mock to return our test metadata
        mock_extract_metadata.return_value = sample_gif_metadata

        # Test successful lookup with the correct SHA
        found_path = pipeline.find_original_gif_by_sha(sample_gif_metadata.gif_sha, temp_dirs["raw"])
        assert found_path == gif_path

        # Test with non-existent SHA
        found_path = pipeline.find_original_gif_by_sha("nonexistent", temp_dirs["raw"])
        assert found_path is None

    @patch('src.giflab.pipeline.extract_gif_metadata')
    def test_find_original_gif_by_folder_name(self, mock_extract_metadata, test_config, temp_dirs, sample_gif_metadata):
        """Test finding original GIF by render folder name."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)

        # Create test GIF file
        gif_path = temp_dirs["raw"] / "test.gif"
        gif_path.touch()

        # Setup mock to return our test metadata
        mock_extract_metadata.return_value = sample_gif_metadata

        # Test successful lookup with full SHA folder name
        full_sha = "abc123def456" + "0" * 52  # Pad to 64 chars for realistic SHA
        folder_name = f"test_gif_{full_sha}"

        # Update mock to return the full SHA
        sample_gif_metadata.gif_sha = full_sha
        mock_extract_metadata.return_value = sample_gif_metadata

        found_path = pipeline.find_original_gif_by_folder_name(folder_name, temp_dirs["raw"])
        assert found_path == gif_path

        # Test with invalid folder name (no underscore)
        found_path = pipeline.find_original_gif_by_folder_name("invalid", temp_dirs["raw"])
        assert found_path is None

        # Test with invalid SHA in folder name
        found_path = pipeline.find_original_gif_by_folder_name("test_invalidsha", temp_dirs["raw"])
        assert found_path is None


class TestCreatePipeline:
    """Test pipeline factory function."""

    def test_create_pipeline(self, temp_dirs):
        """Test pipeline creation with factory function."""
        pipeline = create_pipeline(
            raw_dir=temp_dirs["raw"],
            workers=4,
            resume=False
        )

        assert isinstance(pipeline, CompressionPipeline)
        assert pipeline.workers == 4
        assert pipeline.resume is False


class TestExecuteSingleJob:
    """Test standalone job execution function."""

    @patch('src.giflab.pipeline.CompressionPipeline.execute_job')
    def test_execute_single_job(self, mock_execute, sample_job):
        """Test single job execution."""
        mock_result = Mock(spec=CompressionResult)
        mock_execute.return_value = mock_result

        result = execute_single_job(sample_job)

        assert result == mock_result
        mock_execute.assert_called_once_with(sample_job)
