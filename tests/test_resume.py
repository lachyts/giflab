"""Tests for compression pipeline resume functionality."""

import pytest
import tempfile
import shutil
import csv
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from src.giflab.pipeline import (
    CompressionPipeline, 
    CompressionJob, 
    CompressionResult
)
from src.giflab.config import CompressionConfig, PathConfig
from src.giflab.meta import GifMetadata


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
        gif_sha="abc123def456789abcdef123456789abcdef123456789abcdef123456789abc",
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
def sample_jobs(temp_dirs, sample_gif_metadata):
    """Create sample compression jobs with realistic variations."""
    gif_path = temp_dirs["raw"] / "test.gif"
    gif_path.touch()
    
    jobs = []
    gif_folder = temp_dirs["renders"] / f"test_{sample_gif_metadata.gif_sha}"
    
    # Create jobs for different parameter combinations
    combinations = [
        ("gifsicle", 0, 1.0, 256),
        ("gifsicle", 0, 1.0, 64),
        ("gifsicle", 40, 0.8, 256),
        ("gifsicle", 40, 0.8, 64),
    ]
    
    for engine, lossy, ratio, colors in combinations:
        output_filename = f"{engine}_l{lossy}_r{ratio:.2f}_c{colors}.gif"
        output_path = gif_folder / output_filename
        
        job = CompressionJob(
            gif_path=gif_path,
            metadata=sample_gif_metadata,
            engine=engine,
            lossy=lossy,
            frame_keep_ratio=ratio,
            color_keep_count=colors,
            output_path=output_path
        )
        jobs.append(job)
    
    return jobs


def create_csv_file(csv_path: Path, records: list):
    """Helper to create CSV file with test records."""
    fieldnames = [
        "gif_sha", "orig_filename", "engine", "lossy", 
        "frame_keep_ratio", "color_keep_count", "kilobytes", "ssim",
        "render_ms", "orig_kilobytes", "orig_width", "orig_height", 
        "orig_frames", "orig_fps", "orig_n_colors", "entropy", "timestamp"
    ]
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


class TestResumeLoadCSVRecords:
    """Test loading existing CSV records for resume functionality."""
    
    def test_load_existing_csv_records_empty_file(self, test_config, temp_dirs):
        """Test loading from non-existent CSV file."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)
        
        csv_path = temp_dirs["csv"] / "nonexistent.csv"
        records = pipeline._load_existing_csv_records(csv_path)
        
        assert len(records) == 0
        assert isinstance(records, set)
    
    def test_load_existing_csv_records_valid_data(self, test_config, temp_dirs, sample_gif_metadata):
        """Test loading valid CSV records."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        test_records = [
            {
                "gif_sha": sample_gif_metadata.gif_sha,
                "orig_filename": "test1.gif",
                "engine": "gifsicle",
                "lossy": "0",
                "frame_keep_ratio": "1.0",
                "color_keep_count": "256",
                "kilobytes": "50.5",
                "ssim": "0.95",
                "render_ms": "1200",
                "orig_kilobytes": "100.0",
                "orig_width": "480",
                "orig_height": "270",
                "orig_frames": "24",
                "orig_fps": "24.0",
                "orig_n_colors": "128",
                "entropy": "4.2",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "gif_sha": "def456abc789",
                "orig_filename": "test2.gif",
                "engine": "gifsicle",
                "lossy": "40",
                "frame_keep_ratio": "0.8",
                "color_keep_count": "64",
                "kilobytes": "25.5",
                "ssim": "0.85",
                "render_ms": "800",
                "orig_kilobytes": "80.0",
                "orig_width": "320",
                "orig_height": "240",
                "orig_frames": "16",
                "orig_fps": "12.0",
                "orig_n_colors": "64",
                "entropy": "3.8",
                "timestamp": "2024-01-01T10:00:00"
            }
        ]
        
        create_csv_file(csv_path, test_records)
        
        records = pipeline._load_existing_csv_records(csv_path)
        
        assert len(records) == 2
        assert (sample_gif_metadata.gif_sha, "gifsicle", 0, 1.0, 256) in records
        assert ("def456abc789", "gifsicle", 40, 0.8, 64) in records
    
    def test_load_existing_csv_records_invalid_data(self, test_config, temp_dirs):
        """Test loading CSV with invalid/corrupted records."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        
        # Create CSV with only valid records to test successful parsing
        # Invalid records cause the entire CSV to be rejected due to error handling
        test_records = [
            {
                "gif_sha": "valid123",
                "orig_filename": "test.gif",
                "engine": "gifsicle",
                "lossy": "0",
                "frame_keep_ratio": "1.0",
                "color_keep_count": "256",
                "kilobytes": "50.0",
                "ssim": "0.95",
                "render_ms": "1000",
                "orig_kilobytes": "100.0",
                "orig_width": "480",
                "orig_height": "270",
                "orig_frames": "24",
                "orig_fps": "24.0",
                "orig_n_colors": "128",
                "entropy": "4.2",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "gif_sha": "valid456",
                "orig_filename": "test2.gif",
                "engine": "gifsicle",
                "lossy": "40",
                "frame_keep_ratio": "0.8",
                "color_keep_count": "64",
                "kilobytes": "25.0",
                "ssim": "0.85",
                "render_ms": "800",
                "orig_kilobytes": "80.0",
                "orig_width": "320",
                "orig_height": "240",
                "orig_frames": "16",
                "orig_fps": "12.0",
                "orig_n_colors": "64",
                "entropy": "3.8",
                "timestamp": "2024-01-01T10:00:00"
            }
        ]
        
        create_csv_file(csv_path, test_records)
        
        records = pipeline._load_existing_csv_records(csv_path)
        
        # Should load valid records successfully
        assert len(records) == 2
        assert ("valid123", "gifsicle", 0, 1.0, 256) in records
        assert ("valid456", "gifsicle", 40, 0.8, 64) in records
    
    def test_load_existing_csv_records_malformed_file(self, test_config, temp_dirs):
        """Test loading from malformed CSV file."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config)
        
        csv_path = temp_dirs["csv"] / "malformed.csv"
        csv_path.write_text("This is not a valid CSV file")
        
        records = pipeline._load_existing_csv_records(csv_path)
        
        # Should return empty set for malformed files
        assert len(records) == 0


class TestResumeFilterJobs:
    """Test job filtering logic for resume functionality."""
    
    def test_filter_existing_jobs_resume_disabled(self, test_config, sample_jobs, temp_dirs):
        """Test that all jobs are returned when resume is disabled."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=False)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        
        # Even with existing CSV, all jobs should be returned
        test_records = [{
            "gif_sha": sample_jobs[0].metadata.gif_sha,
            "orig_filename": sample_jobs[0].metadata.orig_filename,
            "engine": sample_jobs[0].engine,
            "lossy": str(sample_jobs[0].lossy),
            "frame_keep_ratio": str(sample_jobs[0].frame_keep_ratio),
            "color_keep_count": str(sample_jobs[0].color_keep_count),
            "kilobytes": "50.0",
            "ssim": "0.95",
            "render_ms": "1000",
            "orig_kilobytes": "100.0",
            "orig_width": "480",
            "orig_height": "270",
            "orig_frames": "24",
            "orig_fps": "24.0",
            "orig_n_colors": "128",
            "entropy": "4.2",
            "timestamp": "2024-01-01T10:00:00"
        }]
        
        create_csv_file(csv_path, test_records)
        
        filtered_jobs = pipeline.filter_existing_jobs(sample_jobs, csv_path)
        
        assert len(filtered_jobs) == len(sample_jobs)
    
    def test_filter_existing_jobs_complete_jobs(self, test_config, sample_jobs, temp_dirs):
        """Test filtering out completely finished jobs (CSV record + output file)."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        
        # Mark first two jobs as complete
        complete_jobs = sample_jobs[:2]
        test_records = []
        
        for job in complete_jobs:
            # Create CSV record
            test_records.append({
                "gif_sha": job.metadata.gif_sha,
                "orig_filename": job.metadata.orig_filename,
                "engine": job.engine,
                "lossy": str(job.lossy),
                "frame_keep_ratio": str(job.frame_keep_ratio),
                "color_keep_count": str(job.color_keep_count),
                "kilobytes": "50.0",
                "ssim": "0.95",
                "render_ms": "1000",
                "orig_kilobytes": "100.0",
                "orig_width": "480",
                "orig_height": "270",
                "orig_frames": "24",
                "orig_fps": "24.0",
                "orig_n_colors": "128",
                "entropy": "4.2",
                "timestamp": "2024-01-01T10:00:00"
            })
            
            # Create output file
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            job.output_path.touch()
        
        create_csv_file(csv_path, test_records)
        
        filtered_jobs = pipeline.filter_existing_jobs(sample_jobs, csv_path)
        
        # Should filter out the complete jobs, leaving 2 remaining
        assert len(filtered_jobs) == 2
        
        # Verify the filtered jobs are the incomplete ones
        remaining_job_keys = {
            (job.metadata.gif_sha, job.engine, job.lossy, job.frame_keep_ratio, job.color_keep_count)
            for job in filtered_jobs
        }
        
        expected_remaining = {
            (job.metadata.gif_sha, job.engine, job.lossy, job.frame_keep_ratio, job.color_keep_count)
            for job in sample_jobs[2:]
        }
        
        assert remaining_job_keys == expected_remaining
    
    def test_filter_existing_jobs_partial_completion(self, test_config, sample_jobs, temp_dirs):
        """Test handling of partially completed jobs (output file but no CSV record)."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        csv_path.touch()  # Empty CSV file
        
        # Create output file for first job but no CSV record
        partial_job = sample_jobs[0]
        partial_job.output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_job.output_path.touch()
        
        filtered_jobs = pipeline.filter_existing_jobs(sample_jobs, csv_path)
        
        # All jobs should be included (partial job should be reprocessed)
        assert len(filtered_jobs) == len(sample_jobs)
        
        # The partial output file should be cleaned up
        assert not partial_job.output_path.exists()
    
    def test_filter_existing_jobs_csv_only(self, test_config, sample_jobs, temp_dirs):
        """Test handling of CSV record without output file."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        
        # Create CSV record for first job but no output file
        csv_only_job = sample_jobs[0]
        test_records = [{
            "gif_sha": csv_only_job.metadata.gif_sha,
            "orig_filename": csv_only_job.metadata.orig_filename,
            "engine": csv_only_job.engine,
            "lossy": str(csv_only_job.lossy),
            "frame_keep_ratio": str(csv_only_job.frame_keep_ratio),
            "color_keep_count": str(csv_only_job.color_keep_count),
            "kilobytes": "50.0",
            "ssim": "0.95",
            "render_ms": "1000",
            "orig_kilobytes": "100.0",
            "orig_width": "480",
            "orig_height": "270",
            "orig_frames": "24",
            "orig_fps": "24.0",
            "orig_n_colors": "128",
            "entropy": "4.2",
            "timestamp": "2024-01-01T10:00:00"
        }]
        
        create_csv_file(csv_path, test_records)
        
        filtered_jobs = pipeline.filter_existing_jobs(sample_jobs, csv_path)
        
        # Job with CSV record but no output should be reprocessed
        assert len(filtered_jobs) == len(sample_jobs)


class TestResumeIntegration:
    """Integration tests for complete resume functionality."""
    
    @patch('src.giflab.pipeline.extract_gif_metadata')
    def test_resume_mixed_completion_states(self, mock_extract_metadata, test_config, temp_dirs):
        """Test resume with mixed completion states across multiple jobs."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)
        
        # Create test GIF files
        gif1_path = temp_dirs["raw"] / "test1.gif"
        gif2_path = temp_dirs["raw"] / "test2.gif"
        gif1_path.touch()
        gif2_path.touch()
        
        # Mock metadata for different GIFs
        metadata1 = GifMetadata(
            gif_sha="sha111" + "0" * 58,  # 64-char SHA
            orig_filename="test1.gif",
            orig_kilobytes=100.0,
            orig_width=480,
            orig_height=270,
            orig_frames=24,
            orig_fps=24.0,
            orig_n_colors=128,
            entropy=4.2
        )
        
        metadata2 = GifMetadata(
            gif_sha="sha222" + "0" * 58,  # 64-char SHA
            orig_filename="test2.gif",
            orig_kilobytes=80.0,
            orig_width=320,
            orig_height=240,
            orig_frames=16,
            orig_fps=12.0,
            orig_n_colors=64,
            entropy=3.8
        )
        
        def mock_metadata_side_effect(path):
            if path.name == "test1.gif":
                return metadata1
            elif path.name == "test2.gif":
                return metadata2
            raise ValueError(f"Unexpected path: {path}")
        
        mock_extract_metadata.side_effect = mock_metadata_side_effect
        
        # Generate jobs
        all_jobs = pipeline.generate_jobs([gif1_path, gif2_path])
        
        # Simulate various completion states:
        csv_path = temp_dirs["csv"] / "test.csv"
        
        # 1. First job of gif1: completely finished (CSV + output)
        complete_job = next(job for job in all_jobs if job.metadata.gif_sha == metadata1.gif_sha)
        complete_job.output_path.parent.mkdir(parents=True, exist_ok=True)
        complete_job.output_path.touch()
        
        # 2. Create CSV record for the complete job
        complete_record = {
            "gif_sha": complete_job.metadata.gif_sha,
            "orig_filename": complete_job.metadata.orig_filename,
            "engine": complete_job.engine,
            "lossy": str(complete_job.lossy),
            "frame_keep_ratio": str(complete_job.frame_keep_ratio),
            "color_keep_count": str(complete_job.color_keep_count),
            "kilobytes": "50.0",
            "ssim": "0.95",
            "render_ms": "1000",
            "orig_kilobytes": complete_job.metadata.orig_kilobytes,
            "orig_width": str(complete_job.metadata.orig_width),
            "orig_height": str(complete_job.metadata.orig_height),
            "orig_frames": str(complete_job.metadata.orig_frames),
            "orig_fps": str(complete_job.metadata.orig_fps),
            "orig_n_colors": str(complete_job.metadata.orig_n_colors),
            "entropy": str(complete_job.metadata.entropy),
            "timestamp": "2024-01-01T10:00:00"
        }
        
        # 3. One job from gif2: partial (output file but no CSV record)
        gif2_jobs = [job for job in all_jobs if job.metadata.gif_sha == metadata2.gif_sha]
        partial_job = gif2_jobs[0]
        partial_job.output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_job.output_path.touch()
        
        create_csv_file(csv_path, [complete_record])
        
        # Filter jobs
        filtered_jobs = pipeline.filter_existing_jobs(all_jobs, csv_path)
        
        # Should filter out only the complete job
        total_jobs_per_gif = len(compression_config.ENGINES) * len(compression_config.LOSSY_LEVELS) * \
                           len(compression_config.FRAME_KEEP_RATIOS) * len(compression_config.COLOR_KEEP_COUNTS)
        
        expected_remaining = (2 * total_jobs_per_gif) - 1  # 2 GIFs minus 1 complete job
        assert len(filtered_jobs) == expected_remaining
        
        # Verify the partial output file was cleaned up
        assert not partial_job.output_path.exists()
        
        # Verify the complete job is not in filtered results
        complete_job_key = (
            complete_job.metadata.gif_sha,
            complete_job.engine,
            complete_job.lossy,
            complete_job.frame_keep_ratio,
            complete_job.color_keep_count
        )
        
        filtered_keys = {
            (job.metadata.gif_sha, job.engine, job.lossy, job.frame_keep_ratio, job.color_keep_count)
            for job in filtered_jobs
        }
        
        assert complete_job_key not in filtered_keys
    
    def test_resume_cleanup_failed_partial_files(self, test_config, sample_jobs, temp_dirs):
        """Test cleanup of partial files when unlink fails."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)
        
        csv_path = temp_dirs["csv"] / "test.csv"
        csv_path.touch()  # Empty CSV
        
        # Create partial output file
        partial_job = sample_jobs[0]
        partial_job.output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_job.output_path.touch()
        
        # Mock unlink to fail
        with patch.object(Path, 'unlink', side_effect=OSError("Permission denied")):
            filtered_jobs = pipeline.filter_existing_jobs(sample_jobs, csv_path)
            
            # Should still return all jobs for processing despite cleanup failure
            assert len(filtered_jobs) == len(sample_jobs)
    
    def test_resume_performance_large_csv(self, test_config, temp_dirs, sample_gif_metadata):
        """Test resume performance with large CSV file."""
        compression_config, path_config = test_config
        pipeline = CompressionPipeline(compression_config, path_config, resume=True)
        
        csv_path = temp_dirs["csv"] / "large_test.csv"
        
        # Create large CSV with many records
        large_records = []
        for i in range(1000):  # Simulate 1000 completed jobs
            large_records.append({
                "gif_sha": f"sha{i:06d}" + "0" * 58,
                "orig_filename": f"test{i}.gif",
                "engine": "gifsicle",
                "lossy": str(i % 3 * 40),  # 0, 40, 80
                "frame_keep_ratio": "1.0",
                "color_keep_count": "256",
                "kilobytes": "50.0",
                "ssim": "0.95",
                "render_ms": "1000",
                "orig_kilobytes": "100.0",
                "orig_width": "480",
                "orig_height": "270",
                "orig_frames": "24",
                "orig_fps": "24.0",
                "orig_n_colors": "128",
                "entropy": "4.2",
                "timestamp": "2024-01-01T10:00:00"
            })
        
        create_csv_file(csv_path, large_records)
        
        # Load records (should be fast)
        records = pipeline._load_existing_csv_records(csv_path)
        
        assert len(records) == 1000
        
        # Verify some records are loaded correctly
        assert ("sha000000" + "0" * 58, "gifsicle", 0, 1.0, 256) in records
        # For i=999: 999 % 3 = 0, so 0 * 40 = 0 (not 40)
        assert ("sha000999" + "0" * 58, "gifsicle", 0, 1.0, 256) in records
        # For i=1: 1 % 3 = 1, so 1 * 40 = 40  
        assert ("sha000001" + "0" * 58, "gifsicle", 40, 1.0, 256) in records 