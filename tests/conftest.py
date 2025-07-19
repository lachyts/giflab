"""
Test configuration and shared fixtures for faster test execution.

This file implements the fast test suite recommendations:
- Session-scoped fixtures for expensive operations
- Module-scoped fixtures for moderate setup
- Function-scoped only for test-specific data
"""

import csv
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Disable CLIP loading during tests to prevent memory issues
os.environ['GIFLAB_DISABLE_CLIP'] = '1'

from src.giflab.config import CompressionConfig, PathConfig
from src.giflab.meta import GifMetadata


@pytest.fixture(scope="session")
def session_temp_dirs():
    """Session-scoped temporary directories - reused across all tests."""
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
    
    # Cleanup only at end of session
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session") 
def session_test_config(session_temp_dirs):
    """Session-scoped test configuration - reused across all tests."""
    compression_config = CompressionConfig(
        FRAME_KEEP_RATIOS=[1.0, 0.8],
        COLOR_KEEP_COUNTS=[256, 64], 
        LOSSY_LEVELS=[0, 40],
        ENGINES=["gifsicle"]
    )
    
    path_config = PathConfig(
        RAW_DIR=session_temp_dirs["raw"],
        RENDERS_DIR=session_temp_dirs["renders"],
        CSV_DIR=session_temp_dirs["csv"],
        BAD_GIFS_DIR=session_temp_dirs["bad_gifs"],
        LOGS_DIR=session_temp_dirs["logs"]
    )
    
    return compression_config, path_config


@pytest.fixture(scope="module")
def sample_gif_metadata():
    """Module-scoped sample GIF metadata - reused within each test module."""
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


@pytest.fixture(scope="module")
def sample_csv_data():
    """Module-scoped sample CSV data - reused within each test module."""
    return [
        {
            'gif_sha': 'sha123',
            'orig_filename': 'test1.gif',
            'engine': 'original',
            'lossy': '0',
            'frame_keep_ratio': '1.00',
            'color_keep_count': '256',
            'kilobytes': '100.5',
            'ssim': '1.000',
            'timestamp': '2024-01-01T10:00:00Z'
        },
        {
            'gif_sha': 'sha123',
            'orig_filename': 'test1.gif',
            'engine': 'gifsicle',
            'lossy': '40',
            'frame_keep_ratio': '0.80',
            'color_keep_count': '64',
            'kilobytes': '50.2',
            'ssim': '0.936',
            'timestamp': '2024-01-01T10:01:00Z'
        },
        {
            'gif_sha': 'sha456',
            'orig_filename': 'test2.gif',
            'engine': 'original',
            'lossy': '0',
            'frame_keep_ratio': '1.00',
            'color_keep_count': '256',
            'kilobytes': '200.0',
            'ssim': '1.000',
            'timestamp': '2024-01-01T11:00:00Z'
        }
    ]


@pytest.fixture(scope="module")
def sample_jobs_data(sample_gif_metadata):
    """Module-scoped sample jobs data - reused within each test module."""
    from src.giflab.pipeline import CompressionJob
    
    return [
        CompressionJob(
            gif_path=Path("/fake/path/test1.gif"),
            gif_metadata=sample_gif_metadata,
            output_path=Path("/fake/output/test1_output.gif"),
            frame_keep_ratio=1.0,
            color_keep_count=256,
            lossy_level=0,
            engine="gifsicle"
        ),
        CompressionJob(
            gif_path=Path("/fake/path/test2.gif"),
            gif_metadata=sample_gif_metadata,
            output_path=Path("/fake/output/test2_output.gif"),
            frame_keep_ratio=0.8,
            color_keep_count=64,
            lossy_level=40,
            engine="gifsicle"
        )
    ]


# Legacy compatibility fixtures that delegate to the optimized versions
# These maintain backwards compatibility while using optimized implementations

@pytest.fixture
def temp_dirs(session_temp_dirs):
    """Function-scoped alias to session temp dirs for backwards compatibility."""
    # Clean up any existing files from previous tests for isolation
    for dir_path in session_temp_dirs.values():
        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path, ignore_errors=True)
    return session_temp_dirs


@pytest.fixture  
def test_config(session_test_config):
    """Function-scoped alias to session config for backwards compatibility."""
    return session_test_config


@pytest.fixture
def sample_job(session_temp_dirs, sample_gif_metadata):
    """Create a sample compression job - function scoped since it may be modified."""
    from src.giflab.pipeline import CompressionJob
    
    gif_path = session_temp_dirs["raw"] / "test.gif"
    if not gif_path.exists():
        gif_path.touch()  # Create empty test file only if it doesn't exist
    
    # Create path that matches new per-GIF folder structure
    gif_folder = session_temp_dirs["renders"] / "test_abc123def456"
    gif_folder.mkdir(exist_ok=True)
    output_path = gif_folder / "gifsicle_l0_r1.00_c256.gif"
    
    return CompressionJob(
        gif_path=gif_path,
        gif_metadata=sample_gif_metadata,
        output_path=output_path,
        frame_keep_ratio=1.0,
        color_keep_count=256,
        lossy_level=0,
        engine="gifsicle"
    )


@pytest.fixture
def sample_jobs(sample_jobs_data):
    """Function-scoped alias to module jobs data for backwards compatibility."""
    return sample_jobs_data


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_data):
    """Create a sample CSV file - function scoped since file paths vary."""
    csv_path = tmp_path / "results.csv"
    
    if sample_csv_data:
        fieldnames = sample_csv_data[0].keys()
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_csv_data)
    
    return csv_path


@pytest.fixture
def sample_gifs(tmp_path):
    """Create sample GIF files - function scoped since paths may vary."""
    gif_dir = tmp_path / "gifs"
    gif_dir.mkdir()
    
    # Create dummy GIF files
    gif_files = []
    for i in range(3):
        gif_path = gif_dir / f"test{i+1}.gif"
        gif_path.touch()
        gif_files.append(gif_path)
    
    return gif_files


# Mock fixtures with appropriate scoping
@pytest.fixture
def mock_tagger():
    """Create a mock HybridCompressionTagger - function scoped since mocks may vary."""
    with patch('src.giflab.tag_pipeline.HybridCompressionTagger') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


# Configure pytest for faster execution
def pytest_configure(config):
    """Configure pytest for optimal performance."""
    # Disable capturing for faster execution if not in debug mode
    if not config.getoption("--capture"):
        config.option.capture = "no"


def pytest_collection_modifyitems(config, items):
    """Optimize test collection and execution order."""
    # Sort tests to run faster ones first
    def test_priority(item):
        # Unit tests should run before integration tests
        if "unit" in item.nodeid.lower():
            return 0
        elif "integration" in item.nodeid.lower():
            return 2
        else:
            return 1
    
    items.sort(key=test_priority)