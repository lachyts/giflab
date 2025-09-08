"""Test suite for cleanup utility functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from giflab.core.runner import GifLabRunner


class TestCleanupFailedExperiment:
    """Test the _cleanup_failed_experiment method in GifLabRunner."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal GifLabRunner instance
        self.runner = GifLabRunner()
        self.runner.logger = Mock()

    def test_cleanup_with_no_output_dir(self):
        """Test cleanup when output_dir is None."""
        self.runner.output_dir = None

        # Should return early without any action
        self.runner._cleanup_failed_experiment()

        # Logger should not be called
        self.runner.logger.info.assert_not_called()
        self.runner.logger.warning.assert_not_called()

    @patch("giflab.core.runner.rmtree")
    def test_cleanup_failed_experiment_success(self, mock_rmtree):
        """Test successful cleanup of failed experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_123"
            output_dir.mkdir()

            # Create only temp_synthetic directory (failed experiment)
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()

            self.runner.output_dir = output_dir

            # Mock glob to return no results (no meaningful files)
            with patch.object(Path, "glob", return_value=[]):
                self.runner._cleanup_failed_experiment()

            # Should attempt to remove the directory
            mock_rmtree.assert_called_once_with(output_dir)
            self.runner.logger.info.assert_called_once()

    @patch("giflab.core.runner.rmtree")
    def test_cleanup_with_meaningful_results(self, mock_rmtree):
        """Test that directories with meaningful results are NOT cleaned up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_123"
            output_dir.mkdir()

            # Create temp_synthetic and a CSV file (meaningful result)
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()
            csv_file = output_dir / "results.csv"
            csv_file.write_text("test,data")

            self.runner.output_dir = output_dir

            # Mock glob to return the CSV file
            with patch.object(Path, "glob", return_value=[csv_file]):
                self.runner._cleanup_failed_experiment()

            # Should NOT attempt to remove the directory
            mock_rmtree.assert_not_called()
            self.runner.logger.info.assert_not_called()

    def test_cleanup_handles_symlink_relative_path(self):
        """Test cleanup properly handles relative symlink targets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_dir = base_dir / "experiment_123"
            output_dir.mkdir()

            # Create only temp_synthetic directory
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()

            # Create a relative symlink
            latest_link = base_dir / "latest"
            latest_link.symlink_to("experiment_123")

            self.runner.output_dir = output_dir

            with patch.object(Path, "glob", return_value=[]), patch(
                "giflab.core.runner.rmtree"
            ) as mock_rmtree:
                self.runner._cleanup_failed_experiment()

                # Directory should be removed
                mock_rmtree.assert_called_once_with(output_dir)

                # Symlink should be removed
                assert not latest_link.exists()

    def test_cleanup_handles_symlink_absolute_path(self):
        """Test cleanup properly handles absolute symlink targets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_dir = base_dir / "experiment_123"
            output_dir.mkdir()

            # Create only temp_synthetic directory
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()

            # Create an absolute symlink
            latest_link = base_dir / "latest"
            latest_link.symlink_to(output_dir.absolute())

            self.runner.output_dir = output_dir

            with patch.object(Path, "glob", return_value=[]), patch(
                "giflab.core.runner.rmtree"
            ) as mock_rmtree:
                self.runner._cleanup_failed_experiment()

                # Directory should be removed
                mock_rmtree.assert_called_once_with(output_dir)

                # Symlink should be removed
                assert not latest_link.exists()

    def test_cleanup_ignores_symlink_to_different_directory(self):
        """Test cleanup doesn't remove symlink pointing to different directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_dir = base_dir / "experiment_123"
            output_dir.mkdir()
            other_dir = base_dir / "other_experiment"
            other_dir.mkdir()

            # Create only temp_synthetic directory
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()

            # Create symlink pointing to different directory
            latest_link = base_dir / "latest"
            latest_link.symlink_to("other_experiment")

            self.runner.output_dir = output_dir

            with patch.object(Path, "glob", return_value=[]), patch(
                "giflab.core.runner.rmtree"
            ) as mock_rmtree:
                self.runner._cleanup_failed_experiment()

                # Directory should be removed
                mock_rmtree.assert_called_once_with(output_dir)

                # Symlink should NOT be removed (points to different directory)
                assert latest_link.exists()

    @patch("giflab.core.runner.rmtree")
    def test_cleanup_handles_rmtree_exception(self, mock_rmtree):
        """Test cleanup handles rmtree exceptions gracefully."""
        mock_rmtree.side_effect = OSError("Permission denied")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_123"
            output_dir.mkdir()

            # Create only temp_synthetic directory
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()

            self.runner.output_dir = output_dir

            with patch.object(Path, "glob", return_value=[]):
                # Should not raise exception
                self.runner._cleanup_failed_experiment()

            # Should log warning about cleanup failure
            self.runner.logger.warning.assert_called_once()
            assert "Failed to clean up failed experiment directory" in str(
                self.runner.logger.warning.call_args
            )

    def test_cleanup_empty_directory(self):
        """Test cleanup of completely empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_123"
            output_dir.mkdir()
            # Leave directory empty

            self.runner.output_dir = output_dir

            with patch.object(Path, "glob", return_value=[]), patch(
                "giflab.core.runner.rmtree"
            ) as mock_rmtree:
                self.runner._cleanup_failed_experiment()

                # Empty directory should be cleaned up
                mock_rmtree.assert_called_once_with(output_dir)
                self.runner.logger.info.assert_called_once()

    def test_cleanup_directory_with_multiple_files(self):
        """Test cleanup doesn't occur when directory has multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "experiment_123"
            output_dir.mkdir()

            # Create temp_synthetic and another file
            temp_synthetic = output_dir / "temp_synthetic"
            temp_synthetic.mkdir()
            other_file = output_dir / "other.txt"
            other_file.write_text("content")

            self.runner.output_dir = output_dir

            with patch.object(Path, "glob", return_value=[]), patch(
                "giflab.core.runner.rmtree"
            ) as mock_rmtree:
                self.runner._cleanup_failed_experiment()

                # Should NOT clean up directory with multiple contents
                mock_rmtree.assert_not_called()
