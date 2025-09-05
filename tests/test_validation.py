"""Tests for input validation functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from giflab.input_validation import (
    ValidationError,
    sanitize_filename,
    validate_config_paths,
    validate_file_extension,
    validate_output_path,
    validate_path_security,
    validate_raw_dir,
    validate_worker_count,
)


class TestValidateRawDir:
    """Tests for validate_raw_dir function."""

    @pytest.mark.fast
    def test_validate_raw_dir_success(self, tmp_path):
        """Test successful RAW_DIR validation."""
        # Create a test directory with a GIF file
        test_dir = tmp_path / "test_raw"
        test_dir.mkdir()
        (test_dir / "test.gif").write_text("fake gif content")

        result = validate_raw_dir(test_dir)
        assert result == test_dir
        assert result.exists()
        assert result.is_dir()

    @pytest.mark.fast
    def test_validate_raw_dir_no_gifs_required(self, tmp_path):
        """Test RAW_DIR validation without requiring GIF files."""
        test_dir = tmp_path / "empty_dir"
        test_dir.mkdir()

        result = validate_raw_dir(test_dir, require_gifs=False)
        assert result == test_dir

    @pytest.mark.fast
    def test_validate_raw_dir_empty_path(self):
        """Test validation with empty path."""
        with pytest.raises(ValidationError, match="RAW_DIR cannot be empty"):
            validate_raw_dir("")

    @pytest.mark.fast
    def test_validate_raw_dir_nonexistent(self, tmp_path):
        """Test validation with non-existent directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(ValidationError, match="RAW_DIR does not exist"):
            validate_raw_dir(nonexistent)

    @pytest.mark.fast
    def test_validate_raw_dir_not_directory(self, tmp_path):
        """Test validation with file instead of directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("not a directory")

        with pytest.raises(ValidationError, match="RAW_DIR is not a directory"):
            validate_raw_dir(test_file)

    @pytest.mark.fast
    def test_validate_raw_dir_no_gifs(self, tmp_path):
        """Test validation with directory containing no GIF files."""
        test_dir = tmp_path / "no_gifs"
        test_dir.mkdir()
        (test_dir / "test.txt").write_text("not a gif")

        with pytest.raises(ValidationError, match="RAW_DIR contains no GIF files"):
            validate_raw_dir(test_dir, require_gifs=True)

    @pytest.mark.fast
    def test_validate_raw_dir_unreadable(self, tmp_path):
        """Test validation with unreadable directory."""
        test_dir = tmp_path / "unreadable"
        test_dir.mkdir()

        # Mock os.access to return False for read permission
        with patch("os.access", return_value=False):
            with pytest.raises(ValidationError, match="RAW_DIR is not readable"):
                validate_raw_dir(test_dir)


class TestValidatePathSecurity:
    """Tests for validate_path_security function."""

    @pytest.mark.fast
    def test_validate_path_security_success(self, tmp_path):
        """Test successful path security validation."""
        test_path = tmp_path / "safe_path"
        result = validate_path_security(test_path)
        assert result == test_path

    @pytest.mark.fast
    def test_validate_path_security_empty(self):
        """Test validation with empty path."""
        with pytest.raises(ValidationError, match="Path cannot be empty"):
            validate_path_security("")

    @pytest.mark.fast
    def test_validate_path_security_dangerous_chars(self):
        """Test validation with dangerous characters."""
        dangerous_paths = [
            "/path/with;semicolon",
            "/path/with&ampersand",
            "/path/with|pipe",
            "/path/with`backtick",
            "/path/with$dollar",
            "/path/with$(command)",
            "/path/with\nnewline",
            "/path/with\rcarriage",
        ]

        for dangerous_path in dangerous_paths:
            with pytest.raises(
                ValidationError, match="potentially dangerous characters"
            ):
                validate_path_security(dangerous_path)

    @pytest.mark.fast
    def test_validate_path_security_traversal(self):
        """Test validation with path traversal attempts."""
        traversal_paths = [
            "../../../etc/passwd",
            "safe/../../../etc/passwd",
            "/path/../../../etc/passwd",
        ]

        for traversal_path in traversal_paths:
            with pytest.raises(ValidationError, match="directory traversal"):
                validate_path_security(traversal_path)

    @pytest.mark.fast
    def test_validate_path_security_null_bytes(self):
        """Test validation with null bytes."""
        # Test with null bytes in the string directly
        with pytest.raises(ValidationError, match="null bytes"):
            validate_path_security("path/with\x00null")

    @pytest.mark.fast
    def test_validate_path_security_too_long(self):
        """Test validation with excessively long path."""
        long_path = "/path/" + "a" * 5000
        with pytest.raises(ValidationError, match="Path too long"):
            validate_path_security(long_path)


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    @pytest.mark.fast
    def test_validate_output_path_success(self, tmp_path):
        """Test successful output path validation."""
        output_path = tmp_path / "output.csv"
        result = validate_output_path(output_path)
        assert result == output_path

    @pytest.mark.fast
    def test_validate_output_path_create_parent(self, tmp_path):
        """Test output path validation with parent creation."""
        output_path = tmp_path / "new_dir" / "output.csv"
        result = validate_output_path(output_path, create_parent=True)
        assert result == output_path
        assert output_path.parent.exists()

    @pytest.mark.fast
    def test_validate_output_path_no_create_parent(self, tmp_path):
        """Test output path validation without parent creation."""
        output_path = tmp_path / "nonexistent" / "output.csv"
        with pytest.raises(ValidationError, match="Parent directory does not exist"):
            validate_output_path(output_path, create_parent=False)

    @pytest.mark.fast
    def test_validate_output_path_unwritable_parent(self, tmp_path):
        """Test output path validation with unwritable parent."""
        output_path = tmp_path / "output.csv"

        # Mock os.access to return False for write permission
        with patch("os.access", return_value=False):
            with pytest.raises(ValidationError, match="not writable"):
                validate_output_path(output_path)

    @pytest.mark.fast
    def test_validate_output_path_existing_unwritable(self, tmp_path):
        """Test output path validation with existing unwritable file."""
        output_path = tmp_path / "output.csv"
        output_path.write_text("existing content")

        # Mock os.access to return True for parent but False for file
        def mock_access(path, mode):
            if str(path).endswith("output.csv"):
                return False  # File is not writable
            return True  # Parent is writable

        with patch("os.access", side_effect=mock_access):
            with pytest.raises(ValidationError, match="Output file is not writable"):
                validate_output_path(output_path)


class TestValidateWorkerCount:
    """Tests for validate_worker_count function."""

    @pytest.mark.fast
    def test_validate_worker_count_success(self):
        """Test successful worker count validation."""
        assert validate_worker_count(4) == 4
        assert validate_worker_count(0) == 0

    def test_validate_worker_count_not_integer(self):
        """Test validation with non-integer input."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_worker_count("4")

    def test_validate_worker_count_negative(self):
        """Test validation with negative worker count."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            validate_worker_count(-1)

    @patch("multiprocessing.cpu_count", return_value=4)
    def test_validate_worker_count_too_high(self, mock_cpu_count):
        """Test validation with excessively high worker count."""
        with pytest.raises(ValidationError, match="Worker count too high"):
            validate_worker_count(20)  # 4 CPUs * 4 = 16 max, so 20 is too high


class TestValidateFileExtension:
    """Tests for validate_file_extension function."""

    def test_validate_file_extension_success(self):
        """Test successful file extension validation."""
        path = Path("test.gif")
        result = validate_file_extension(path, [".gif", ".GIF"])
        assert result == path

    def test_validate_file_extension_without_dot(self):
        """Test validation with extensions without dots."""
        path = Path("test.gif")
        result = validate_file_extension(path, ["gif", "png"])
        assert result == path

    def test_validate_file_extension_case_insensitive(self):
        """Test case-insensitive extension validation."""
        path = Path("test.GIF")
        result = validate_file_extension(path, [".gif"])
        assert result == path

    def test_validate_file_extension_invalid(self):
        """Test validation with invalid extension."""
        path = Path("test.txt")
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_file_extension(path, [".gif", ".png"])


class TestValidateConfigPaths:
    """Tests for validate_config_paths function."""

    def test_validate_config_paths_success(self, tmp_path):
        """Test successful config paths validation."""
        config = {
            "RAW_DIR": tmp_path / "raw",
            "OUTPUT_DIR": tmp_path / "output",
            "SOME_PATH": tmp_path / "path",
            "OTHER_VALUE": "not_a_path",
        }

        result = validate_config_paths(config)
        assert "RAW_DIR" in result
        assert "OUTPUT_DIR" in result
        assert "SOME_PATH" in result
        assert "OTHER_VALUE" not in result

    def test_validate_config_paths_invalid(self):
        """Test config paths validation with invalid path."""
        config = {
            "RAW_DIR": "/path/with;semicolon",
        }

        with pytest.raises(ValidationError, match="Invalid RAW_DIR"):
            validate_config_paths(config)

    def test_validate_config_paths_none_values(self):
        """Test config paths validation with None values."""
        config = {
            "RAW_DIR": None,
            "OUTPUT_DIR": Path("valid/path"),
        }

        result = validate_config_paths(config)
        assert "RAW_DIR" not in result
        assert "OUTPUT_DIR" in result


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_filename_success(self):
        """Test successful filename sanitization."""
        result = sanitize_filename("normal_filename.txt")
        assert result == "normal_filename.txt"

    def test_sanitize_filename_empty(self):
        """Test sanitization with empty filename."""
        result = sanitize_filename("")
        assert result == "unnamed"

    def test_sanitize_filename_invalid_chars(self):
        """Test sanitization with invalid characters."""
        result = sanitize_filename('file<>:"|?*name.txt')
        assert result == "file_______name.txt"

    def test_sanitize_filename_custom_replacement(self):
        """Test sanitization with custom replacement character."""
        result = sanitize_filename("file<>name.txt", replacement="-")
        assert result == "file--name.txt"

    def test_sanitize_filename_control_chars(self):
        """Test sanitization with control characters."""
        result = sanitize_filename("file\x00\x1f\x7fname.txt")
        assert result == "file___name.txt"

    def test_sanitize_filename_dots_spaces(self):
        """Test sanitization with leading/trailing dots and spaces."""
        result = sanitize_filename("  ..filename.txt..  ")
        assert result == "filename.txt"

    def test_sanitize_filename_empty_after_sanitization(self):
        """Test sanitization that results in empty string."""
        result = sanitize_filename("...")
        assert result == "unnamed"

    def test_sanitize_filename_too_long(self):
        """Test sanitization with excessively long filename."""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".txt")


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from ValueError."""
        error = ValidationError("test message")
        assert isinstance(error, ValueError)
        assert str(error) == "test message"

    def test_validation_error_with_cause(self):
        """Test ValidationError with cause."""
        try:
            raise OSError("original error")
        except OSError as e:
            try:
                raise ValidationError("validation failed") from e
            except ValidationError as error:
                assert error.__cause__ is e


class TestIntegration:
    """Integration tests for validation functions."""

    def test_cli_validation_integration(self, tmp_path):
        """Test that CLI validation works end-to-end."""
        # Create a valid RAW_DIR
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "test.gif").write_text("fake gif")

        # This should not raise any exceptions
        validated_dir = validate_raw_dir(raw_dir)
        assert validated_dir == raw_dir

        # Test worker validation
        validated_workers = validate_worker_count(2)
        assert validated_workers == 2

    def test_config_validation_integration(self, tmp_path):
        """Test that config validation works with PathConfig."""
        from giflab.config import PathConfig

        # Create a config with valid paths
        config = PathConfig(
            RAW_DIR=tmp_path / "raw",
            CSV_DIR=tmp_path / "csv",
        )

        # This should not raise any exceptions during __post_init__
        assert config.RAW_DIR == tmp_path / "raw"
        assert config.CSV_DIR == tmp_path / "csv"
