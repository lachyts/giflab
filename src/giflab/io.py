"""I/O utilities for atomic writes, CSV operations, and error logging."""

import atexit
import csv
import json
import logging
import os
from shutil import move, copy2
import signal
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    def lock_file(file_handle):
        """Lock file on Windows using msvcrt."""
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            pass  # File locking may not be available in all situations

    def unlock_file(file_handle):
        """Unlock file on Windows using msvcrt."""
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def lock_file(file_handle):
        """Lock file on Unix systems using fcntl."""
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass

    def unlock_file(file_handle):
        """Unlock file on Unix systems using fcntl."""
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration for GifLab.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"giflab_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("giflab")


@contextmanager
def atomic_write(target_path: Path, mode: str = "w"):
    """Context manager for atomic file writes using temporary files.

    Args:
        target_path: Final path where file should be written
        mode: File open mode

    Yields:
        File handle for writing

    Example:
        with atomic_write(Path("data.json")) as f:
            json.dump(data, f)
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in same directory as target
    temp_dir = target_path.parent
    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=temp_dir,
        delete=False,
        suffix=f".tmp_{target_path.name}"
    ) as temp_file:
        try:
            yield temp_file
            temp_file.flush()
            # Atomic move on POSIX systems
            move(temp_file.name, target_path)
        except Exception:
            # Clean up temp file on error
            Path(temp_file.name).unlink(missing_ok=True)
            raise


def append_csv_row(csv_path: Path, row_data: dict[str, Any], fieldnames: list[str]) -> None:
    """Atomically append a row to a CSV file with proper locking to prevent race conditions.

    Args:
        csv_path: Path to CSV file
        row_data: Dictionary of row data to append
        fieldnames: List of CSV column names

    Raises:
        IOError: If file cannot be written
    """
    # Validate against MetricRecordV1 if it appears to be a metrics row. This is
    # done *best-effort* – we only attempt validation when the dictionary
    # contains the key that uniquely identifies a metrics export ("composite_quality").
    #
    # The CSV writer is reused for both compression summaries and full metrics
    # exports. Compression summaries (which lack many metric keys) should skip
    # validation to avoid false errors. Full metric exports *must* pass the
    # schema, raising immediately on detection of malformed records.
    try:
        if "composite_quality" in row_data:
            # Lazy import to avoid circulars and optional dependency load when
            # only basic I/O is required.
            from giflab.schema import validate_metric_record

            validate_metric_record(row_data)  # Will raise ValidationError on failure
    except Exception as e:  # pragma: no cover – propagate clearer message
        raise ValueError(f"Invalid metric record: {e}") from e

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Use file locking to prevent race conditions in multiprocessing
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        try:
            # Acquire exclusive lock (blocks until available)
            lock_file(f)

            # Check if file is empty (needs header) after acquiring lock
            current_pos = f.tell()
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(current_pos)

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header if file is empty
            if file_size == 0:
                writer.writeheader()

            # Write the row
            writer.writerow(row_data)

        finally:
            # Lock is automatically released when file is closed
            unlock_file(f)


def read_csv_as_dicts(csv_path: Path) -> list[dict[str, Any]]:
    """Read CSV file and return as list of dictionaries.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of dictionaries representing CSV rows

    Raises:
        IOError: If file cannot be read
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_json(data: dict[str, Any], json_path: Path) -> None:
    """Atomically save data as JSON file.

    Args:
        data: Data to save as JSON
        json_path: Path where JSON should be saved

    Raises:
        IOError: If file cannot be written
    """
    with atomic_write(json_path) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(json_path: Path) -> dict[str, Any]:
    """Load JSON data from file.

    Args:
        json_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        IOError: If file cannot be read
        json.JSONDecodeError: If JSON is invalid
    """
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def move_bad_gif(gif_path: Path, bad_gifs_dir: Path) -> Path:
    """Move a corrupted/unreadable GIF to the bad_gifs directory.

    Args:
        gif_path: Path to the bad GIF file
        bad_gifs_dir: Directory for bad GIFs

    Returns:
        Path where the bad GIF was moved

    Raises:
        IOError: If file cannot be moved
    """
    bad_gifs_dir.mkdir(parents=True, exist_ok=True)

    # Validate source file exists
    if not gif_path.exists():
        raise OSError(f"Source file does not exist: {gif_path}")

    # Preserve original weird filenames
    dest_path = bad_gifs_dir / gif_path.name

    # Handle name conflicts
    counter = 1
    while dest_path.exists():
        stem = gif_path.stem
        suffix = gif_path.suffix
        dest_path = bad_gifs_dir / f"{stem}_{counter}{suffix}"
        counter += 1

        # Prevent infinite loop with too many conflicts
        if counter > 1000:
            raise OSError(f"Too many filename conflicts when moving {gif_path}")

    try:
        # Use Path.rename() which is more reliable than move()
        gif_path.rename(dest_path)
    except OSError as e:
        # If rename fails (e.g., cross-filesystem), fall back to copy + delete
        try:
            copy2(str(gif_path), str(dest_path))
            gif_path.unlink()
        except Exception as copy_error:
            raise OSError(f"Failed to move {gif_path} to {dest_path}: {copy_error}") from e

    return dest_path


def ensure_directories(*paths: Path) -> None:
    """Ensure that all specified directories exist.

    Args:
        *paths: Directory paths to create
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
