"""Compression pipeline orchestrator with resume functionality."""

import json
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CompressionConfig, PathConfig
from .io import append_csv_row, move_bad_gif, read_csv_as_dicts, setup_logging
from .lossy import LossyEngine, apply_compression_with_all_params
from .meta import GifMetadata, extract_gif_metadata
from .directory_source_detection import detect_source_from_directory
from .metrics import calculate_comprehensive_metrics
from . import __version__ as GIFLAB_VERSION


def _get_git_commit_hash() -> str:
    """Return short git commit hash if repository is available, else 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # pragma: no cover
        pass
    return "unknown"


@dataclass
class CompressionJob:
    """Represents a single compression job (one variant of one GIF)."""

    gif_path: Path
    metadata: GifMetadata
    engine: str
    lossy: int
    frame_keep_ratio: float
    color_keep_count: int
    output_path: Path


@dataclass
class CompressionResult:
    """Result of a compression job execution."""

    # Job identification
    gif_sha: str
    orig_filename: str
    engine: str
    engine_version: str
    lossy: int
    frame_keep_ratio: float
    color_keep_count: int

    # Results
    kilobytes: float
    ssim: float
    render_ms: int

    # Original metadata
    orig_kilobytes: float
    orig_width: int
    orig_height: int
    orig_frames: int
    orig_fps: float
    orig_n_colors: int
    entropy: float | None
    
    # Source tracking
    source_platform: str
    source_metadata: str | None  # JSON string for CSV compatibility

    # Timestamp
    timestamp: str

    # Version metadata (auto-populated)
    giflab_version: str = field(init=False, default=GIFLAB_VERSION)
    code_commit: str = field(init=False, default_factory=_get_git_commit_hash)
    dataset_version: str = field(init=False, default_factory=lambda: datetime.now().strftime("%Y%m%d"))


class CompressionPipeline:
    """Main pipeline for orchestrating GIF compression with resume capability."""

    def __init__(
        self,
        compression_config: CompressionConfig,
        path_config: PathConfig,
        workers: int = 0,
        resume: bool = True,
        detect_source_from_directory: bool = True
    ):
        """Initialize the compression pipeline.

        Args:
            compression_config: Configuration for compression variants
            path_config: Configuration for file paths
            workers: Number of worker processes (0 = CPU count)
            resume: Whether to resume from existing progress
            detect_source_from_directory: Whether to detect source from directory structure
        """
        self.compression_config = compression_config
        self.path_config = path_config
        self.workers = workers if workers > 0 else multiprocessing.cpu_count()
        self.resume = resume
        self.detect_source_from_directory = detect_source_from_directory
        self.logger = setup_logging(path_config.LOGS_DIR)

        # CSV fieldnames based on project scope
        self.csv_fieldnames = [
            "gif_sha", "orig_filename", "engine", "engine_version", "lossy",
            "frame_keep_ratio", "color_keep_count", "kilobytes", "ssim",
            "render_ms", "orig_kilobytes", "orig_width", "orig_height",
            "orig_frames", "orig_fps", "orig_n_colors", "entropy", 
            "source_platform", "source_metadata", "timestamp",
            "giflab_version", "code_commit", "dataset_version"
        ]

        # Setup signal handling for graceful shutdown
        self._setup_signal_handlers()
        self._shutdown_requested = False

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, requesting graceful shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def discover_gifs(self, raw_dir: Path) -> list[Path]:
        """Discover all GIF files in the raw directory.

        Args:
            raw_dir: Directory containing raw GIF files

        Returns:
            List of GIF file paths
        """
        gif_patterns = ["*.gif", "*.GIF"]
        gif_files = []

        for pattern in gif_patterns:
            gif_files.extend(raw_dir.glob(pattern))

        # Remove duplicates that can occur on case-insensitive filesystems
        # where both *.gif and *.GIF patterns match the same files
        unique_gif_files = list(set(gif_files))

        self.logger.info(f"Discovered {len(unique_gif_files)} GIF files in {raw_dir}")
        return unique_gif_files

    def generate_jobs(self, gif_paths: list[Path], raw_dir: Path | None = None) -> list[CompressionJob]:
        """Generate all compression jobs for the given GIF files.

        Args:
            gif_paths: List of GIF file paths
            raw_dir: Raw directory path (needed for directory-based source detection)

        Returns:
            List of compression jobs to execute
        """
        jobs = []

        for gif_path in gif_paths:
            try:
                # Detect source from directory structure if enabled
                if self.detect_source_from_directory and raw_dir is not None:
                    source_platform, source_metadata = detect_source_from_directory(gif_path, raw_dir)
                else:
                    source_platform, source_metadata = "unknown", None
                
                # Extract metadata with source information
                metadata = extract_gif_metadata(
                    gif_path,
                    source_platform=source_platform,
                    source_metadata=source_metadata
                )

                # Create per-GIF folder name: {original_filename}_{gif_sha}
                folder_name = self._create_gif_folder_name(metadata.orig_filename, metadata.gif_sha)
                gif_folder = self.path_config.RENDERS_DIR / folder_name

                # Generate jobs for all compression variants
                for engine in self.compression_config.ENGINES:
                    for lossy in self.compression_config.LOSSY_LEVELS:
                        for ratio in self.compression_config.FRAME_KEEP_RATIOS:
                            for colors in self.compression_config.COLOR_KEEP_COUNTS:
                                # Generate clean output filename within the GIF's folder
                                output_filename = f"{engine}_l{lossy}_r{ratio:.2f}_c{colors}.gif"
                                output_path = gif_folder / output_filename

                                job = CompressionJob(
                                    gif_path=gif_path,
                                    metadata=metadata,
                                    engine=engine,
                                    lossy=lossy,
                                    frame_keep_ratio=ratio,
                                    color_keep_count=colors,
                                    output_path=output_path
                                )
                                jobs.append(job)

            except Exception as e:
                self.logger.error(f"Failed to process {gif_path}: {e}")
                # Move bad GIF
                try:
                    move_bad_gif(gif_path, self.path_config.BAD_GIFS_DIR)
                    self.logger.info(f"Moved bad GIF to: {self.path_config.BAD_GIFS_DIR}")
                except Exception as move_error:
                    self.logger.error(f"Failed to move bad GIF {gif_path}: {move_error}")

        self.logger.info(f"Generated {len(jobs)} compression jobs")
        return jobs

    def _create_gif_folder_name(self, orig_filename: str, gif_sha: str) -> str:
        """Create a filesystem-safe folder name for a GIF and its renders.

        Args:
            orig_filename: Original filename of the GIF
            gif_sha: SHA hash of the GIF

        Returns:
            Safe folder name in format: {sanitized_filename}_{sha}
        """
        # Remove file extension
        name_without_ext = Path(orig_filename).stem

        # Sanitize filename for cross-platform compatibility
        # Replace problematic characters with underscores
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        sanitized_name = "".join(c if c in safe_chars else "_" for c in name_without_ext)

        # Trim to reasonable length (keep space for SHA and underscore)
        max_name_length = 200  # Leave room for SHA (64 chars) + underscore + filesystem limits
        if len(sanitized_name) > max_name_length:
            sanitized_name = sanitized_name[:max_name_length].rstrip("_")

        # Use full SHA for guaranteed uniqueness and reverse lookup capability
        # This ensures we can always trace back to the original file
        return f"{sanitized_name}_{gif_sha}"

    def _load_existing_csv_records(self, csv_path: Path) -> set[tuple[str, str, int, float, int]]:
        """Load existing CSV records to identify completed jobs.

        Args:
            csv_path: Path to CSV file

        Returns:
            Set of tuples (gif_sha, engine, lossy, frame_keep_ratio, color_keep_count)
        """
        if not csv_path.exists():
            return set()

        try:
            existing_records = read_csv_as_dicts(csv_path)
            completed_jobs = set()

            for i, record in enumerate(existing_records):
                try:
                    # Validate required fields exist and are not empty
                    gif_sha = record.get("gif_sha", "").strip()
                    engine = record.get("engine", "").strip()
                    lossy_str = record.get("lossy", "").strip()
                    ratio_str = record.get("frame_keep_ratio", "").strip()
                    colors_str = record.get("color_keep_count", "").strip()

                    if not all([gif_sha, engine, lossy_str, ratio_str, colors_str]):
                        raise ValueError(f"Missing or empty required fields in record {i}")

                    # Validate SHA format (64 hex characters)
                    if len(gif_sha) != 64 or not all(c in "0123456789abcdef" for c in gif_sha.lower()):
                        raise ValueError(f"Invalid SHA format in record {i}: {gif_sha}")

                    key = (
                        gif_sha,
                        engine,
                        int(lossy_str),
                        float(ratio_str),
                        int(colors_str)
                    )
                    completed_jobs.add(key)
                except (KeyError, ValueError, TypeError) as e:
                    self.logger.warning(f"Skipping invalid CSV record {i}: {e}")

            self.logger.info(f"Loaded {len(completed_jobs)} existing CSV records")
            return completed_jobs

        except Exception as e:
            self.logger.error(f"Failed to load existing CSV records: {e}")
            return set()

    def filter_existing_jobs(self, jobs: list[CompressionJob], csv_path: Path) -> list[CompressionJob]:
        """Filter out jobs that have already been completed (if resume=True).

        Args:
            jobs: List of all compression jobs
            csv_path: Path to CSV file to check for existing records

        Returns:
            List of jobs that still need to be executed
        """
        if not self.resume:
            return jobs

        # Load existing CSV records
        existing_records = self._load_existing_csv_records(csv_path)

        filtered_jobs = []
        skipped_count = 0

        for job in jobs:
            # Check if job already exists in CSV
            job_key = (
                job.metadata.gif_sha,
                job.engine,
                job.lossy,
                job.frame_keep_ratio,
                job.color_keep_count
            )

            # Check if output file exists
            output_exists = job.output_path.exists()

            # Check if record exists in CSV
            record_exists = job_key in existing_records

            if output_exists and record_exists:
                # Job is complete, skip
                skipped_count += 1
            else:
                # Job needs to be processed
                filtered_jobs.append(job)

                # Clean up partial results
                if output_exists and not record_exists:
                    self.logger.warning(f"Removing incomplete output file: {job.output_path}")
                    try:
                        job.output_path.unlink()
                    except Exception as e:
                        self.logger.error(f"Failed to remove incomplete file: {e}")

        self.logger.info(f"Filtered {len(jobs)} jobs -> {len(filtered_jobs)} remaining (skipped {skipped_count})")
        return filtered_jobs

    def execute_job(self, job: CompressionJob) -> CompressionResult:
        """Execute a single compression job.

        Args:
            job: Compression job to execute

        Returns:
            Compression result

        Raises:
            RuntimeError: If compression or metrics calculation fails
        """
        try:
            # Ensure the per-GIF folder exists
            job.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert engine string to enum
            engine_enum = LossyEngine.GIFSICLE if job.engine == "gifsicle" else LossyEngine.ANIMATELY

            # Execute compression with all parameters in single pass
            compression_result = apply_compression_with_all_params(
                input_path=job.gif_path,
                output_path=job.output_path,
                lossy_level=job.lossy,
                frame_keep_ratio=job.frame_keep_ratio,
                color_keep_count=job.color_keep_count,
                engine=engine_enum
            )

            # Calculate quality metrics
            metrics_result = calculate_comprehensive_metrics(
                original_path=job.gif_path,
                compressed_path=job.output_path
            )

            # Create compression result
            result = CompressionResult(
                gif_sha=job.metadata.gif_sha,
                orig_filename=job.metadata.orig_filename,
                engine=job.engine,
                engine_version=compression_result.get("engine_version", "unknown"),
                lossy=job.lossy,
                frame_keep_ratio=job.frame_keep_ratio,
                color_keep_count=job.color_keep_count,
                kilobytes=metrics_result["kilobytes"],
                ssim=metrics_result["ssim"],
                render_ms=int(metrics_result["render_ms"]),
                orig_kilobytes=job.metadata.orig_kilobytes,
                orig_width=job.metadata.orig_width,
                orig_height=job.metadata.orig_height,
                orig_frames=job.metadata.orig_frames,
                orig_fps=job.metadata.orig_fps,
                orig_n_colors=job.metadata.orig_n_colors,
                entropy=job.metadata.entropy,
                source_platform=job.metadata.source_platform,
                source_metadata=json.dumps(job.metadata.source_metadata) if job.metadata.source_metadata else None,
                timestamp=datetime.now().isoformat()
            )

            return result

        except Exception as e:
            # Clean up failed output file
            if job.output_path.exists():
                try:
                    job.output_path.unlink()
                except Exception:
                    pass

            raise RuntimeError(f"Job execution failed for {job.gif_path}: {e}") from e

    def run(self, raw_dir: Path, csv_path: Path | None = None) -> dict[str, Any]:
        """Run the complete compression pipeline.

        Args:
            raw_dir: Directory containing raw GIF files
            csv_path: Path to output CSV file (auto-generated if None)

        Returns:
            Dictionary with pipeline execution statistics
        """
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            csv_path = self.path_config.CSV_DIR / f"results_{timestamp}.csv"

        self.logger.info(f"Starting compression pipeline: {raw_dir} -> {csv_path}")

        # Ensure output directories exist
        self.path_config.RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        self.path_config.CSV_DIR.mkdir(parents=True, exist_ok=True)
        self.path_config.BAD_GIFS_DIR.mkdir(parents=True, exist_ok=True)

        # Discover GIFs
        gif_paths = self.discover_gifs(raw_dir)
        if not gif_paths:
            self.logger.warning(f"No GIF files found in {raw_dir}")
            return {"status": "no_files", "processed": 0, "failed": 0, "skipped": 0}

        # Generate jobs
        all_jobs = self.generate_jobs(gif_paths, raw_dir)
        if not all_jobs:
            self.logger.warning("No valid compression jobs generated")
            return {"status": "no_jobs", "processed": 0, "failed": 0, "skipped": 0}

        # Filter existing jobs if resume is enabled
        jobs_to_run = self.filter_existing_jobs(all_jobs, csv_path)

        if not jobs_to_run:
            self.logger.info("All jobs already completed")
            return {"status": "all_complete", "processed": 0, "failed": 0, "skipped": len(all_jobs)}

        self.logger.info(f"Will execute {len(jobs_to_run)} jobs with {self.workers} workers")

        # Execute jobs in parallel
        processed_count = 0
        failed_count = 0
        moved_bad_gifs = set()  # Track which GIFs have already been moved to prevent duplicates

        try:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(execute_single_job, job): job
                    for job in jobs_to_run
                }

                # Process completed jobs
                for future in as_completed(future_to_job):
                    if self._shutdown_requested:
                        self.logger.info("Shutdown requested, cancelling remaining jobs...")
                        executor.shutdown(wait=False)
                        break

                    job = future_to_job[future]

                    try:
                        result = future.result()

                        # Append result to CSV
                        result_dict = asdict(result)
                        append_csv_row(csv_path, result_dict, self.csv_fieldnames)

                        processed_count += 1
                        self.logger.info(
                            f"Completed job {processed_count}/{len(jobs_to_run)}: "
                            f"{job.metadata.orig_filename} ({job.engine}, {job.lossy}, "
                            f"{job.frame_keep_ratio:.2f}, {job.color_keep_count})"
                        )

                    except Exception as e:
                        failed_count += 1
                        self.logger.error(f"Job failed: {e}")

                        # Move bad GIF only once per unique GIF file
                        gif_path_str = str(job.gif_path)
                        if gif_path_str not in moved_bad_gifs:
                            try:
                                move_bad_gif(job.gif_path, self.path_config.BAD_GIFS_DIR)
                                moved_bad_gifs.add(gif_path_str)
                                self.logger.info(f"Moved bad GIF to: {self.path_config.BAD_GIFS_DIR}")
                            except Exception as move_error:
                                self.logger.error(f"Failed to move bad GIF {job.gif_path}: {move_error}")

        except KeyboardInterrupt:
            self.logger.info("Pipeline interrupted by user")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {"status": "error", "processed": processed_count, "failed": failed_count, "error": str(e)}

        # Final statistics
        skipped_count = len(all_jobs) - len(jobs_to_run)
        processed_count + failed_count

        self.logger.info(
            f"Pipeline completed: {processed_count} processed, {failed_count} failed, "
            f"{skipped_count} skipped"
        )

        return {
            "status": "completed",
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_jobs": len(all_jobs),
            "csv_path": str(csv_path)
        }

    def find_original_gif_by_sha(self, gif_sha: str, raw_dir: Path) -> Path | None:
        """Find the original GIF file using its SHA hash.

        Args:
            gif_sha: SHA256 hash of the GIF content
            raw_dir: Directory containing original GIF files

        Returns:
            Path to original GIF file, or None if not found
        """
        for gif_path in raw_dir.glob("*.gif"):
            try:
                metadata = extract_gif_metadata(gif_path)
                if metadata.gif_sha == gif_sha:
                    return gif_path
            except Exception:
                # Skip files that can't be processed
                continue

        # Also check uppercase .GIF files
        for gif_path in raw_dir.glob("*.GIF"):
            try:
                metadata = extract_gif_metadata(gif_path)
                if metadata.gif_sha == gif_sha:
                    return gif_path
            except Exception:
                continue

        return None

    def find_original_gif_by_folder_name(self, folder_name: str, raw_dir: Path) -> Path | None:
        """Find the original GIF file using a render folder name.

        Args:
            folder_name: Name of the render folder (e.g., "funny_cat_abc123def...")
            raw_dir: Directory containing original GIF files

        Returns:
            Path to original GIF file, or None if not found
        """
        # Validate input
        if not folder_name or not isinstance(folder_name, str):
            return None

        # Extract SHA from folder name (everything after the last underscore)
        folder_parts = folder_name.split("_")
        if len(folder_parts) < 2:  # Must have at least filename_sha format
            return None

        # Get the last part which should be the SHA
        gif_sha = folder_parts[-1]

        # Validate SHA format (64 hex characters)
        if len(gif_sha) != 64:
            return None

        # Check if all characters are valid hex characters (case insensitive)
        try:
            # This will raise ValueError if gif_sha contains non-hex characters
            int(gif_sha, 16)
        except ValueError:
            # Invalid hex string
            return None

        return self.find_original_gif_by_sha(gif_sha, raw_dir)


def execute_single_job(job: CompressionJob) -> CompressionResult:
    """Execute a single compression job (for multiprocessing).

    Args:
        job: Compression job to execute

    Returns:
        Compression result
    """
    # Create a temporary pipeline instance for job execution
    from .config import DEFAULT_COMPRESSION_CONFIG, DEFAULT_PATH_CONFIG

    pipeline = CompressionPipeline(
        compression_config=DEFAULT_COMPRESSION_CONFIG,
        path_config=DEFAULT_PATH_CONFIG,
        workers=1,
        resume=False
    )

    return pipeline.execute_job(job)


def create_pipeline(
    raw_dir: Path,
    workers: int = 0,
    resume: bool = True
) -> CompressionPipeline:
    """Factory function to create a compression pipeline with default configs.

    Args:
        raw_dir: Directory containing raw GIF files
        workers: Number of worker processes
        resume: Whether to resume from existing progress

    Returns:
        Configured CompressionPipeline instance
    """
    from .config import DEFAULT_COMPRESSION_CONFIG, DEFAULT_PATH_CONFIG

    return CompressionPipeline(
        compression_config=DEFAULT_COMPRESSION_CONFIG,
        path_config=DEFAULT_PATH_CONFIG,
        workers=workers,
        resume=resume
    )
