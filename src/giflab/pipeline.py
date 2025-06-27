"""Compression pipeline orchestrator with resume functionality."""

import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

from .config import CompressionConfig, PathConfig
from .meta import GifMetadata, extract_gif_metadata
from .io import append_csv_row, move_bad_gif, setup_logging, read_csv_as_dicts
from .lossy import apply_compression_with_all_params, LossyEngine
from .metrics import calculate_comprehensive_metrics


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
    entropy: Optional[float]
    
    # Timestamp
    timestamp: str


class CompressionPipeline:
    """Main pipeline for orchestrating GIF compression with resume capability."""
    
    def __init__(
        self,
        compression_config: CompressionConfig,
        path_config: PathConfig,
        workers: int = 0,
        resume: bool = True
    ):
        """Initialize the compression pipeline.
        
        Args:
            compression_config: Configuration for compression variants
            path_config: Configuration for file paths
            workers: Number of worker processes (0 = CPU count)
            resume: Whether to resume from existing progress
        """
        self.compression_config = compression_config
        self.path_config = path_config
        self.workers = workers if workers > 0 else multiprocessing.cpu_count()
        self.resume = resume
        self.logger = setup_logging(path_config.LOGS_DIR)
        
        # CSV fieldnames based on project scope
        self.csv_fieldnames = [
            "gif_sha", "orig_filename", "engine", "lossy", 
            "frame_keep_ratio", "color_keep_count", "kilobytes", "ssim",
            "render_ms", "orig_kilobytes", "orig_width", "orig_height", 
            "orig_frames", "orig_fps", "orig_n_colors", "entropy", "timestamp"
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
    
    def discover_gifs(self, raw_dir: Path) -> List[Path]:
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
        
        self.logger.info(f"Discovered {len(gif_files)} GIF files in {raw_dir}")
        return gif_files
    
    def generate_jobs(self, gif_paths: List[Path]) -> List[CompressionJob]:
        """Generate all compression jobs for the given GIF files.
        
        Args:
            gif_paths: List of GIF file paths
            
        Returns:
            List of compression jobs to execute
        """
        jobs = []
        
        for gif_path in gif_paths:
            try:
                # Extract metadata
                metadata = extract_gif_metadata(gif_path)
                
                # Generate jobs for all compression variants
                for engine in self.compression_config.ENGINES:
                    for lossy in self.compression_config.LOSSY_LEVELS:
                        for ratio in self.compression_config.FRAME_KEEP_RATIOS:
                            for colors in self.compression_config.COLOR_KEEP_COUNTS:
                                # Generate output filename
                                output_name = (
                                    f"{metadata.gif_sha}_{engine}_"
                                    f"l{lossy}_r{ratio:.2f}_c{colors}.gif"
                                )
                                output_path = self.path_config.RENDERS_DIR / output_name
                                
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
    
    def _load_existing_csv_records(self, csv_path: Path) -> Set[Tuple[str, str, int, float, int]]:
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
            
            for record in existing_records:
                try:
                    key = (
                        record["gif_sha"],
                        record["engine"],
                        int(record["lossy"]),
                        float(record["frame_keep_ratio"]),
                        int(record["color_keep_count"])
                    )
                    completed_jobs.add(key)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Skipping invalid CSV record: {e}")
            
            self.logger.info(f"Loaded {len(completed_jobs)} existing CSV records")
            return completed_jobs
            
        except Exception as e:
            self.logger.error(f"Failed to load existing CSV records: {e}")
            return set()
    
    def filter_existing_jobs(self, jobs: List[CompressionJob], csv_path: Path) -> List[CompressionJob]:
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
                lossy=job.lossy,
                frame_keep_ratio=job.frame_keep_ratio,
                color_keep_count=job.color_keep_count,
                kilobytes=metrics_result["kilobytes"],
                ssim=metrics_result["ssim"],
                render_ms=metrics_result["render_ms"],
                orig_kilobytes=job.metadata.orig_kilobytes,
                orig_width=job.metadata.orig_width,
                orig_height=job.metadata.orig_height,
                orig_frames=job.metadata.orig_frames,
                orig_fps=job.metadata.orig_fps,
                orig_n_colors=job.metadata.orig_n_colors,
                entropy=job.metadata.entropy,
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
            
            raise RuntimeError(f"Job execution failed for {job.gif_path}: {e}")
    
    def run(self, raw_dir: Path, csv_path: Optional[Path] = None) -> Dict[str, Any]:
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
        all_jobs = self.generate_jobs(gif_paths)
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
                        
                        # Move bad GIF if this is the first failure for this GIF
                        try:
                            if all(f.gif_path != job.gif_path for f in future_to_job.values()):
                                move_bad_gif(job.gif_path, self.path_config.BAD_GIFS_DIR)
                        except Exception:
                            pass
        
        except KeyboardInterrupt:
            self.logger.info("Pipeline interrupted by user")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {"status": "error", "processed": processed_count, "failed": failed_count, "error": str(e)}
        
        # Final statistics
        skipped_count = len(all_jobs) - len(jobs_to_run)
        total_expected = processed_count + failed_count
        
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