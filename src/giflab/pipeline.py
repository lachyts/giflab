"""Compression pipeline orchestrator with resume functionality."""

import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .config import CompressionConfig, PathConfig
from .meta import GifMetadata, extract_gif_metadata
from .io import append_csv_row, move_bad_gif, setup_logging


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
    """Result of a compression job."""
    
    job: CompressionJob
    success: bool
    render_ms: int
    output_size_kb: float
    ssim: float
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


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
                move_bad_gif(gif_path, self.path_config.BAD_GIFS_DIR)
        
        self.logger.info(f"Generated {len(jobs)} compression jobs")
        return jobs
    
    def filter_existing_jobs(self, jobs: List[CompressionJob]) -> List[CompressionJob]:
        """Filter out jobs that have already been completed (if resume=True).
        
        Args:
            jobs: List of all compression jobs
            
        Returns:
            List of jobs that still need to be executed
        """
        if not self.resume:
            return jobs
        
        # TODO: Implement job filtering based on existing renders and CSV records
        # This will check for existing output files and CSV entries
        # This will be implemented in Stage 6 (S6)
        self.logger.info("Resume functionality not yet implemented")
        return jobs  
    
    def execute_job(self, job: CompressionJob) -> CompressionResult:
        """Execute a single compression job.
        
        Args:
            job: Compression job to execute
            
        Returns:
            Compression result
        """
        # TODO: Implement actual compression execution
        # This will be implemented in Stage 6 (S6) after individual modules are ready
        raise NotImplementedError("Job execution not yet implemented")
    
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
        
        # Discover GIFs
        gif_paths = self.discover_gifs(raw_dir)
        if not gif_paths:
            self.logger.warning(f"No GIF files found in {raw_dir}")
            return {"status": "no_files", "processed": 0}
        
        # Generate jobs
        all_jobs = self.generate_jobs(gif_paths)
        if not all_jobs:
            self.logger.warning("No valid compression jobs generated")
            return {"status": "no_jobs", "processed": 0}
        
        # Filter existing jobs if resume is enabled
        jobs_to_run = self.filter_existing_jobs(all_jobs)
        
        self.logger.info(f"Will execute {len(jobs_to_run)} jobs with {self.workers} workers")
        
        # TODO: Implement parallel job execution
        # This will be implemented in Stage 6 (S6)
        raise NotImplementedError("Pipeline execution not yet implemented")


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