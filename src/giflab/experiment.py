"""Experimental testing framework for GIF compression workflows.

This module provides comprehensive testing capabilities for comparing different
compression strategies, engine options, and workflow approaches. It's designed
for systematic evaluation of compression effectiveness before running on large datasets.

Key Features:
- Small-scale testing with ~10 diverse GIFs
- Multiple compression workflow comparison
- Extended engine options (dithering, optimization levels)
- Comprehensive results analysis and anomaly detection
- Experimental configuration management
"""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image, ImageDraw

from .config import CompressionConfig, PathConfig, DEFAULT_COMPRESSION_CONFIG
from .io import append_csv_row, setup_logging
from .lossy import LossyEngine, apply_compression_with_all_params
from .lossy_extended import (
    GifsicleOptimizationLevel, GifsicleDitheringMode,
    compress_with_gifsicle_extended, apply_compression_strategy
)
from .meta import GifMetadata, extract_gif_metadata
from .metrics import calculate_comprehensive_metrics
from .pipeline import CompressionJob, CompressionPipeline


@dataclass
class ExperimentalConfig:
    """Configuration for experimental compression testing."""
    
    # Test dataset settings
    TEST_GIFS_COUNT: int = 10
    SAMPLE_GIFS_PATH: Path = Path("data/experimental/sample_gifs")
    RESULTS_PATH: Path = Path("data/experimental/results")
    
    # Compression strategy variants
    STRATEGIES: List[str] = field(default_factory=lambda: [
        "pure_gifsicle",
        "pure_animately", 
        "animately_then_gifsicle",
        "gifsicle_dithered",
        "gifsicle_optimized"
    ])
    
    # Extended gifsicle options
    GIFSICLE_OPTIMIZATION_LEVELS: List[str] = field(default_factory=lambda: [
        "basic",      # --optimize
        "level1",     # -O1
        "level2",     # -O2
        "level3"      # -O3
    ])
    
    GIFSICLE_DITHERING_OPTIONS: List[str] = field(default_factory=lambda: [
        "none",       # --no-dither
        "floyd",      # --dither (default Floyd-Steinberg)
        "ordered"     # --dither=ordered
    ])
    
    # Test parameters (reduced for experimental use)
    FRAME_KEEP_RATIOS: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.5])
    COLOR_KEEP_COUNTS: List[int] = field(default_factory=lambda: [256, 64, 16])
    LOSSY_LEVELS: List[int] = field(default_factory=lambda: [0, 40, 120])
    
    # Analysis settings
    ANOMALY_THRESHOLD: float = 2.0  # Standard deviations for anomaly detection
    ENABLE_DETAILED_ANALYSIS: bool = True
    
    def __post_init__(self):
        """Ensure paths exist."""
        self.SAMPLE_GIFS_PATH.mkdir(parents=True, exist_ok=True)
        self.RESULTS_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentJob:
    """Represents a single experimental compression job."""
    
    gif_path: Path
    metadata: GifMetadata
    strategy: str
    engine: str
    optimization_level: str
    dithering_option: str
    lossy: int
    frame_keep_ratio: float
    color_keep_count: int
    output_path: Path
    
    def get_identifier(self) -> str:
        """Get unique identifier for this job."""
        return f"{self.strategy}_{self.engine}_{self.optimization_level}_{self.dithering_option}_l{self.lossy}_r{self.frame_keep_ratio:.2f}_c{self.color_keep_count}"


@dataclass
class ExperimentResult:
    """Results from experimental compression job."""
    
    job: ExperimentJob
    success: bool
    metrics: Dict[str, Any]
    compression_result: Dict[str, Any]
    error: Optional[str] = None
    processing_time_ms: int = 0


class ExperimentalPipeline:
    """Main experimental pipeline for testing compression strategies."""
    
    def __init__(self, config: ExperimentalConfig, workers: int = 0):
        """Initialize experimental pipeline.
        
        Args:
            config: Experimental configuration
            workers: Number of worker processes (0 = CPU count)
        """
        self.config = config
        self.workers = workers if workers > 0 else multiprocessing.cpu_count()
        self.logger = setup_logging(Path("logs"))
        
        # Create results directory structure
        self.results_dir = self.config.RESULTS_PATH / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV fieldnames for experimental results
        self.csv_fieldnames = [
            "gif_sha", "orig_filename", "strategy", "engine", "optimization_level",
            "dithering_option", "lossy", "frame_keep_ratio", "color_keep_count",
            "success", "error", "processing_time_ms", "kilobytes", "ssim",
            "render_ms", "orig_kilobytes", "orig_width", "orig_height",
            "orig_frames", "orig_fps", "orig_n_colors", "composite_quality",
            "mse", "rmse", "fsim", "gmsd", "chist", "edge_similarity",
            "texture_similarity", "sharpness_similarity", "timestamp"
        ]
    
    def generate_sample_gifs(self) -> List[Path]:
        """Generate diverse sample GIFs for testing.
        
        Returns:
            List of paths to generated sample GIFs
        """
        sample_gifs = []
        
        # Create different types of test GIFs
        gif_configs = [
            # (name, frames, size, content_type)
            ("simple_gradient", 5, (100, 100), "gradient"),
            ("complex_animation", 12, (150, 150), "shapes"),
            ("text_content", 8, (200, 100), "text"),
            ("photo_realistic", 6, (120, 120), "photo"),
            ("high_contrast", 10, (80, 80), "contrast"),
            ("many_colors", 15, (100, 100), "colors"),
            ("small_frames", 20, (50, 50), "micro"),
            ("large_frames", 3, (300, 200), "large"),
            ("single_frame", 1, (100, 100), "static"),
            ("rapid_motion", 25, (100, 100), "motion")
        ]
        
        self.logger.info(f"Generating {len(gif_configs)} sample GIFs...")
        
        for name, frames, size, content_type in gif_configs:
            gif_path = self.config.SAMPLE_GIFS_PATH / f"{name}.gif"
            if not gif_path.exists():
                self._create_test_gif(gif_path, frames, size, content_type)
            sample_gifs.append(gif_path)
        
        self.logger.info(f"Generated {len(sample_gifs)} sample GIFs")
        return sample_gifs
    
    def _create_test_gif(self, path: Path, frames: int, size: tuple, content_type: str):
        """Create a test GIF with specific characteristics."""
        images = []
        
        for i in range(frames):
            if content_type == "gradient":
                img = self._create_gradient_frame(size, i, frames)
            elif content_type == "shapes":
                img = self._create_shapes_frame(size, i, frames)
            elif content_type == "text":
                img = self._create_text_frame(size, i, frames)
            elif content_type == "photo":
                img = self._create_photo_frame(size, i, frames)
            elif content_type == "contrast":
                img = self._create_contrast_frame(size, i, frames)
            elif content_type == "colors":
                img = self._create_colors_frame(size, i, frames)
            elif content_type == "micro":
                img = self._create_micro_frame(size, i, frames)
            elif content_type == "large":
                img = self._create_large_frame(size, i, frames)
            elif content_type == "static":
                img = self._create_static_frame(size, i, frames)
            elif content_type == "motion":
                img = self._create_motion_frame(size, i, frames)
            else:
                img = self._create_simple_frame(size, i, frames)
            
            images.append(img)
        
        # Save GIF
        if images:
            images[0].save(
                path,
                save_all=True,
                append_images=images[1:],
                duration=100,  # 100ms per frame
                loop=0
            )
    
    def _create_gradient_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with gradient content."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Animated gradient
        for x in range(size[0]):
            for y in range(size[1]):
                r = int((x / size[0]) * 255)
                g = int((y / size[1]) * 255)
                b = int(((frame / total_frames) * 255))
                draw.point((x, y), (r, g, b))
        
        return img
    
    def _create_shapes_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with animated shapes."""
        img = Image.new('RGB', size, (50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        # Moving circle
        center_x = int((frame / total_frames) * (size[0] - 40) + 20)
        center_y = size[1] // 2
        radius = 15
        
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                    fill=(255, 100, 100))
        
        # Static rectangle
        draw.rectangle([size[0] - 40, 10, size[0] - 10, 40], 
                      fill=(100, 255, 100))
        
        return img
    
    def _create_text_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with text content."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Simple text animation
        text = f"Frame {frame + 1:02d}"
        text_color = (int((frame / total_frames) * 255), 50, 50)
        
        # Draw text at different positions
        x = 10 + (frame * 5) % (size[0] - 60)
        y = size[1] // 2 - 10
        
        draw.text((x, y), text, fill=text_color)
        
        return img
    
    def _create_photo_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with photo-realistic content."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Simulate photo-like content with noise and patterns
        import random
        random.seed(frame)
        
        for _ in range(size[0] * size[1] // 10):
            x = random.randint(0, size[0] - 1)
            y = random.randint(0, size[1] - 1)
            color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            draw.point((x, y), color)
        
        return img
    
    def _create_contrast_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with high contrast content."""
        img = Image.new('RGB', size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # High contrast pattern
        for x in range(0, size[0], 20):
            for y in range(0, size[1], 20):
                if (x + y + frame * 10) % 40 < 20:
                    draw.rectangle([x, y, x + 19, y + 19], fill=(255, 255, 255))
        
        return img
    
    def _create_colors_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with many colors."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Color spectrum
        for x in range(size[0]):
            for y in range(size[1]):
                r = int((x / size[0]) * 255)
                g = int((y / size[1]) * 255)
                b = int(((x + y + frame * 10) / (size[0] + size[1])) * 255) % 255
                draw.point((x, y), (r, g, b))
        
        return img
    
    def _create_micro_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a small frame with minimal content."""
        img = Image.new('RGB', size, (100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Simple pixel animation
        x = frame % size[0]
        y = frame % size[1]
        draw.point((x, y), (255, 255, 255))
        
        return img
    
    def _create_large_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a large frame with detailed content."""
        img = Image.new('RGB', size, (128, 128, 128))
        draw = ImageDraw.Draw(img)
        
        # Detailed pattern
        for x in range(0, size[0], 10):
            for y in range(0, size[1], 10):
                color = (
                    int((x / size[0]) * 255),
                    int((y / size[1]) * 255),
                    int((frame / total_frames) * 255)
                )
                draw.rectangle([x, y, x + 8, y + 8], fill=color)
        
        return img
    
    def _create_static_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a static frame."""
        img = Image.new('RGB', size, (150, 150, 150))
        draw = ImageDraw.Draw(img)
        
        # Simple static pattern
        draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(255, 255, 255))
        draw.text((20, 20), "STATIC", fill=(255, 255, 255))
        
        return img
    
    def _create_motion_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a frame with rapid motion."""
        img = Image.new('RGB', size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Multiple moving objects
        for obj in range(5):
            x = int(((frame + obj * 5) / total_frames) * (size[0] - 20)) + 10
            y = int(((frame + obj * 3) / total_frames) * (size[1] - 20)) + 10
            color = (255, obj * 50, frame * 10) 
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
        
        return img
    
    def _create_simple_frame(self, size: tuple, frame: int, total_frames: int) -> Image.Image:
        """Create a simple frame."""
        color = (frame * 30 % 255, 100, 150)
        img = Image.new('RGB', size, color)
        return img
    
    def generate_jobs(self, gif_paths: List[Path]) -> List[ExperimentJob]:
        """Generate experimental jobs for all strategies and parameters.
        
        Args:
            gif_paths: List of paths to GIF files
            
        Returns:
            List of experimental jobs
        """
        jobs = []
        
        for gif_path in gif_paths:
            try:
                metadata = extract_gif_metadata(gif_path)
                
                # Create output directory for this GIF
                gif_folder = self.results_dir / metadata.gif_sha[:8]
                gif_folder.mkdir(parents=True, exist_ok=True)
                
                # Generate jobs for each strategy
                for strategy in self.config.STRATEGIES:
                    for lossy in self.config.LOSSY_LEVELS:
                        for ratio in self.config.FRAME_KEEP_RATIOS:
                            for colors in self.config.COLOR_KEEP_COUNTS:
                                
                                if strategy == "pure_gifsicle":
                                    for opt_level in self.config.GIFSICLE_OPTIMIZATION_LEVELS:
                                        for dither in self.config.GIFSICLE_DITHERING_OPTIONS:
                                            jobs.append(self._create_job(
                                                gif_path, metadata, gif_folder,
                                                strategy, "gifsicle", opt_level, dither,
                                                lossy, ratio, colors
                                            ))
                                
                                elif strategy == "pure_animately":
                                    jobs.append(self._create_job(
                                        gif_path, metadata, gif_folder,
                                        strategy, "animately", "default", "none",
                                        lossy, ratio, colors
                                    ))
                                
                                elif strategy == "animately_then_gifsicle":
                                    # This will be a two-step process
                                    jobs.append(self._create_job(
                                        gif_path, metadata, gif_folder,
                                        strategy, "hybrid", "default", "none",
                                        lossy, ratio, colors
                                    ))
                                
                                elif strategy == "gifsicle_dithered":
                                    jobs.append(self._create_job(
                                        gif_path, metadata, gif_folder,
                                        strategy, "gifsicle", "basic", "floyd",
                                        lossy, ratio, colors
                                    ))
                                
                                elif strategy == "gifsicle_optimized":
                                    jobs.append(self._create_job(
                                        gif_path, metadata, gif_folder,
                                        strategy, "gifsicle", "level3", "none",
                                        lossy, ratio, colors
                                    ))
                
            except Exception as e:
                self.logger.error(f"Failed to generate jobs for {gif_path}: {e}")
        
        self.logger.info(f"Generated {len(jobs)} experimental jobs")
        return jobs
    
    def _create_job(self, gif_path: Path, metadata: GifMetadata, gif_folder: Path,
                   strategy: str, engine: str, opt_level: str, dither: str,
                   lossy: int, ratio: float, colors: int) -> ExperimentJob:
        """Create a single experimental job."""
        identifier = f"{strategy}_{engine}_{opt_level}_{dither}_l{lossy}_r{ratio:.2f}_c{colors}"
        output_path = gif_folder / f"{identifier}.gif"
        
        return ExperimentJob(
            gif_path=gif_path,
            metadata=metadata,
            strategy=strategy,
            engine=engine,
            optimization_level=opt_level,
            dithering_option=dither,
            lossy=lossy,
            frame_keep_ratio=ratio,
            color_keep_count=colors,
            output_path=output_path
        )
    
    def execute_job(self, job: ExperimentJob) -> ExperimentResult:
        """Execute a single experimental job.
        
        Args:
            job: Experimental job to execute
            
        Returns:
            Experiment result
        """
        start_time = time.time()
        
        try:
            # Execute compression based on strategy
            if job.strategy == "animately_then_gifsicle":
                compression_result = self._execute_hybrid_compression(job)
            else:
                compression_result = self._execute_single_engine_compression(job)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(job.gif_path, job.output_path)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return ExperimentResult(
                job=job,
                success=True,
                metrics=metrics,
                compression_result=compression_result,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return ExperimentResult(
                job=job,
                success=False,
                metrics={},
                compression_result={},
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def _execute_single_engine_compression(self, job: ExperimentJob) -> Dict[str, Any]:
        """Execute compression with a single engine."""
        if job.engine == "gifsicle":
            return self._execute_gifsicle_compression(job)
        elif job.engine == "animately":
            return self._execute_animately_compression(job)
        else:
            raise ValueError(f"Unknown engine: {job.engine}")
    
    def _execute_gifsicle_compression(self, job: ExperimentJob) -> Dict[str, Any]:
        """Execute gifsicle compression with extended options."""
        # Map string optimization levels to enum
        opt_level_map = {
            "basic": GifsicleOptimizationLevel.BASIC,
            "level1": GifsicleOptimizationLevel.LEVEL1,
            "level2": GifsicleOptimizationLevel.LEVEL2,
            "level3": GifsicleOptimizationLevel.LEVEL3
        }
        
        # Map string dithering modes to enum
        dither_mode_map = {
            "none": GifsicleDitheringMode.NONE,
            "floyd": GifsicleDitheringMode.FLOYD,
            "ordered": GifsicleDitheringMode.ORDERED
        }
        
        opt_level = opt_level_map.get(job.optimization_level, GifsicleOptimizationLevel.BASIC)
        dither_mode = dither_mode_map.get(job.dithering_option, GifsicleDitheringMode.NONE)
        
        return compress_with_gifsicle_extended(
            input_path=job.gif_path,
            output_path=job.output_path,
            lossy_level=job.lossy,
            frame_keep_ratio=job.frame_keep_ratio,
            color_keep_count=job.color_keep_count,
            optimization_level=opt_level,
            dithering_mode=dither_mode
        )
    
    def _execute_animately_compression(self, job: ExperimentJob) -> Dict[str, Any]:
        """Execute animately compression."""
        engine_enum = LossyEngine.ANIMATELY
        
        return apply_compression_with_all_params(
            input_path=job.gif_path,
            output_path=job.output_path,
            lossy_level=job.lossy,
            frame_keep_ratio=job.frame_keep_ratio,
            color_keep_count=job.color_keep_count,
            engine=engine_enum
        )
    
    def _execute_hybrid_compression(self, job: ExperimentJob) -> Dict[str, Any]:
        """Execute hybrid compression (Animately then Gifsicle)."""
        # Step 1: Process with Animately
        temp_path = job.output_path.parent / f"temp_{job.output_path.name}"
        
        animately_result = apply_compression_with_all_params(
            input_path=job.gif_path,
            output_path=temp_path,
            lossy_level=0,  # No lossy in first step
            frame_keep_ratio=job.frame_keep_ratio,
            color_keep_count=job.color_keep_count,
            engine=LossyEngine.ANIMATELY
        )
        
        # Step 2: Apply lossy compression with Gifsicle
        gifsicle_result = apply_compression_with_all_params(
            input_path=temp_path,
            output_path=job.output_path,
            lossy_level=job.lossy,
            frame_keep_ratio=1.0,  # No frame reduction in second step
            color_keep_count=256,  # No color reduction in second step (already done)
            engine=LossyEngine.GIFSICLE
        )
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        # Combine results
        return {
            "render_ms": animately_result["render_ms"] + gifsicle_result["render_ms"],
            "engine": "hybrid",
            "step1_result": animately_result,
            "step2_result": gifsicle_result
        }
    
    def run_experiment(self, sample_gifs: Optional[List[Path]] = None) -> Path:
        """Run the complete experimental pipeline.
        
        Args:
            sample_gifs: Optional list of sample GIFs (generates if None)
            
        Returns:
            Path to results CSV file
        """
        self.logger.info("Starting experimental compression pipeline...")
        
        # Generate or use provided sample GIFs
        if sample_gifs is None:
            sample_gifs = self.generate_sample_gifs()
        
        # Generate experimental jobs
        jobs = self.generate_jobs(sample_gifs)
        
        if not jobs:
            self.logger.warning("No experimental jobs generated")
            return Path()
        
        # Execute jobs in parallel
        results = []
        csv_path = self.results_dir / "experiment_results.csv"
        
        self.logger.info(f"Executing {len(jobs)} jobs with {self.workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self.execute_job, job): job for job in jobs}
            
            # Process results as they complete
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                
                # Write result to CSV immediately
                self._write_result_to_csv(result, csv_path)
                
                # Log progress
                if result.success:
                    self.logger.info(f"✅ Completed: {result.job.get_identifier()}")
                else:
                    self.logger.error(f"❌ Failed: {result.job.get_identifier()} - {result.error}")
        
        # Generate analysis report
        if self.config.ENABLE_DETAILED_ANALYSIS:
            self._generate_analysis_report(results, csv_path)
        
        self.logger.info(f"Experiment completed. Results saved to: {csv_path}")
        return csv_path
    
    def _write_result_to_csv(self, result: ExperimentResult, csv_path: Path):
        """Write a single result to the CSV file."""
        row = {
            "gif_sha": result.job.metadata.gif_sha,
            "orig_filename": result.job.metadata.orig_filename,
            "strategy": result.job.strategy,
            "engine": result.job.engine,
            "optimization_level": result.job.optimization_level,
            "dithering_option": result.job.dithering_option,
            "lossy": result.job.lossy,
            "frame_keep_ratio": result.job.frame_keep_ratio,
            "color_keep_count": result.job.color_keep_count,
            "success": result.success,
            "error": result.error or "",
            "processing_time_ms": result.processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add metrics if successful
        if result.success and result.metrics:
            row.update({
                "kilobytes": result.metrics.get("kilobytes", 0),
                "ssim": result.metrics.get("ssim", 0),
                "render_ms": result.compression_result.get("render_ms", 0),
                "orig_kilobytes": result.job.metadata.orig_kilobytes,
                "orig_width": result.job.metadata.orig_width,
                "orig_height": result.job.metadata.orig_height,
                "orig_frames": result.job.metadata.orig_frames,
                "orig_fps": result.job.metadata.orig_fps,
                "orig_n_colors": result.job.metadata.orig_n_colors,
                "composite_quality": result.metrics.get("composite_quality", 0),
                "mse": result.metrics.get("mse", 0),
                "rmse": result.metrics.get("rmse", 0),
                "fsim": result.metrics.get("fsim", 0),
                "gmsd": result.metrics.get("gmsd", 0),
                "chist": result.metrics.get("chist", 0),
                "edge_similarity": result.metrics.get("edge_similarity", 0),
                "texture_similarity": result.metrics.get("texture_similarity", 0),
                "sharpness_similarity": result.metrics.get("sharpness_similarity", 0)
            })
        
        append_csv_row(csv_path, row, self.csv_fieldnames)
    
    def _generate_analysis_report(self, results: List[ExperimentResult], csv_path: Path):
        """Generate detailed analysis report."""
        analysis_path = csv_path.parent / "analysis_report.json"
        
        # Basic statistics
        total_jobs = len(results)
        successful_jobs = sum(1 for r in results if r.success)
        failed_jobs = total_jobs - successful_jobs
        
        # Strategy performance
        strategy_stats = {}
        for result in results:
            strategy = result.job.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "successful": 0, "failed": 0}
            
            strategy_stats[strategy]["total"] += 1
            if result.success:
                strategy_stats[strategy]["successful"] += 1
            else:
                strategy_stats[strategy]["failed"] += 1
        
        # Error analysis
        error_counts = {}
        for result in results:
            if not result.success and result.error:
                error_type = result.error.split(":")[0]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Performance metrics (for successful jobs)
        successful_results = [r for r in results if r.success]
        if successful_results:
            processing_times = [r.processing_time_ms for r in successful_results]
            ssim_scores = [r.metrics.get("ssim", 0) for r in successful_results if r.metrics]
            compression_ratios = [
                r.job.metadata.orig_kilobytes / r.metrics.get("kilobytes", 1)
                for r in successful_results
                if r.metrics and r.metrics.get("kilobytes", 0) > 0
            ]
            
            performance_stats = {
                "processing_time_ms": {
                    "mean": sum(processing_times) / len(processing_times),
                    "min": min(processing_times),
                    "max": max(processing_times)
                },
                "ssim_scores": {
                    "mean": sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0,
                    "min": min(ssim_scores) if ssim_scores else 0,
                    "max": max(ssim_scores) if ssim_scores else 0
                },
                "compression_ratios": {
                    "mean": sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
                    "min": min(compression_ratios) if compression_ratios else 0,
                    "max": max(compression_ratios) if compression_ratios else 0
                }
            }
        else:
            performance_stats = {}
        
        # Generate report
        report = {
            "experiment_summary": {
                "total_jobs": total_jobs,
                "successful_jobs": successful_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0
            },
            "strategy_performance": strategy_stats,
            "error_analysis": error_counts,
            "performance_metrics": performance_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        with open(analysis_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Analysis report saved to: {analysis_path}")


def create_experimental_pipeline(
    test_gifs_count: int = 10,
    workers: int = 0,
    enable_analysis: bool = True
) -> ExperimentalPipeline:
    """Factory function to create experimental pipeline.
    
    Args:
        test_gifs_count: Number of test GIFs to generate
        workers: Number of worker processes
        enable_analysis: Whether to enable detailed analysis
        
    Returns:
        Configured experimental pipeline
    """
    config = ExperimentalConfig(
        TEST_GIFS_COUNT=test_gifs_count,
        ENABLE_DETAILED_ANALYSIS=enable_analysis
    )
    
    return ExperimentalPipeline(config, workers) 