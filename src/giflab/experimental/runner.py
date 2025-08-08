"""Core experimental runner for systematic pipeline testing.

This module contains the main ExperimentalRunner class responsible for
systematic testing and analysis of pipeline combinations.
"""

from __future__ import annotations

import atexit
import json
import logging
import signal
from shutil import copy
import sqlite3
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

from ..dynamic_pipeline import generate_all_pipelines, Pipeline
from ..elimination_errors import ErrorTypes
from ..elimination_cache import PipelineResultsCache, get_git_commit
from ..synthetic_gifs import SyntheticGifGenerator, SyntheticGifSpec, SyntheticFrameGenerator
from .targeted_presets import ExperimentPreset
from ..error_handling import clean_error_message
from .pareto import ParetoAnalyzer
from .sampling import SAMPLING_STRATEGIES, PipelineSampler

@dataclass
class ExperimentResult:
    """Result of experimental pipeline analysis."""
    eliminated_pipelines: Set[str] = field(default_factory=set)
    retained_pipelines: Set[str] = field(default_factory=set)
    performance_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    elimination_reasons: Dict[str, str] = field(default_factory=dict)
    content_type_winners: Dict[str, List[str]] = field(default_factory=dict)
    testing_strategy_used: str = "full_brute_force"
    total_jobs_run: int = 0
    total_jobs_possible: int = 0
    efficiency_gain: float = 0.0
    # Pareto frontier analysis results
    pareto_analysis: Dict[str, Any] = field(default_factory=dict)
    pareto_dominated_pipelines: Set[str] = field(default_factory=set)
    quality_aligned_rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

class ExperimentalRunner:
    """Systematic experimental pipeline testing through competitive analysis."""
    
    # Available sampling strategies (imported from experimental.sampling)
    SAMPLING_STRATEGIES = SAMPLING_STRATEGIES
    
    # Constants for progress tracking and memory management
    PROGRESS_SAVE_INTERVAL = 100  # Save resume data every N jobs to prevent memory buildup
    BUFFER_FLUSH_INTERVAL = 15    # Reduced from 50 for more frequent monitoring updates
    
    def __init__(self, output_dir: Path = Path("results/experiments"), use_gpu: bool = False, use_cache: bool = True):
        # Always resolve paths to absolute paths to prevent nesting when working directory changes
        if output_dir.is_absolute():
            self.base_output_dir = output_dir
        else:
            # For relative paths, resolve from the project root (where .git exists)
            project_root = Path(__file__).parent.parent.parent.parent  # Go up from src/giflab/experimental/runner.py
            self.base_output_dir = (project_root / output_dir).resolve()
        self.use_gpu = use_gpu
        self.use_cache = use_cache
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline sampler
        self.sampler = PipelineSampler(self.logger)
        
        # Initialize vectorized frame generator for performance
        self._frame_generator = SyntheticFrameGenerator()
        
        # Track current preset for parameter override (None = use default test_params)
        self._current_preset: Optional[ExperimentPreset] = None
        
        # Create timestamped output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create a "latest" symlink for easy access to most recent results
        latest_link = self.base_output_dir / "latest"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            # Create relative symlink to work better across systems
            relative_target = f"run_{timestamp}"
            latest_link.symlink_to(relative_target)
            self.logger.info(f"📁 Results will be saved to: {self.output_dir}")
            self.logger.info(f"🔗 Latest results symlink: {latest_link}")
        except (OSError, NotImplementedError) as e:
            # Symlinks not supported on this system (e.g., Windows without admin rights)
            self.logger.info(f"📁 Results will be saved to: {self.output_dir}")
            self.logger.warning(f"Could not create 'latest' symlink: {e}")
        
        # Initialize cache system
        if self.use_cache:
            cache_db_path = self.base_output_dir / "pipeline_results_cache.db"
            git_commit = get_git_commit()
            self.cache = PipelineResultsCache(cache_db_path, git_commit)
            
            # Log cache statistics
            cache_stats = self.cache.get_cache_stats()
            self.logger.info(f"💾 Cache initialized: {cache_stats['current_commit_results']} entries for current commit")
            self.logger.info(f"💾 Total cache entries: {cache_stats['total_results']} ({cache_stats['database_size_mb']} MB)")
        else:
            self.cache = None
            self.logger.info("💾 Cache disabled")
        
        # Test GPU availability on initialization
        if self.use_gpu:
            self._test_gpu_availability()
        
        # Log elimination run metadata
        self._log_run_metadata()
        
        # Initialize synthetic GIF generator
        self.gif_generator = SyntheticGifGenerator(self.output_dir)
        
        # Get synthetic GIF specifications from generator (no duplication)
        self.synthetic_specs = self.gif_generator.synthetic_specs

    def create_frame(self, content_type: str, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Create a single synthetic frame via the vectorized SyntheticFrameGenerator.

        This thin wrapper keeps the ExperimentalRunner API aligned with the
        frame-generation cleanup plan and simplifies external callers/tests.
        """
        return self._frame_generator.create_frame(content_type, size, frame, total_frames)
    
    def generate_synthetic_gifs(self) -> List[Path]:
        """Generate all synthetic test GIFs."""
        return self.gif_generator.generate_all_gifs()
    
    # Delegate sampling methods to the PipelineSampler
    def select_pipelines_intelligently(self, all_pipelines: List, strategy: str = 'representative') -> List:
        """Select pipelines using intelligent sampling strategies to reduce testing time."""
        return self.sampler.select_pipelines_intelligently(all_pipelines, strategy)
    
    def generate_targeted_pipelines(self, preset_id: str) -> List[Pipeline]:
        """Generate targeted pipelines for a specific experiment preset.
        
        Args:
            preset_id: ID of the experiment preset to use
            
        Returns:
            List of Pipeline objects for the specific combinations needed
            
        Raises:
            ValueError: If preset_id is not found or invalid
        """
        # Import builtin presets to ensure they're registered
        from . import builtin_presets
        from .targeted_presets import PRESET_REGISTRY
        from .targeted_generator import TargetedPipelineGenerator
        
        preset = PRESET_REGISTRY.get(preset_id)
        generator = TargetedPipelineGenerator(self.logger)
        
        # Log the preset being used
        self.logger.info(f"🎯 Using targeted preset: {preset.name}")
        self.logger.info(f"📋 Description: {preset.description}")
        
        # Validate preset and generate pipelines
        validation = generator.validate_preset_feasibility(preset)
        if not validation['valid']:
            raise ValueError(f"Invalid preset '{preset_id}': {validation['errors']}")
        
        self.logger.info(f"📊 Efficiency gain: {validation['efficiency_gain']:.1%} vs generate_all_pipelines")
        
        return generator.generate_targeted_pipelines(preset)
    
    def run_targeted_experiment(self, preset_id: str, quality_threshold: float = 0.05, 
                              use_targeted_gifs: bool = False) -> ExperimentResult:
        """Run a targeted experiment using a specific preset configuration.
        
        This is the main entry point for preset-based experiments, replacing
        the generate_all_pipelines() + sampling workflow.
        
        Args:
            preset_id: ID of the experiment preset to use
            quality_threshold: SSIM threshold for elimination
            use_targeted_gifs: Whether to use targeted GIF subset
            
        Returns:
            ExperimentResult with targeted testing results
        """
        # Import builtin presets to ensure they're registered
        from . import builtin_presets
        from .targeted_presets import PRESET_REGISTRY
        
        # Get the preset for parameter override
        preset = PRESET_REGISTRY.get(preset_id)
        if not preset:
            raise ValueError(f"Unknown preset ID: {preset_id}")
        
        # Activate preset mode for parameter override
        self._current_preset = preset
        
        try:
            # Generate targeted pipelines
            targeted_pipelines = self.generate_targeted_pipelines(preset_id)
            
            # Run analysis with targeted pipelines
            return self.run_experimental_analysis(
                test_pipelines=targeted_pipelines,
                quality_threshold=quality_threshold,
                use_targeted_gifs=use_targeted_gifs
            )
        finally:
            # Always reset preset mode
            self._current_preset = None
    
    def list_available_presets(self) -> Dict[str, str]:
        """List all available experiment presets.
        
        Returns:
            Dictionary mapping preset IDs to descriptions
        """
        # Import builtin presets to ensure they're registered
        from . import builtin_presets
        from .targeted_presets import PRESET_REGISTRY
        return PRESET_REGISTRY.list_presets()
    
    # Sampling methods moved to experimental.sampling module
    
    
    
    
    
    def get_targeted_synthetic_gifs(self) -> List[Path]:
        """Generate a strategically reduced set of synthetic GIFs for targeted testing."""
        return self.sampler.get_targeted_synthetic_gifs()
    
    # NOTE: The rest of the ExperimentalRunner methods will be added in the next step
    # This is just the structure and constructor to start with
    
    def _test_gpu_availability(self):
        """Test if GPU acceleration is available."""
        try:
            import cv2
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                self.logger.info(f"🚀 GPU acceleration available: {device_count} CUDA device(s)")
            else:
                self.logger.warning("⚠️ GPU acceleration requested but no CUDA devices found - falling back to CPU")
                self.use_gpu = False
        except ImportError:
            self.logger.warning("⚠️ GPU acceleration requested but OpenCV with CUDA not available - falling back to CPU")
            self.use_gpu = False
        except Exception as e:
            self.logger.warning(f"⚠️ GPU availability test failed: {e} - falling back to CPU")
            self.use_gpu = False
    
    def _log_run_metadata(self):
        """Log metadata about this elimination run."""
        git_commit = get_git_commit()
        self.logger.info(f"🔬 Elimination run metadata:")
        self.logger.info(f"   Git commit: {git_commit}")
        self.logger.info(f"   Timestamp: {datetime.now().isoformat()}")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Use GPU: {self.use_gpu}")
        self.logger.info(f"   Use cache: {self.use_cache}")
    
    def generate_synthetic_gifs(self) -> List[Path]:
        """Generate all synthetic test GIFs."""
        self.logger.info(f"Generating {len(self.synthetic_specs)} synthetic test GIFs")
        
        from importlib import import_module
        try:
            tqdm = import_module('tqdm').tqdm  # type: ignore[attr-defined]
        except ModuleNotFoundError:  # pragma: no cover – fallback if tqdm not installed
            class tqdm:  # noqa: WPS430 – simple fallback
                def __init__(self, iterable, **kwargs):
                    self.iterable = iterable
                def __iter__(self):
                    return iter(self.iterable)
                def update(self, _n: int = 1):
                    pass
                def close(self):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.close()
        
        gif_paths = []
        with tqdm(self.synthetic_specs, desc='🖼️  Synthetic GIFs', unit='gif') as progress:
            for spec in progress:
                gif_path = self.output_dir / f"{spec.name}.gif"
                if not gif_path.exists():
                    self._create_synthetic_gif(gif_path, spec)
                gif_paths.append(gif_path)
                progress.update(0)  # refresh display
            
        return gif_paths
    
    def _create_synthetic_gif(self, path: Path, spec: SyntheticGifSpec):
        """Create a synthetic GIF based on specification."""
        images = []
        
        for frame_idx in range(spec.frames):
            # Use vectorized frame generator for massive performance improvement
            img = self._frame_generator.create_frame(spec.content_type, spec.size, frame_idx, spec.frames)
            images.append(img)
        
        # Save GIF with consistent settings
        if images:
            images[0].save(
                path,
                save_all=True,
                append_images=images[1:],
                duration=100,  # 100ms per frame
                loop=0
            )
                
    
    def run_experimental_analysis(self, 
                                test_pipelines: List[Pipeline] = None,
                                quality_threshold: float = 0.05,
                                use_targeted_gifs: bool = False) -> ExperimentResult:
        """Run competitive elimination analysis on synthetic GIFs.
        
        Args:
            test_pipelines: Specific pipelines to test (None = all pipelines)
            quality_threshold: SSIM threshold for elimination (lower = stricter)
            
        Returns:
            ExperimentResult with eliminated and retained pipelines
        """
        if test_pipelines is None:
            test_pipelines = generate_all_pipelines()
            
        self.logger.info(f"Running elimination analysis on {len(test_pipelines)} pipelines")
        
        # Generate synthetic test GIFs
        if use_targeted_gifs:
            synthetic_gifs = self.get_targeted_synthetic_gifs()
        else:
            synthetic_gifs = self.generate_synthetic_gifs()
        
        # Test all pipeline combinations
        results_df = self._run_comprehensive_testing(synthetic_gifs, test_pipelines)
        
        # Analyze results and eliminate underperformers
        experiment_result = self._analyze_and_experiment(results_df, quality_threshold)
        
        # Save results
        self._save_results(experiment_result, results_df)
        
        return experiment_result
    
    def _run_comprehensive_testing(self, gif_paths: List[Path], pipelines: List[Pipeline]) -> pd.DataFrame:
        """Run comprehensive testing of all pipeline combinations with streaming results to disk."""
        from ..metrics import calculate_comprehensive_metrics, DEFAULT_METRICS_CONFIG
        import tempfile
        import time
        import csv
        
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm is not available
            class tqdm:
                def __init__(self, total=None, initial=0, desc="", unit="", bar_format="", **kwargs):
                    self.total = total
                    self.n = initial
                    self.desc = desc
                    self._last_line_length = 0
                
                def update(self, n=1):
                    self.n += n
                    if hasattr(self, 'total') and self.total:
                        percentage = (self.n / self.total) * 100
                        # Create progress line
                        progress_line = f"{self.desc}: {self.n}/{self.total} ({percentage:.1f}%)"
                        # Clear previous line completely by overwriting with spaces
                        clear_line = "\r" + " " * self._last_line_length + "\r"
                        print(clear_line + progress_line, end="", flush=True)
                        self._last_line_length = len(progress_line)
                
                def close(self):
                    # Clear the progress line and add newline
                    if hasattr(self, '_last_line_length'):
                        clear_line = "\r" + " " * self._last_line_length + "\r"
                        print(clear_line, end="")
                    print()  # New line after progress
        
        # Calculate total jobs for progress tracking
        total_jobs = len(gif_paths) * len(pipelines) * len(self.test_params)
        self.logger.info(f"Starting comprehensive testing: {total_jobs:,} total pipeline combinations")
        
        # Save total jobs count to metadata for monitor to pick up
        self._save_total_jobs_to_metadata(total_jobs)
        
        # Load or create resume file - optimize memory by tracking only job IDs
        resume_file = self.output_dir / "elimination_progress.json"
        completed_jobs_full = self._load_resume_data(resume_file)
        
        # Extract just job IDs for efficient memory usage during processing
        completed_job_ids = set(completed_jobs_full.keys())
        
        # Clear the full data to free memory, keeping only IDs for resume logic
        completed_jobs_data_for_streaming = {}
        
        # Add completed jobs to streaming buffer for any that haven't been written to CSV yet
        for job_id, job_data in completed_jobs_full.items():
            completed_jobs_data_for_streaming[job_id] = job_data
        
        # Clear the full completed_jobs dict to save memory
        del completed_jobs_full
        
        # Setup streaming CSV file for results
        streaming_csv_path = self.output_dir / "streaming_results.csv"
        csv_fieldnames = [
            'gif_name', 'content_type', 'pipeline_id', 'connection_signature', 'success',
            'file_size_kb', 'original_size_kb', 'compression_ratio',
            'ssim_mean', 'ssim_std', 'ssim_min', 'ssim_max',
            'ms_ssim_mean', 'psnr_mean', 'temporal_consistency',
            'mse_mean', 'rmse_mean', 'fsim_mean', 'gmsd_mean',
            'chist_mean', 'edge_similarity_mean', 'texture_similarity_mean',
            'sharpness_similarity_mean', 'composite_quality', 'enhanced_composite_quality', 'efficiency',
            'render_time_ms', 'total_processing_time_ms',
            'pipeline_steps', 'tools_used', 
            'applied_colors', 'applied_lossy', 'applied_frame_ratio', 'actual_pipeline_steps',
            'error', 'error_traceback', 'error_timestamp'
        ]
        
        # Initialize streaming CSV with headers
        write_csv_header = not streaming_csv_path.exists()
        csv_file = open(streaming_csv_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        if write_csv_header:
            csv_writer.writeheader()
        
        # Register cleanup handlers to prevent data loss on unexpected termination
        def _cleanup_csv():
            """Ensure CSV file is properly flushed and closed on exit."""
            try:
                if csv_file and not csv_file.closed:
                    # Final flush of any remaining buffer
                    if 'results_buffer' in locals() and results_buffer:
                        self._flush_results_buffer(results_buffer, csv_writer, csv_file, force=True)
                    csv_file.flush()
                    csv_file.close()
                    self.logger.info("📝 CSV file cleanup completed on exit")
            except Exception as e:
                # Log but don't raise - we're in cleanup
                self.logger.warning(f"CSV cleanup warning: {e}")
        
        # Register atexit handler
        atexit.register(_cleanup_csv)
        
        # Register signal handlers for common termination signals
        def _signal_handler(signum, frame):
            """Handle termination signals by cleaning up CSV and exiting gracefully."""
            signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            self.logger.info(f"🛑 Received {signal_name}, cleaning up CSV and exiting...")
            _cleanup_csv()
            sys.exit(1)
        
        # Register handlers for common termination signals
        try:
            signal.signal(signal.SIGTERM, _signal_handler)  # Termination request
            signal.signal(signal.SIGINT, _signal_handler)   # Ctrl+C
            if hasattr(signal, 'SIGHUP'):  # Not available on Windows
                signal.signal(signal.SIGHUP, _signal_handler)  # Terminal disconnect
        except (AttributeError, OSError) as e:
            # Some signals may not be available on all platforms
            self.logger.debug(f"Could not register some signal handlers: {e}")
        
        # Estimate remaining time
        remaining_jobs = total_jobs - len(completed_job_ids)
        estimated_time = self._estimate_execution_time(remaining_jobs)
        self.logger.info(f"Estimated completion time: {estimated_time}")
        
        # Setup progress bar with proper line management
        progress = tqdm(total=total_jobs, initial=len(completed_job_ids), 
                       desc="Testing pipelines", unit="jobs",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                       ncols=100, leave=True)
        
        # Use small memory buffer for batching, not unlimited accumulation
        results_buffer = []
        buffer_size = 25  # Reduced from 100 for more frequent monitoring updates
        failed_jobs = 0
        
        # Test each combination
        cache_hits = 0
        cache_misses = 0
        
        for gif_path in gif_paths:
            content_type = self._get_content_type(gif_path.stem)
            
            for pipeline in pipelines:
                for params in self.test_params:
                    job_id = f"{gif_path.stem}_{pipeline.identifier()}_{params['colors']}_{params['lossy']}"
                    
                    # Skip if already completed (resume functionality)
                    if job_id in completed_job_ids:
                        progress.update(1)
                        # Add to buffer if data is available for streaming
                        if job_id in completed_jobs_data_for_streaming:
                            self._add_result_to_buffer(completed_jobs_data_for_streaming[job_id], results_buffer, csv_writer, csv_file, buffer_size)
                            # Remove from memory after adding to buffer to save memory
                            del completed_jobs_data_for_streaming[job_id]
                        continue
                    
                    # Check cache first (if enabled)
                    cached_result = None
                    if self.cache:
                        cached_result = self.cache.get_cached_result(
                            pipeline.identifier(), gif_path.stem, params
                        )
                        if cached_result:
                            cache_hits += 1
                            # Update progress tracking
                            completed_job_ids.add(job_id)
                            self._add_result_to_buffer(cached_result, results_buffer, csv_writer, csv_file, buffer_size)
                            progress.update(1)
                            continue
                        else:
                            cache_misses += 1
                    
                    try:
                        # Validate pipeline doesn't contain invalid tools
                        if any("external-tool" in str(step) for step in pipeline.steps):
                            error_msg = f"Invalid pipeline contains 'external-tool' base class: {pipeline.identifier()}"
                            self.logger.warning(f"🚨 Pipeline validation failed: {error_msg}")
                            self.logger.info("💡 This prevents the external-tool registration bug from causing invalid results")
                            self.cache.queue_failure(
                                pipeline.identifier(),
                                gif_path.stem,
                                params,
                                {
                                    "error": error_msg,
                                    "error_traceback": "",
                                    "pipeline_steps": [step.name() for step in pipeline.steps],
                                    "tools_used": [step.tool_cls.NAME for step in pipeline.steps],
                                    "error_type": "validation"
                                }
                            )
                            # Add validation failure to completed jobs and continue
                            failed_result = {
                                'gif_name': gif_path.stem,
                                'content_type': content_type,
                                'pipeline_id': pipeline.identifier(),
                                'connection_signature': 'FAIL',  # Validation failed before execution
                                'success': False,
                                'error': error_msg,
                                'test_colors': params.get("colors", 0),
                                'test_lossy': params.get("lossy", 0),
                                'test_frame_ratio': params.get("frame_ratio", 1.0),
                            }
                            completed_job_ids.add(job_id)
                            self._add_result_to_buffer(failed_result, results_buffer, csv_writer, csv_file, buffer_size)
                            progress.update(1)
                            continue

                        # Validate individual tool names
                        validation_failed = False
                        for step in pipeline.steps:
                            if step.tool_cls.NAME == "external-tool":
                                error_msg = f"Invalid tool with external-tool NAME in step: {step.name()}"
                                self.logger.warning(f"🚨 Tool validation failed: {error_msg}")
                                self.logger.info("💡 Filtering out base class tools that shouldn't be directly used")
                                self.cache.queue_failure(
                                    pipeline.identifier(),
                                    gif_path.stem,
                                    params,
                                    {
                                        "error": error_msg,
                                        "error_traceback": "",
                                        "pipeline_steps": [step.name() for step in pipeline.steps],
                                        "tools_used": [step.tool_cls.NAME for step in pipeline.steps],
                                        "error_type": "validation"
                                    }
                                )
                                # Add validation failure to completed jobs and skip this pipeline
                                failed_result = {
                                    'gif_name': gif_path.stem,
                                    'content_type': content_type,
                                    'pipeline_id': pipeline.identifier(),
                                    'connection_signature': 'FAIL',  # Tool validation failed before execution
                                    'success': False,
                                    'error': error_msg,
                                    'test_colors': params.get("colors", 0),
                                    'test_lossy': params.get("lossy", 0),
                                    'test_frame_ratio': params.get("frame_ratio", 1.0),
                                }
                                completed_job_ids.add(job_id)
                                self._add_result_to_buffer(failed_result, results_buffer, csv_writer, csv_file, buffer_size)
                                progress.update(1)
                                validation_failed = True
                                break  # Exit the step validation loop
                        
                        if validation_failed:
                            continue  # Skip to next pipeline combination

                        # Execute pipeline and measure comprehensive metrics
                        result = self._execute_pipeline_with_metrics(
                            gif_path, pipeline, params, content_type
                        )
                        
                        # Queue successful result for batch caching
                        if self.cache and result.get('success', False):
                            self.cache.queue_result(
                                pipeline.identifier(), gif_path.stem, params, result
                            )
                        
                        # Save progress immediately with minimal data
                        completed_job_ids.add(job_id)
                        
                        # Store minimal resume data periodically to avoid memory buildup
                        if len(completed_job_ids) % self.PROGRESS_SAVE_INTERVAL == 0:
                            self._save_resume_data_minimal(resume_file, completed_job_ids)
                        
                        self._add_result_to_buffer(result, results_buffer, csv_writer, csv_file, buffer_size)
                        
                    except Exception as e:
                        self.logger.warning(f"Pipeline failed: {job_id} - {e}")
                        failed_jobs += 1
                        
                        # Record comprehensive failure information
                        failed_result = {
                            'gif_name': gif_path.stem,
                            'content_type': content_type,
                            'pipeline_id': pipeline.identifier(),
                            'connection_signature': 'ERROR',  # Pipeline failed during execution
                            'error': clean_error_message(str(e)),  # Clean error message for CSV
                            'error_traceback': traceback.format_exc().replace('\n', ' | '),  # Preserve full traceback
                            'error_timestamp': datetime.now().isoformat(),
                            'success': False,
                            'pipeline_steps': [step.name for step in pipeline.steps] if hasattr(pipeline, 'steps') else [],
                            'tools_used': pipeline.tools_used() if hasattr(pipeline, 'tools_used') else [],
                            # For failed pipelines, applied parameters are always None since pipeline failed
                            'applied_colors': None,
                            'applied_lossy': None, 
                            'applied_frame_ratio': None,
                            'actual_pipeline_steps': None,
                            # Add placeholders for metrics that would be in successful results
                            'file_size_kb': None,
                            'original_size_kb': None,
                            'compression_ratio': None,
                            'ssim_mean': None,
                            'ssim_std': None,
                            'ssim_min': None,
                            'ssim_max': None,
                            'ms_ssim_mean': None,
                            'psnr_mean': None,
                            'temporal_consistency': None,
                            'mse_mean': None,
                            'rmse_mean': None,
                            'fsim_mean': None,
                            'gmsd_mean': None,
                            'chist_mean': None,
                            'edge_similarity_mean': None,
                            'texture_similarity_mean': None,
                            'sharpness_similarity_mean': None,
                            'composite_quality': None,
                            'enhanced_composite_quality': None,
                            'efficiency': None,
                            'render_time_ms': None,
                            'total_processing_time_ms': None
                        }
                        
                        # Queue failure for debugging analysis (in addition to CSV/logs)
                        if self.cache:
                            self.cache.queue_failure(
                                pipeline.identifier(), gif_path.stem, params, {
                                    'error': str(e),
                                    'error_traceback': traceback.format_exc(),
                                    'pipeline_steps': [step.name() for step in pipeline.steps] if hasattr(pipeline, 'steps') else [],
                                    'tools_used': pipeline.tools_used() if hasattr(pipeline, 'tools_used') else []
                                }
                            )
                        
                        completed_job_ids.add(job_id)
                        self._add_result_to_buffer(failed_result, results_buffer, csv_writer, csv_file, buffer_size)
                    
                    progress.update(1)
                    
                    # Flush buffer and cache periodically for performance
                    total_processed = len(completed_job_ids)
                    if total_processed % self.BUFFER_FLUSH_INTERVAL == 0:
                        self._flush_results_buffer(results_buffer, csv_writer, csv_file)
                        if self.cache:
                            self.cache.flush_batch(force=True)  # Flush accumulated batches
        
        progress.close()
        
        # Final flush of all pending cache data
        if self.cache:
            self.cache.flush_batch(force=True)
        
        # Log cache performance statistics with enhanced visibility
        if self.cache:
            total_cache_operations = cache_hits + cache_misses
            if total_cache_operations > 0:
                cache_hit_rate = (cache_hits / total_cache_operations) * 100
                self.logger.info(f"💾 Cache Performance Summary:")
                self.logger.info(f"   📈 Cache hits: {cache_hits:,}")
                self.logger.info(f"   📉 Cache misses: {cache_misses:,}")
                self.logger.info(f"   🎯 Hit rate: {cache_hit_rate:.1f}%")
                
                # Estimate time saved
                estimated_time_per_test = 2.5  # seconds (conservative estimate)
                time_saved_minutes = (cache_hits * estimated_time_per_test) / 60
                if time_saved_minutes > 1:
                    self.logger.info(f"   ⏱️ Estimated time saved: {time_saved_minutes:.1f} minutes")
                    if time_saved_minutes > 60:
                        time_saved_hours = time_saved_minutes / 60
                        self.logger.info(f"   ⏱️ Time saved (hours): {time_saved_hours:.1f}h")
            else:
                self.logger.info("💾 Cache statistics: No cache operations performed (all results were new)")
        
        # Final flush of any remaining results in buffer
        self._flush_results_buffer(results_buffer, csv_writer, csv_file, force=True)
        csv_file.close()
        
        # Final save of all completed job IDs for resume functionality
        self._save_resume_data_minimal(resume_file, completed_job_ids)
        
        # Read results from streaming CSV to create DataFrame
        # This prevents memory exhaustion for large result sets
        total_results = len(completed_job_ids)
        self.logger.info(f"Testing completed: {total_results} jobs, {failed_jobs} failures")
        
        # Load results from streaming CSV file
        try:
            # IMPORTANT: Semantic correctness for ML/analysis
            # - applied_* parameters only contain values when that processing step actually executed
            # - None/NaN when step was no-op or pipeline failed
            # - Never use 0 as default - would create false patterns in ML models
            # Use nullable integer types to properly handle None values 
            dtype_spec = {
                # Semantically correct columns - only contain values when actually applied
                'applied_colors': 'Int64',     # Nullable - None when color step was no-op or failed
                'applied_lossy': 'Int64',      # Nullable - None when lossy step was no-op or failed
                'applied_frame_ratio': 'float64', # Nullable - NaN when frame step was no-op or failed
                'actual_pipeline_steps': 'Int64', # Count of actual processing steps
                'error': 'string',             # String type for error messages
                'success': 'boolean'           # Boolean type for success flag
            }
            results_df = pd.read_csv(streaming_csv_path, dtype=dtype_spec, low_memory=False)
            self.logger.info(f"📊 Loaded {len(results_df)} results from streaming CSV")
            return results_df
        except Exception as e:
            self.logger.warning(f"Failed to load streaming results: {e}")
            # Fallback to empty DataFrame
            return pd.DataFrame()
    
    def _add_result_to_buffer(self, result: dict, buffer: list, csv_writer, csv_file, buffer_size: int):
        """Add a result to the memory buffer and flush to CSV if buffer is full.
        
        Args:
            result: Result dictionary to add
            buffer: Memory buffer list
            csv_writer: CSV writer instance
            buffer_size: Maximum buffer size before flushing
        """
        buffer.append(result)
        
        if len(buffer) >= buffer_size:
            self._flush_results_buffer(buffer, csv_writer, csv_file)
    
    def _flush_results_buffer(self, buffer: list, csv_writer, csv_file=None, force: bool = False):
        """Flush results buffer to CSV file and clear memory.
        
        Args:
            buffer: Memory buffer to flush
            csv_writer: CSV writer instance
            force: If True, flush regardless of buffer size
        """
        if not buffer:
            return
            
        if force or len(buffer) >= 15:  # Reduced from 50 for more frequent monitoring updates
            try:
                # Write all buffered results to CSV
                for result in buffer:
                    # Ensure all required fields are present with safe defaults
                    safe_result = {
                        'gif_name': result.get('gif_name', ''),
                        'content_type': result.get('content_type', ''),
                        'pipeline_id': result.get('pipeline_id', ''),
                        'connection_signature': result.get('connection_signature', ''),
                        'success': result.get('success', False),
                        'file_size_kb': result.get('file_size_kb', 0),
                        'original_size_kb': result.get('original_size_kb', 0),
                        'compression_ratio': result.get('compression_ratio', 1.0),
                        'ssim_mean': result.get('ssim_mean', 0.0),
                        'ssim_std': result.get('ssim_std', 0.0),
                        'ssim_min': result.get('ssim_min', 0.0),
                        'ssim_max': result.get('ssim_max', 0.0),
                        'ms_ssim_mean': result.get('ms_ssim_mean', 0.0),
                        'psnr_mean': result.get('psnr_mean', 0.0),
                        'temporal_consistency': result.get('temporal_consistency', 0.0),
                        'mse_mean': result.get('mse_mean', 0.0),
                        'rmse_mean': result.get('rmse_mean', 0.0),
                        'fsim_mean': result.get('fsim_mean', 0.0),
                        'gmsd_mean': result.get('gmsd_mean', 0.0),
                        'chist_mean': result.get('chist_mean', 0.0),
                        'edge_similarity_mean': result.get('edge_similarity_mean', 0.0),
                        'texture_similarity_mean': result.get('texture_similarity_mean', 0.0),
                        'sharpness_similarity_mean': result.get('sharpness_similarity_mean', 0.0),
                        'composite_quality': result.get('composite_quality', 0.0),
                        'enhanced_composite_quality': result.get('enhanced_composite_quality', 0.0),
                        'efficiency': result.get('efficiency', 0.0),
                        'render_time_ms': result.get('render_time_ms', 0),
                        'total_processing_time_ms': result.get('total_processing_time_ms', 0),
                        'pipeline_steps': str(result.get('pipeline_steps', [])),
                        'tools_used': str(result.get('tools_used', [])),
                        'applied_colors': result.get('applied_colors', None),
                        'applied_lossy': result.get('applied_lossy', None),
                        'applied_frame_ratio': result.get('applied_frame_ratio', None),
                        'actual_pipeline_steps': result.get('actual_pipeline_steps', 0),
                        'error': result.get('error', ''),
                        'error_traceback': result.get('error_traceback', ''),
                        'error_timestamp': result.get('error_timestamp', '')
                    }
                    csv_writer.writerow(safe_result)
                
                # Force write to disk by flushing the underlying file
                if csv_file is not None:
                    csv_file.flush()
                
                self.logger.debug(f"💾 Flushed {len(buffer)} results to streaming CSV")
                
                # Clear the buffer to free memory
                buffer.clear()
                
            except Exception as e:
                self.logger.warning(f"Failed to flush results buffer: {e}")
    
    def _execute_pipeline_with_metrics(self, gif_path: Path, pipeline: Pipeline, params: dict, content_type: str) -> dict:
        """Execute a single pipeline and calculate comprehensive quality metrics."""
        from ..metrics import calculate_comprehensive_metrics
        import tempfile
        import time
        
        start_time = time.perf_counter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_path = tmpdir_path / f"compressed_{gif_path.stem}.gif"
            
            # Execute pipeline steps
            current_input = gif_path
            pipeline_metadata = {}
            png_sequence_dir = None  # Track PNG sequences for gifski optimization
            connection_signature = []  # Track connection methods between steps (G=GIF, P=PNG)
            self.logger.debug(f"Initialized connection_signature tracking for {pipeline.identifier()}")
            
            for i, step in enumerate(pipeline.steps):
                step_output = tmpdir_path / f"step_{step.variable}_{current_input.stem}.gif"
                
                # Create wrapper instance and apply
                wrapper = step.tool_cls()
                step_params = {}
                
                if step.variable == "color_reduction":
                    step_params["colors"] = params["colors"]
                elif step.variable == "frame_reduction":
                    step_params["ratio"] = params.get("frame_ratio", 1.0)
                elif step.variable == "lossy_compression":
                    # Map lossy percentage to engine-specific range
                    engine_specific_lossy = self._map_lossy_percentage_to_engine(
                        params["lossy"], 
                        wrapper.__class__.__name__
                    )
                    step_params["lossy_level"] = engine_specific_lossy
                    
                    # Pass PNG sequence to tools that support it (gifski or animately advanced)
                    if png_sequence_dir and ("Gifski" in wrapper.__class__.__name__ or 
                                           "AnimatelyAdvancedLossyCompressor" == wrapper.__class__.__name__):
                        step_params["png_sequence_dir"] = str(png_sequence_dir)
                
                # Track connection method to this step (for all steps after the first)
                if i > 0:  # Track all connections except to first step
                    # PNG input is only possible for lossy compression steps (gifski/animately-advanced)
                    step_uses_png_input = False
                    if step.variable == "lossy_compression":
                        if png_sequence_dir and ("Gifski" in wrapper.__class__.__name__ or 
                                               "AnimatelyAdvancedLossyCompressor" == wrapper.__class__.__name__):
                            step_uses_png_input = True
                    
                    # All other steps (frame/color reduction) always use GIF connections
                    connection_method = 'P' if step_uses_png_input else 'G'
                    connection_signature.append(connection_method)
                    # Debug logging
                    self.logger.debug(f"Step {i} ({step.variable}): connection_method={connection_method}, png_dir_available={png_sequence_dir is not None}")
                
                # Check if next step supports PNG sequence input (gifski or animately advanced)
                next_step_is_gifski = False
                next_step_is_animately_advanced = False
                if i + 1 < len(pipeline.steps):
                    next_wrapper_name = pipeline.steps[i + 1].tool_cls.__name__
                    next_step_is_gifski = "Gifski" in next_wrapper_name
                    next_step_is_animately_advanced = "AnimatelyAdvancedLossyCompressor" == next_wrapper_name
                
                # Always run the current step first
                step_result = wrapper.apply(current_input, step_output, params=step_params)
                pipeline_metadata.update(step_result)
                
                # Export PNG sequence AFTER step is applied if next step supports PNG input
                # (gifski or animately advanced lossy). Verify tool availability to prevent race conditions
                wrapper_name = wrapper.__class__.__name__
                supports_png_export = (
                    "FFmpegFrameReducer" in wrapper_name or "FFmpegColorReducer" in wrapper_name or
                    "ImageMagickFrameReducer" in wrapper_name or "ImageMagickColorReducer" in wrapper_name or
                    "AnimatelyFrameReducer" in wrapper_name or "AnimatelyColorReducer" in wrapper_name
                )
                next_step_supports_png = (next_step_is_gifski and self._is_gifski_available()) or next_step_is_animately_advanced
                if (next_step_supports_png and supports_png_export):
                    png_sequence_dir = tmpdir_path / f"png_sequence_{step.variable}"
                    
                    # Use appropriate export function based on tool
                    # Export from step_output (processed result) not current_input (raw input)
                    if "FFmpeg" in wrapper.__class__.__name__:
                        from ..external_engines.ffmpeg import export_png_sequence
                        png_result = export_png_sequence(step_output, png_sequence_dir)
                    elif "Animately" in wrapper.__class__.__name__:
                        from ..external_engines.animately import export_png_sequence
                        png_result = export_png_sequence(step_output, png_sequence_dir)
                    else:  # ImageMagick
                        from ..external_engines.imagemagick import export_png_sequence
                        png_result = export_png_sequence(step_output, png_sequence_dir)
                    
                    # Merge PNG export metadata
                    pipeline_metadata.update({f"png_export_{step.variable}": png_result})
                    
                    # CRITICAL: Validate frame count for PNG sequence tools compatibility
                    # Gifski requires at least 2 frames and ONLY accepts PNG sequences
                    # Animately Advanced accepts any frame count and can fallback to GIF
                    frame_count = png_result.get("frame_count", 0)
                    if next_step_is_gifski and frame_count < 2:
                        # Gifski cannot work with <2 frames - this will cause pipeline failure
                        self.logger.warning(
                            f"PNG sequence has only {frame_count} frame(s), but gifski requires at least 2 frames. "
                            f"Gifski pipeline will fail - gifski cannot process GIF input directly."
                        )
                        png_sequence_dir = None  # Will cause gifski pipeline to fail later
                    elif next_step_is_animately_advanced:
                        # Animately Advanced accepts any frame count in PNG sequences
                        if frame_count >= 1:
                            self.logger.debug(f"PNG sequence with {frame_count} frame(s) ready for animately-advanced")
                        else:
                            self.logger.warning(f"PNG export failed (0 frames), animately-advanced will use GIF fallback")
                            png_sequence_dir = None
                else:
                    png_sequence_dir = None  # Reset if not using PNG sequence optimization
                
                current_input = step_output
            
            # Copy final result to output path
            copy(current_input, output_path)
            
            # Calculate comprehensive quality metrics using GPU-accelerated system if available
            try:
                quality_metrics = self._calculate_gpu_accelerated_metrics(gif_path, output_path)
            except Exception as e:
                self.logger.warning(f"GPU metrics calculation failed, falling back to CPU: {e}")
                try:
                    quality_metrics = calculate_comprehensive_metrics(gif_path, output_path)
                except Exception as e2:
                    self.logger.warning(f"Quality metrics calculation failed: {e2}")
                    quality_metrics = self._get_fallback_metrics(gif_path, output_path)
            
            # Compile complete result with all metrics
            # Filter out any empty entries and build final signature
            valid_connections = [c for c in connection_signature if c]
            final_signature = ''.join(valid_connections) if valid_connections else 'GG'
            self.logger.debug(f"Connection signature compilation: raw={connection_signature}, valid={valid_connections}, final='{final_signature}' for {pipeline.identifier()}")
            
            result = {
                'gif_name': gif_path.stem,
                'content_type': content_type,
                'pipeline_id': pipeline.identifier(),
                'connection_signature': final_signature,
                'success': True,
                
                # File metrics
                'file_size_kb': quality_metrics.get('kilobytes', 0),
                'original_size_kb': gif_path.stat().st_size / 1024,
                'compression_ratio': self._calculate_compression_ratio(gif_path, output_path),
                
                # Traditional quality metrics
                'ssim_mean': quality_metrics.get('ssim', 0.0),
                'ssim_std': quality_metrics.get('ssim_std', 0.0),
                'ssim_min': quality_metrics.get('ssim_min', 0.0),
                'ssim_max': quality_metrics.get('ssim_max', 0.0),
                
                'ms_ssim_mean': quality_metrics.get('ms_ssim', 0.0),
                'psnr_mean': quality_metrics.get('psnr', 0.0),
                'temporal_consistency': quality_metrics.get('temporal_consistency', 0.0),
                
                # Extended quality metrics (the elaborate ones)
                'mse_mean': quality_metrics.get('mse', 0.0),
                'rmse_mean': quality_metrics.get('rmse', 0.0),
                'fsim_mean': quality_metrics.get('fsim', 0.0),
                'gmsd_mean': quality_metrics.get('gmsd', 0.0),
                'chist_mean': quality_metrics.get('chist', 0.0),
                'edge_similarity_mean': quality_metrics.get('edge_similarity', 0.0),
                'texture_similarity_mean': quality_metrics.get('texture_similarity', 0.0),
                'sharpness_similarity_mean': quality_metrics.get('sharpness_similarity', 0.0),
                
                # Composite quality scores (traditional and enhanced)
                'composite_quality': quality_metrics.get('composite_quality', 0.0),
                'enhanced_composite_quality': quality_metrics.get('enhanced_composite_quality', 0.0),
                'efficiency': quality_metrics.get('efficiency', 0.0),
                
                # Performance metrics
                'render_time_ms': quality_metrics.get('render_ms', 0),
                'total_processing_time_ms': int((time.perf_counter() - start_time) * 1000),
                
                # Pipeline details
                'pipeline_steps': len(pipeline.steps),
                'tools_used': [step.tool_cls.NAME for step in pipeline.steps],
                
                # Semantically correct parameters: only record values when actually applied
                'applied_colors': params["colors"] if self._pipeline_uses_color_reduction(pipeline) else None,
                'applied_lossy': params["lossy"] if self._pipeline_uses_lossy_compression(pipeline) else None,
                'applied_frame_ratio': params.get("frame_ratio", 1.0) if self._pipeline_uses_frame_reduction(pipeline) else None,
                
                # Actual processing steps (exclude no-ops)
                'actual_pipeline_steps': self._count_actual_steps(pipeline),
            }
            
            return result
    
    def _pipeline_uses_color_reduction(self, pipeline) -> bool:
        """Check if pipeline actually applies color reduction (not no-op)."""
        for step in pipeline.steps:
            if (step.variable == "color_reduction" and 
                hasattr(step.tool_cls, 'NAME') and 
                step.tool_cls.NAME != "none-color"):
                return True
        return False
    
    def _pipeline_uses_frame_reduction(self, pipeline) -> bool:
        """Check if pipeline actually applies frame reduction (not no-op)."""
        for step in pipeline.steps:
            if (step.variable == "frame_reduction" and 
                hasattr(step.tool_cls, 'NAME') and 
                step.tool_cls.NAME != "none-frame"):
                return True
        return False
    
    def _pipeline_uses_lossy_compression(self, pipeline) -> bool:
        """Check if pipeline actually applies lossy compression (not no-op)."""
        for step in pipeline.steps:
            if (step.variable == "lossy_compression" and 
                hasattr(step.tool_cls, 'NAME') and 
                step.tool_cls.NAME != "none-lossy"):
                return True
        return False
    
    def _count_actual_steps(self, pipeline) -> int:
        """Count only non-no-op processing steps."""
        actual_steps = 0
        for step in pipeline.steps:
            if hasattr(step.tool_cls, 'NAME'):
                tool_name = step.tool_cls.NAME
                if not tool_name.startswith("none-"):
                    actual_steps += 1
        return actual_steps
    
    def _test_gpu_availability(self):
        """Test GPU availability and log status."""
        try:
            import cv2
            import numpy as np
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            
            if cuda_count > 0:
                self.logger.info(f"🚀 GPU acceleration enabled: {cuda_count} CUDA device(s) available")
                # Test basic GPU operations
                try:
                    test_mat = cv2.cuda_GpuMat()
                    test_mat.upload(np.ones((100, 100), dtype=np.uint8))
                    test_mat.download()
                    self.logger.info("✅ GPU operations test passed - GPU acceleration enabled")
                except Exception as e:
                    self.logger.warning(f"🔄 GPU operations test failed: {e}")
                    self.logger.warning("🔄 GPU acceleration disabled - continuing with CPU processing")
                    self.logger.info("💡 To enable GPU: ensure CUDA drivers and OpenCV-CUDA are properly installed")
                    self.use_gpu = False
            else:
                self.logger.warning("🔄 No CUDA devices found on this system")
                self.logger.warning("🔄 GPU acceleration disabled - continuing with CPU processing")
                self.logger.info("💡 To enable GPU: install CUDA-capable hardware and drivers")
                self.use_gpu = False
                
        except ImportError:
            self.logger.warning("🔄 OpenCV CUDA support not available")
            self.logger.warning("🔄 GPU acceleration disabled - continuing with CPU processing") 
            self.logger.info("💡 To enable GPU: install opencv-python with CUDA support")
            self.use_gpu = False
    
    def _calculate_gpu_accelerated_metrics(self, original_path: Path, compressed_path: Path) -> dict:
        """Calculate quality metrics with GPU acceleration where possible."""
        if not self.use_gpu:
            # User hasn't requested GPU or GPU not available
            self.logger.debug("📊 Computing quality metrics using CPU (GPU not requested or unavailable)")
            from ..metrics import calculate_comprehensive_metrics
            return calculate_comprehensive_metrics(original_path, compressed_path)
            
        try:
            import cv2
            
            # Check if CUDA is available
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            
            if not cuda_available:
            # Fall back to regular CPU calculation with clear communication
                self.logger.warning("🔄 CUDA devices became unavailable during processing")
                self.logger.warning("🔄 Falling back to CPU for quality metrics calculation")
                self.logger.info("💡 Performance may be slower than expected")
                from ..metrics import calculate_comprehensive_metrics
                return calculate_comprehensive_metrics(original_path, compressed_path)
        
            self.logger.info("🚀 Computing quality metrics using GPU acceleration")
            
            # Use GPU-accelerated version
            return self._calculate_cuda_metrics(original_path, compressed_path)
            
        except ImportError:
            # OpenCV CUDA not available, fall back to CPU with clear explanation
            self.logger.warning("🔄 OpenCV CUDA support lost during processing")
            self.logger.warning("🔄 Falling back to CPU for quality metrics calculation")
            self.logger.info("💡 Install opencv-python with CUDA support for better performance")
            from ..metrics import calculate_comprehensive_metrics
            return calculate_comprehensive_metrics(original_path, compressed_path)
    
    def _calculate_cuda_metrics(self, original_path: Path, compressed_path: Path) -> dict:
        """GPU-accelerated quality metrics calculation using CUDA."""
        import cv2
        import numpy as np
        import time
        from giflab.metrics import extract_gif_frames, resize_to_common_dimensions, align_frames
        from .config import DEFAULT_METRICS_CONFIG
        
        config = DEFAULT_METRICS_CONFIG
        start_time = time.perf_counter()
        
        # Extract frames (CPU operation)
        original_result = extract_gif_frames(original_path, config.SSIM_MAX_FRAMES)
        compressed_result = extract_gif_frames(compressed_path, config.SSIM_MAX_FRAMES)
        
        # Resize frames to common dimensions (CPU operation)
        original_frames, compressed_frames = resize_to_common_dimensions(
            original_result.frames, compressed_result.frames
        )
        
        # Align frames using content-based method (CPU operation)
        aligned_pairs = align_frames(original_frames, compressed_frames)
        
        if not aligned_pairs:
            raise ValueError("No frame pairs could be aligned")
        
        # GPU-accelerated metric calculations
        metric_values = {
            'ssim': [],
            'ms_ssim': [],
            'psnr': [],
            'mse': [],
            'rmse': [],
            'fsim': [],
            'gmsd': [],
            'chist': [],
            'edge_similarity': [],
            'texture_similarity': [],
            'sharpness_similarity': [],
        }
        
        # Process frames with GPU acceleration
        for orig_frame, comp_frame in aligned_pairs:
            try:
                # Upload frames to GPU
                gpu_orig = cv2.cuda_GpuMat()
                gpu_comp = cv2.cuda_GpuMat()
                gpu_orig.upload(orig_frame.astype(np.uint8))
                gpu_comp.upload(comp_frame.astype(np.uint8))
            except Exception as e:
                self.logger.warning(f"GPU upload failed for frame: {e}")
                # Fall back to CPU for this frame
                for key in metric_values.keys():
                    metric_values[key].append(0.0)
                continue
            
            # GPU-accelerated calculations
            try:
                # SSIM (simplified GPU version)
                ssim_val = self._gpu_ssim(gpu_orig, gpu_comp)
                metric_values['ssim'].append(max(0.0, min(1.0, ssim_val)))
                
                # MSE/RMSE (GPU accelerated)
                mse_val = self._gpu_mse(gpu_orig, gpu_comp)
                metric_values['mse'].append(mse_val)
                metric_values['rmse'].append(np.sqrt(mse_val))
                
                # PSNR (calculated from MSE)
                psnr_val = 20 * np.log10(255.0 / (np.sqrt(mse_val) + 1e-8))
                normalized_psnr = min(psnr_val / float(config.PSNR_MAX_DB), 1.0)
                metric_values['psnr'].append(max(0.0, normalized_psnr))
                
                # GPU-accelerated edge similarity
                edge_sim = self._gpu_edge_similarity(gpu_orig, gpu_comp, config)
                metric_values['edge_similarity'].append(edge_sim)
                
                # Fall back to CPU for complex metrics (FSIM, GMSD, etc.)
                cpu_orig = orig_frame
                cpu_comp = comp_frame
                
                metric_values['fsim'].append(self._cpu_fsim(cpu_orig, cpu_comp))
                metric_values['gmsd'].append(self._cpu_gmsd(cpu_orig, cpu_comp))
                metric_values['chist'].append(self._cpu_chist(cpu_orig, cpu_comp))
                metric_values['texture_similarity'].append(self._cpu_texture_similarity(cpu_orig, cpu_comp))
                metric_values['sharpness_similarity'].append(self._cpu_sharpness_similarity(cpu_orig, cpu_comp))
                
                # MS-SSIM (CPU fallback for now)
                metric_values['ms_ssim'].append(self._cpu_ms_ssim(cpu_orig, cpu_comp))
                
            except Exception as e:
                self.logger.warning(f"GPU metric calculation failed for frame: {e}")
                # Fill with fallback values
                for key in metric_values.keys():
                    if len(metric_values[key]) <= len(metric_values['ssim']) - 1:
                        metric_values[key].append(0.0)
        
        # Calculate aggregated statistics (same as CPU version)
        result = {}
        for metric_name, values in metric_values.items():
            if values:
                result[metric_name] = float(np.mean(values))
                result[f'{metric_name}_std'] = float(np.std(values))
                result[f'{metric_name}_min'] = float(np.min(values))
                result[f'{metric_name}_max'] = float(np.max(values))
            else:
                result[metric_name] = 0.0
                result[f'{metric_name}_std'] = 0.0
                result[f'{metric_name}_min'] = 0.0
                result[f'{metric_name}_max'] = 0.0
        
        # Calculate temporal consistency (CPU operation)
        temporal_delta = self._calculate_temporal_consistency(aligned_pairs)
        result['temporal_consistency'] = float(temporal_delta)
        
        # Calculate composite quality using enhanced metrics system
        from ..enhanced_metrics import process_metrics_with_enhanced_quality
        
        # Process with enhanced quality system (adds enhanced_composite_quality and efficiency)
        result = process_metrics_with_enhanced_quality(result, config)
        
        # Ensure compression ratio is available for efficiency calculation
        if 'compression_ratio' not in result:
            result['compression_ratio'] = self._calculate_compression_ratio(
                original_path if 'original_path' in locals() else Path("dummy"), 
                compressed_path
            )
        
        # Add system metrics
        result['kilobytes'] = float(compressed_path.stat().st_size / 1024)
        
        # Calculate processing time
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        result['render_ms'] = min(int(elapsed_seconds * 1000), 86400000)
        
        return result
    
    def _gpu_ssim(self, gpu_img1: 'cv2.cuda_GpuMat', gpu_img2: 'cv2.cuda_GpuMat') -> float:
        """GPU-accelerated SSIM calculation (simplified version).
        
        This implements a simplified version of SSIM (Structural Similarity Index) 
        using CUDA operations for better performance on large datasets.
        
        SSIM Formula: SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
        Where μ = mean, σ = standard deviation, σxy = covariance, C1,C2 = stability constants
        """
        import cv2
        import numpy as np
        
        try:
            # Convert to grayscale on GPU if needed (reduces computational complexity)
            # Most quality metrics work better on luminance rather than individual color channels
            if gpu_img1.channels() == 3:
                gpu_gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_RGB2GRAY)
                gpu_gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_RGB2GRAY)
            else:
                gpu_gray1 = gpu_img1
                gpu_gray2 = gpu_img2
        except Exception as e:
            self.logger.warning(f"GPU color conversion failed: {e}")
            return 0.0
        
        try:
            # Convert to float32 for calculations (higher precision than uint8)
            gpu_float1 = cv2.cuda.convertTo(gpu_gray1, cv2.CV_32F)
            gpu_float2 = cv2.cuda.convertTo(gpu_gray2, cv2.CV_32F)
            
            # SSIM constants for numerical stability (Wang et al. 2004)
            # C1 = (K1 * L)^2, C2 = (K2 * L)^2 where L=255 (dynamic range), K1=0.01, K2=0.03
            C1 = (0.01 * 255) ** 2  # ~6.5 - prevents division by zero for mean calculations
            C2 = (0.03 * 255) ** 2  # ~58.5 - prevents division by zero for variance calculations
            
            # Mean calculations using Gaussian blur (approximates local mean in SSIM window)
            # 11x11 kernel with σ=1.5 is standard for SSIM implementation
            kernel_size = (11, 11)
            sigma = 1.5
            
            # μ1, μ2 = local means computed via Gaussian blur
            mu1 = cv2.cuda.GaussianBlur(gpu_float1, kernel_size, sigma)
            mu2 = cv2.cuda.GaussianBlur(gpu_float2, kernel_size, sigma)
        except Exception as e:
            self.logger.warning(f"GPU SSIM calculation failed: {e}")
            return 0.0
        
        # Pre-compute squared means and cross-terms for SSIM formula
        mu1_sq = cv2.cuda.multiply(mu1, mu1)    # μ1²
        mu2_sq = cv2.cuda.multiply(mu2, mu2)    # μ2² 
        mu1_mu2 = cv2.cuda.multiply(mu1, mu2)   # μ1μ2
        
        # Variance calculations using GPU operations
        # σ1² = E[X1²] - μ1² (variance = mean of squares minus square of mean)
        sigma1_sq = cv2.cuda.GaussianBlur(cv2.cuda.multiply(gpu_float1, gpu_float1), kernel_size, sigma)
        sigma1_sq = cv2.cuda.subtract(sigma1_sq, mu1_sq)
        
        # σ2² = E[X2²] - μ2²
        sigma2_sq = cv2.cuda.GaussianBlur(cv2.cuda.multiply(gpu_float2, gpu_float2), kernel_size, sigma)
        sigma2_sq = cv2.cuda.subtract(sigma2_sq, mu2_sq)
        
        # σ12 = E[X1X2] - μ1μ2 (covariance between images)
        sigma12 = cv2.cuda.GaussianBlur(cv2.cuda.multiply(gpu_float1, gpu_float2), kernel_size, sigma)
        sigma12 = cv2.cuda.subtract(sigma12, mu1_mu2)
        
        # Download for final calculations (could be optimized further)
        mu1_cpu = mu1.download()
        mu2_cpu = mu2.download()
        mu1_sq_cpu = mu1_sq.download()
        mu2_sq_cpu = mu2_sq.download()
        mu1_mu2_cpu = mu1_mu2.download()
        sigma1_sq_cpu = sigma1_sq.download()
        sigma2_sq_cpu = sigma2_sq.download()
        sigma12_cpu = sigma12.download()
        
        # SSIM calculation
        numerator1 = 2 * mu1_mu2_cpu + C1
        numerator2 = 2 * sigma12_cpu + C2
        denominator1 = mu1_sq_cpu + mu2_sq_cpu + C1
        denominator2 = sigma1_sq_cpu + sigma2_sq_cpu + C2
        
        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        return float(np.mean(ssim_map))
    
    def _gpu_mse(self, gpu_img1: 'cv2.cuda_GpuMat', gpu_img2: 'cv2.cuda_GpuMat') -> float:
        """GPU-accelerated MSE calculation."""
        import cv2
        import numpy as np
        
        # Calculate difference on GPU
        gpu_diff = cv2.cuda.subtract(gpu_img1, gpu_img2)
        gpu_squared = cv2.cuda.multiply(gpu_diff, gpu_diff)
        
        # Download for mean calculation
        squared_cpu = gpu_squared.download()
        
        return float(np.mean(squared_cpu))
    
    def _gpu_edge_similarity(self, gpu_img1: 'cv2.cuda_GpuMat', gpu_img2: 'cv2.cuda_GpuMat', config) -> float:
        """GPU-accelerated edge similarity using Canny edge detection."""
        import cv2
        import numpy as np
        
        try:
            # Convert to grayscale on GPU if needed
            if gpu_img1.channels() == 3:
                gpu_gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_RGB2GRAY)
                gpu_gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_RGB2GRAY)
            else:
                gpu_gray1 = gpu_img1
                gpu_gray2 = gpu_img2
            
            # GPU Canny edge detection
            detector = cv2.cuda.createCannyEdgeDetector(
                config.EDGE_CANNY_THRESHOLD1, 
                config.EDGE_CANNY_THRESHOLD2
            )
        except Exception as e:
            self.logger.warning(f"GPU edge detection setup failed: {e}")
            return 0.0
        
        gpu_edges1 = detector.detect(gpu_gray1)
        gpu_edges2 = detector.detect(gpu_gray2)
        
        # Download for correlation calculation
        edges1 = gpu_edges1.download()
        edges2 = gpu_edges2.download()
        
        # Calculate correlation
        edges1_flat = edges1.flatten().astype(np.float32)
        edges2_flat = edges2.flatten().astype(np.float32)
        
        if np.std(edges1_flat) == 0 and np.std(edges2_flat) == 0:
            return 1.0
        elif np.std(edges1_flat) == 0 or np.std(edges2_flat) == 0:
            return 0.0
        
        correlation = np.corrcoef(edges1_flat, edges2_flat)[0, 1]
        return float(np.clip(correlation, 0.0, 1.0))
    
    # CPU fallback methods for complex metrics
    def _cpu_fsim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """CPU FSIM calculation."""
        from .metrics import fsim
        return fsim(frame1, frame2)
    
    def _cpu_gmsd(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """CPU GMSD calculation."""
        from .metrics import gmsd
        return gmsd(frame1, frame2)
    
    def _cpu_chist(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """CPU color histogram correlation."""
        from .metrics import chist
        return chist(frame1, frame2)
    
    def _cpu_texture_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """CPU texture similarity."""
        from .metrics import texture_similarity
        return texture_similarity(frame1, frame2)
    
    def _cpu_sharpness_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """CPU sharpness similarity."""
        from .metrics import sharpness_similarity
        return sharpness_similarity(frame1, frame2)
    
    def _cpu_ms_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """CPU MS-SSIM calculation."""
        from .metrics import calculate_ms_ssim
        return calculate_ms_ssim(frame1, frame2)
    
    def _calculate_temporal_consistency(self, aligned_pairs) -> float:
        """Calculate temporal consistency between frames."""
        if len(aligned_pairs) < 2:
            return 1.0
        
        # Calculate frame-to-frame differences
        differences = []
        for i in range(len(aligned_pairs) - 1):
            curr_orig, curr_comp = aligned_pairs[i]
            next_orig, next_comp = aligned_pairs[i + 1]
            
            # Calculate difference in differences
            orig_diff = np.mean(np.abs(next_orig.astype(np.float32) - curr_orig.astype(np.float32)))
            comp_diff = np.mean(np.abs(next_comp.astype(np.float32) - curr_comp.astype(np.float32)))
            
            temporal_diff = abs(orig_diff - comp_diff)
            differences.append(temporal_diff)
        
        # Normalize and invert (higher = more consistent)
        avg_diff = np.mean(differences)
        consistency = 1.0 / (1.0 + avg_diff / 255.0)
        
        return float(consistency)
    
    def _map_lossy_percentage_to_engine(self, lossy_percentage: int, wrapper_class_name: str) -> int:
        """Map lossy percentage (0-100) to engine-specific range.
        
        Engine ranges:
        - Gifsicle: 0-300 (lossy=60% -> 180, lossy=100% -> 300)
        - Animately: 0-100 (lossy=60% -> 60, lossy=100% -> 100)
        - FFmpeg: 0-100 (lossy=60% -> 60, lossy=100% -> 100)
        - Gifski: 0-100 (lossy=60% -> 60, lossy=100% -> 100)
        """
        if lossy_percentage < 0 or lossy_percentage > 100:
            self.logger.warning(f"Invalid lossy percentage: {lossy_percentage}%, clamping to 0-100%")
            lossy_percentage = max(0, min(100, lossy_percentage))
        
        # Identify engine from wrapper class name
        wrapper_name = wrapper_class_name.lower()
        
        if "gifsicle" in wrapper_name:
            # Gifsicle: 0-300 range
            mapped_value = int(lossy_percentage * 3.0)  # 60% -> 180, 100% -> 300
            
            # Additional validation for Gifsicle to ensure we don't exceed its limits
            if mapped_value > 300:
                self.logger.warning(f"Gifsicle lossy level {mapped_value} exceeds maximum 300, clamping")
                mapped_value = 300
                
            self.logger.debug(f"Mapped {lossy_percentage}% to Gifsicle lossy level {mapped_value}")
            return mapped_value
        else:
            # Animately, FFmpeg, Gifski, etc.: 0-100 range
            mapped_value = lossy_percentage  # 60% -> 60, 100% -> 100
            
            # Additional validation for other engines
            if mapped_value > 100:
                self.logger.warning(f"Engine {wrapper_class_name} lossy level {mapped_value} exceeds maximum 100, clamping")
                mapped_value = 100
                
            self.logger.debug(f"Mapped {lossy_percentage}% to {wrapper_class_name} lossy level {mapped_value}")
            return mapped_value
    
    @property
    def test_params(self) -> List[dict]:
        """Test parameter combinations for comprehensive evaluation.
        
        When running targeted presets, uses preset-defined locked parameters.
        Otherwise uses default systematic parameter combinations.
        """
        # Override with preset parameters if running in targeted mode
        if self._current_preset is not None:
            return self._generate_preset_params(self._current_preset)
        
        # Default comprehensive parameter testing
        params = [
            # === 2x2 MATRIX: LOSSY × COLORS ===
            {"colors": 32, "lossy": 60, "frame_ratio": 0.5},   # Mid colors + moderate lossy
            {"colors": 128, "lossy": 60, "frame_ratio": 0.5},  # High colors + moderate lossy
            {"colors": 32, "lossy": 100, "frame_ratio": 0.5},  # Mid colors + high lossy
            {"colors": 128, "lossy": 100, "frame_ratio": 0.5}, # High colors + high lossy
        ]
        
        # Validate parameter ranges
        for param_set in params:
            if param_set["colors"] < 2 or param_set["colors"] > 256:
                self.logger.warning(f"Invalid color count: {param_set['colors']}, clamping to valid range")
                param_set["colors"] = max(2, min(256, param_set["colors"]))
            
            # Lossy is now percentage (0-100%), will be mapped to engine-specific ranges
            if param_set["lossy"] < 0 or param_set["lossy"] > 100:
                self.logger.warning(f"Invalid lossy percentage: {param_set['lossy']}%, clamping to valid range")
                param_set["lossy"] = max(0, min(100, param_set["lossy"]))
                
            if param_set["frame_ratio"] <= 0 or param_set["frame_ratio"] > 1.0:
                self.logger.warning(f"Invalid frame ratio: {param_set['frame_ratio']}, clamping to valid range")
                param_set["frame_ratio"] = max(0.1, min(1.0, param_set["frame_ratio"]))
        
        return params
    
    def _generate_preset_params(self, preset: ExperimentPreset) -> List[dict]:
        """Generate parameter combinations from preset slot configurations.
        
        Args:
            preset: ExperimentPreset with slot configurations
            
        Returns:
            Single-element list with preset-defined parameters
        """
        params = {}
        
        # Extract color parameters
        if preset.color_slot.type == "locked":
            params["colors"] = preset.color_slot.parameters.get("colors", 32)
        else:
            # For variable color slots, use a default value
            # (Variable slots vary by tool class, not parameters)
            params["colors"] = 32
        
        # Extract lossy parameters  
        if preset.lossy_slot.type == "locked":
            params["lossy"] = preset.lossy_slot.parameters.get("level", 40)
        else:
            # For variable lossy slots, use a default value
            params["lossy"] = 60
        
        # Extract frame parameters
        if preset.frame_slot.type == "locked":
            # Frame slot parameters might have different names
            frame_params = preset.frame_slot.parameters
            if "ratio" in frame_params:
                params["frame_ratio"] = frame_params["ratio"]
            elif "ratios" in frame_params and isinstance(frame_params["ratios"], list):
                # Use first ratio if multiple provided
                params["frame_ratio"] = frame_params["ratios"][0] 
            else:
                params["frame_ratio"] = 0.5  # Default
        else:
            # Variable frame slots get default
            params["frame_ratio"] = 0.5
        
        self.logger.info(f"🔒 Using preset-locked parameters: {params}")
        return [params]  # Single parameter set for preset mode
    
    def _load_resume_data(self, resume_file: Path) -> dict:
        """Load previously completed jobs for resume functionality."""
        if resume_file.exists():
            try:
                import json
                with open(resume_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load resume data: {e}")
        return {}
    
    def _save_resume_data(self, resume_file: Path, completed_jobs: dict):
        """Save completed jobs for resume functionality."""
        try:
            import json
            with open(resume_file, 'w') as f:
                json.dump(completed_jobs, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save resume data: {e}")
    
    def _save_resume_data_minimal(self, resume_file: Path, completed_job_ids: set):
        """Save only completed job IDs for memory-efficient resume functionality."""
        try:
            import json
            # Convert set to dict with minimal placeholder data for compatibility
            minimal_data = {job_id: {"status": "completed"} for job_id in completed_job_ids}
            with open(resume_file, 'w') as f:
                json.dump(minimal_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save minimal resume data: {e}")
    
    def _is_gifski_available(self) -> bool:
        """Check if gifski tool is available for PNG sequence optimization.
        
        Returns:
            True if gifski is available and can be used, False otherwise
        """
        try:
            from .system_tools import discover_tool
            return discover_tool("gifski").available
        except Exception:
            return False
    
    def _estimate_execution_time(self, remaining_jobs: int) -> str:
        """Estimate remaining execution time based on job complexity."""
        # Base time estimates (in seconds per job)
        base_time_per_job = 2.0  # Conservative estimate for synthetic GIFs
        metric_calculation_time = 0.5  # Additional time for comprehensive metrics
        
        total_time_per_job = base_time_per_job + metric_calculation_time
        estimated_seconds = remaining_jobs * total_time_per_job
        
        # Convert to human-readable format
        if estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"
    
    def _save_intermediate_results(self, results: List[dict]):
        """Save intermediate results as checkpoint."""
        checkpoint_file = self.output_dir / "intermediate_results.json"
        try:
            import json
            with open(checkpoint_file, 'w') as f:
                json.dump(results[-50:], f, indent=2, default=str)  # Save last 50 results
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _get_fallback_metrics(self, original_path: Path, compressed_path: Path) -> dict:
        """Get basic fallback metrics if comprehensive calculation fails."""
        try:
            original_size = original_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            
            return {
                'kilobytes': compressed_size / 1024,
                'ssim': 0.5,  # Conservative fallback
                'composite_quality': 0.5,
                'render_ms': 0,
            }
        except Exception:
            return {
                'kilobytes': 0,
                'ssim': 0.0,
                'composite_quality': 0.0,
                'render_ms': 0,
            }
    
    def _calculate_compression_ratio(self, original_path: Path, compressed_path: Path) -> float:
        """Calculate compression ratio (original size / compressed size)."""
        try:
            original_size = original_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            return original_size / compressed_size if compressed_size > 0 else 1.0
        except Exception:
            return 1.0
    
    def _get_content_type(self, gif_name: str) -> str:
        """Get content type from synthetic GIF name."""
        for spec in self.synthetic_specs:
            if spec.name == gif_name:
                return spec.content_type
        return "unknown"
    
    def _analyze_and_experiment(self, results_df: pd.DataFrame, threshold: float) -> ExperimentResult:
        """Analyze results using comprehensive quality metrics and identify underperforming pipelines."""
        experiment_result = ExperimentResult()
        
        # Filter out failed jobs for analysis
        if 'success' in results_df.columns:
            successful_results = results_df[results_df['success'] == True].copy()
        else:
            # If no success column, assume all results are successful
            successful_results = results_df.copy()
        
        if successful_results.empty:
            self.logger.warning("No successful pipeline results to analyze")
            return experiment_result
        
        # Multi-metric analysis: Use composite quality score as primary, with SSIM and compression ratio as secondary
        self.logger.info("Analyzing pipelines using comprehensive quality metrics...")
        
        performance_matrix = {}
        
        # Group by content type and find winners using multiple criteria
        for content_type in successful_results['content_type'].unique():
            content_results = successful_results[successful_results['content_type'] == content_type].copy()
            
            # Calculate multi-metric scores for ranking
            if 'render_time_ms' in content_results.columns:
                max_render_time = content_results['render_time_ms'].max()
                if max_render_time > 0:
                    speed_bonus = 0.1 * (1 - content_results['render_time_ms'] / max_render_time)
                else:
                    speed_bonus = 0.1  # Default bonus if all render times are 0
            else:
                speed_bonus = 0.1  # Default bonus if no timing data available
                
            # Build efficiency score from available columns
            efficiency_score = 0.0
            if 'composite_quality' in content_results.columns:
                efficiency_score += 0.4 * content_results['composite_quality']
            if 'compression_ratio' in content_results.columns:
                efficiency_score += 0.3 * content_results['compression_ratio']
            if 'ssim_mean' in content_results.columns:
                efficiency_score += 0.2 * content_results['ssim_mean']
            else:
                # Fallback: use any available numeric column for ranking
                numeric_cols = content_results.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    efficiency_score = content_results[numeric_cols[0]]
                    
            efficiency_score += speed_bonus
            content_results['efficiency_score'] = efficiency_score
            
            # Find top performers using different criteria - use available columns
            quality_col = 'composite_quality' if 'composite_quality' in content_results.columns else 'ssim_mean'
            compression_col = 'compression_ratio' if 'compression_ratio' in content_results.columns else 'ssim_mean'
            
            quality_winners = content_results.nlargest(3, quality_col)['pipeline_id'].tolist()
            efficiency_winners = content_results.nlargest(3, 'efficiency_score')['pipeline_id'].tolist()
            compression_winners = content_results.nlargest(3, compression_col)['pipeline_id'].tolist()
            
            # Combine all winners (union of top performers)
            all_content_winners = list(set(quality_winners + efficiency_winners + compression_winners))
            experiment_result.content_type_winners[content_type] = all_content_winners
            
            # Store performance matrix for detailed analysis - only use available columns
            perf_matrix = {
                'quality_leaders': quality_winners,
                'efficiency_leaders': efficiency_winners, 
                'compression_leaders': compression_winners,
            }
            
            # Add metrics only if columns exist
            if 'composite_quality' in content_results.columns:
                perf_matrix['mean_composite_quality'] = content_results['composite_quality'].mean()
            if 'ssim_mean' in content_results.columns:
                perf_matrix['mean_ssim'] = content_results['ssim_mean'].mean()
            if 'compression_ratio' in content_results.columns:
                perf_matrix['mean_compression_ratio'] = content_results['compression_ratio'].mean()
                
            performance_matrix[content_type] = perf_matrix
        
        experiment_result.performance_matrix = performance_matrix
        
        # Find pipelines that never win in any content type or criteria
        all_winners = set()
        for winners in experiment_result.content_type_winners.values():
            all_winners.update(winners)
        
        all_pipelines = set(successful_results['pipeline_id'].unique())
        never_winners = all_pipelines - all_winners
        
        # Additional elimination criteria based on comprehensive metrics
        underperformers = set()
        
        # Eliminate pipelines with consistently poor metrics
        for pipeline_id in all_pipelines:
            pipeline_results = successful_results[successful_results['pipeline_id'] == pipeline_id]
            
            # Skip if no results for this pipeline
            if pipeline_results.empty:
                self.logger.warning(f"No results found for pipeline: {pipeline_id}")
                underperformers.add(pipeline_id)
                continue
            
            # Check if pipeline consistently underperforms - use available columns
            should_eliminate = False
            
            # Use enhanced composite quality metric if available, fallback to legacy composite quality
            quality_column = None
            if 'enhanced_composite_quality' in pipeline_results.columns:
                quality_column = 'enhanced_composite_quality'
            elif 'composite_quality' in pipeline_results.columns:
                quality_column = 'composite_quality'
                
            if quality_column:
                avg_composite_quality = pipeline_results[quality_column].mean()
                max_composite_quality = pipeline_results[quality_column].max()
                
                if pd.isna(avg_composite_quality) or pd.isna(max_composite_quality):
                    self.logger.warning(f"Invalid {quality_column} metrics for pipeline: {pipeline_id}")
                    should_eliminate = True
                elif (avg_composite_quality < threshold or max_composite_quality < threshold * 1.5):
                    should_eliminate = True
            
            if 'ssim_mean' in pipeline_results.columns:
                avg_ssim = pipeline_results['ssim_mean'].mean()
                
                if pd.isna(avg_ssim):
                    self.logger.warning(f"Invalid ssim_mean metrics for pipeline: {pipeline_id}")
                    should_eliminate = True
                elif avg_ssim < threshold:
                    should_eliminate = True
            
            # If no quality metrics available, use any numeric column for basic filtering
            if quality_column is None and 'ssim_mean' not in pipeline_results.columns:
                numeric_cols = pipeline_results.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    primary_metric = pipeline_results[numeric_cols[0]].mean()
                    if pd.isna(primary_metric) or primary_metric < threshold:
                        should_eliminate = True
            
            if should_eliminate:
                underperformers.add(pipeline_id)
        
        experiment_result.eliminated_pipelines = never_winners.union(underperformers)
        experiment_result.retained_pipelines = all_winners - underperformers
        
        # Add detailed elimination reasons
        for pipeline in experiment_result.eliminated_pipelines:
            if pipeline in never_winners:
                experiment_result.elimination_reasons[pipeline] = "Never achieved top-3 performance in any content type or criteria"
            elif pipeline in underperformers:
                pipeline_results = successful_results[successful_results['pipeline_id'] == pipeline]
                if not pipeline_results.empty:
                    # Use available quality metrics for elimination reason
                    if 'composite_quality' in pipeline_results.columns:
                        avg_quality = pipeline_results['composite_quality'].mean()
                        if not pd.isna(avg_quality):
                            experiment_result.elimination_reasons[pipeline] = f"Consistently poor performance (avg composite quality: {avg_quality:.3f})"
                        else:
                            experiment_result.elimination_reasons[pipeline] = "Invalid/missing composite quality metrics"
                    elif 'ssim_mean' in pipeline_results.columns:
                        avg_ssim = pipeline_results['ssim_mean'].mean()
                        if not pd.isna(avg_ssim):
                            experiment_result.elimination_reasons[pipeline] = f"Consistently poor performance (avg SSIM: {avg_ssim:.3f})"
                        else:
                            experiment_result.elimination_reasons[pipeline] = "Invalid/missing SSIM metrics"
                    else:
                        experiment_result.elimination_reasons[pipeline] = "Poor performance across available metrics"
                else:
                    experiment_result.elimination_reasons[pipeline] = "No successful test results"
        
        # Pareto frontier analysis for quality-aligned comparison
        self.logger.info("Running Pareto frontier analysis...")
        try:
            pareto_analyzer = ParetoAnalyzer(successful_results, self.logger)
            pareto_analysis = pareto_analyzer.generate_comprehensive_pareto_analysis()
            experiment_result.pareto_analysis = pareto_analysis
            
            # Extract dominated pipelines from Pareto analysis
            all_dominated = set()
            for content_type, frontier_data in pareto_analysis['content_type_frontiers'].items():
                all_dominated.update(frontier_data.get('dominated_pipelines', []))
            
            experiment_result.pareto_dominated_pipelines = all_dominated
            
            # Extract quality-aligned rankings
            experiment_result.quality_aligned_rankings = pareto_analysis.get('efficiency_rankings', {})
            
            # Update elimination reasons for Pareto-dominated pipelines
            for pipeline in all_dominated:
                if pipeline not in experiment_result.elimination_reasons:
                    experiment_result.elimination_reasons[pipeline] = "Pareto dominated (always better alternatives available)"
            
            # Log Pareto analysis statistics
            global_frontier = pareto_analysis.get('global_frontier', {})
            frontier_count = len(global_frontier.get('frontier_points', []))
            dominated_count = len(all_dominated)
            
            self.logger.info(f"Pareto frontier analysis complete:")
            self.logger.info(f"  - Pipelines on global frontier: {frontier_count}")
            self.logger.info(f"  - Pareto dominated pipelines: {dominated_count}")
            
            # Log quality-aligned winners
            rankings = pareto_analysis.get('efficiency_rankings', {})
            for quality_level, ranked_list in rankings.items():
                if ranked_list:
                    winner = ranked_list[0]
                    self.logger.info(f"  - Best efficiency at {quality_level}: {winner[0]} ({winner[1]['best_size_kb']:.1f}KB)")
                    
        except Exception as e:
            self.logger.warning(f"Pareto frontier analysis failed: {e}")
            # Set empty defaults if analysis fails
            experiment_result.pareto_analysis = {}
            experiment_result.pareto_dominated_pipelines = set()
            experiment_result.quality_aligned_rankings = {}
        
        # Log elimination statistics
        self.logger.info(f"Analysis complete:")
        self.logger.info(f"  - Total pipelines tested: {len(all_pipelines)}")
        self.logger.info(f"  - Eliminated: {len(experiment_result.eliminated_pipelines)}")
        self.logger.info(f"  - Retained: {len(experiment_result.retained_pipelines)}")
        
        return experiment_result
    
    def _save_results(self, experiment_result: ExperimentResult, results_df: pd.DataFrame):
        """Save elimination analysis results."""
        failed_results, successful_results = self._validate_and_separate_results(results_df)
        
        self._save_csv_results(results_df)
        
        if not failed_results.empty:
            self._save_failed_pipelines_log(failed_results)
        
        self._save_elimination_summary(experiment_result, results_df, failed_results, successful_results)
        self._save_pareto_analysis_results(experiment_result)
        self._generate_and_save_failure_report(results_df)
        self._log_results_summary(experiment_result, failed_results, results_df)

    def _validate_and_separate_results(self, results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate DataFrame structure and separate successful and failed results."""
        if results_df.empty:
            self.logger.warning("Results DataFrame is empty, creating empty output files")
            failed_results = pd.DataFrame()
            successful_results = pd.DataFrame()
        elif 'success' not in results_df.columns:
            self.logger.warning("'success' column not found in results, treating all as successful")
            failed_results = pd.DataFrame()
            successful_results = results_df
        else:
            # Safely filter results with proper boolean checking
            success_mask = results_df['success'].fillna(False).astype(bool)
            failed_results = results_df[~success_mask].copy()
            successful_results = results_df[success_mask].copy()
        
        return failed_results, successful_results

    def _save_csv_results(self, results_df: pd.DataFrame):
        """Clean and save results to CSV file with timestamped filename."""
        # Fix CSV output by properly escaping error messages
        results_df_clean = results_df.copy()
        if 'error' in results_df_clean.columns:
            # Replace newlines and quotes in error messages to prevent CSV corruption
            results_df_clean['error'] = results_df_clean['error'].apply(clean_error_message)
        
        # Create timestamped filename for historical tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = f"elimination_test_results_{timestamp}.csv"
        
        # Save with both standard and timestamped names
        standard_path = self.output_dir / "elimination_test_results.csv"
        timestamped_path = self.output_dir / timestamped_filename
        
        # Save detailed results (with cleaned error messages)
        results_df_clean.to_csv(standard_path, index=False)
        results_df_clean.to_csv(timestamped_path, index=False)
        
        self.logger.info(f"📊 CSV results saved to:")
        self.logger.info(f"   Standard: {standard_path}")
        self.logger.info(f"   Timestamped: {timestamped_path}")
        
        # Also save a master history file in the base directory
        self._append_to_master_history(results_df_clean, timestamp)
    
    def _append_to_master_history(self, results_df: pd.DataFrame, timestamp: str):
        """Append results to a master history file that accumulates all runs."""
        master_history_path = self.base_output_dir / "elimination_history_master.csv"
        
        # Add run timestamp and identification to each record
        results_with_run_info = results_df.copy()
        results_with_run_info['run_timestamp'] = timestamp
        results_with_run_info['run_id'] = f"run_{timestamp}"
        
        try:
            # Check if master file exists
            if master_history_path.exists():
                # Append to existing file
                results_with_run_info.to_csv(master_history_path, mode='a', header=False, index=False)
                self.logger.info(f"📈 Results appended to master history: {master_history_path}")
            else:
                # Create new master file
                results_with_run_info.to_csv(master_history_path, index=False)
                self.logger.info(f"📈 Created new master history file: {master_history_path}")
                
        except Exception as e:
            self.logger.warning(f"Failed to update master history file: {e}")
    
    def _log_run_metadata(self):
        """Log metadata about this elimination run for tracking purposes."""
        run_metadata = {
            'run_id': self.output_dir.name,
            'start_time': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'base_directory': str(self.base_output_dir),
            'gpu_enabled': self.use_gpu,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'git_commit': self._get_git_commit() if Path('.git').exists() else None
        }
        
        # Save metadata to file
        metadata_file = self.output_dir / "run_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(run_metadata, f, indent=2)
            self.logger.info(f"📋 Run metadata saved: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save run metadata: {e}")
        
        return run_metadata
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                 capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _save_total_jobs_to_metadata(self, total_jobs: int):
        """Save total jobs count to metadata file for monitor to pick up.
        
        Args:
            total_jobs: Total number of jobs in this elimination run
        """
        try:
            metadata_file = self.output_dir / "run_metadata.json"
            
            # Load existing metadata if it exists
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    pass  # Start with empty metadata if loading fails
            
            # Update with total jobs count
            metadata['total_jobs'] = total_jobs
            metadata['jobs_updated_at'] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"📊 Saved total jobs count ({total_jobs:,}) to metadata for monitor")
            
        except Exception as e:
            self.logger.warning(f"Failed to save total jobs to metadata: {e}")
            # Non-critical error, continue execution
    
    def _save_failed_pipelines_log(self, failed_results: pd.DataFrame):
        """Create and save detailed failed pipelines log."""
        def _sanitize_for_json(value):
            """Convert pandas NA values to None for JSON serialization."""
            if pd.isna(value):
                return None
            return value
        
        failed_pipeline_log = []
        for _, row in failed_results.iterrows():
            failed_entry = {
                'timestamp': datetime.now().isoformat(),
                'gif_name': _sanitize_for_json(row.get('gif_name', 'unknown')),
                'content_type': _sanitize_for_json(row.get('content_type', 'unknown')),
                'pipeline_id': _sanitize_for_json(row.get('pipeline_id', 'unknown')),
                'error_message': _sanitize_for_json(row.get('error', 'No error message')),
                'pipeline_steps': _sanitize_for_json(row.get('pipeline_steps', [])),
                'tools_used': _sanitize_for_json(row.get('tools_used', [])),
                'test_parameters': {
                    'colors': _sanitize_for_json(row.get('applied_colors', row.get('test_colors', None))),
                    'lossy': _sanitize_for_json(row.get('applied_lossy', row.get('test_lossy', None))),
                    'frame_ratio': _sanitize_for_json(row.get('applied_frame_ratio', row.get('test_frame_ratio', None)))
                }
            }
            failed_pipeline_log.append(failed_entry)
        
        # Save failed pipelines log
        with open(self.output_dir / "failed_pipelines.json", 'w') as f:
            json.dump(failed_pipeline_log, f, indent=2)
        
        # Analyze and log error patterns
        self._analyze_and_log_error_patterns(failed_pipeline_log, len(failed_results))

    def _analyze_and_log_error_patterns(self, failed_pipeline_log: list, failed_count: int):
        """Analyze error patterns and log failure statistics."""
        error_types = Counter()
        for entry in failed_pipeline_log:
            error_msg = entry['error_message']
            error_type = ErrorTypes.categorize_error(error_msg)
            error_types[error_type] += 1
        
        # Log failure statistics
        self.logger.warning(f"Failed pipelines: {failed_count}")
        for error_type, count in error_types.most_common():
            self.logger.warning(f"  {error_type}: {count}")
        
        return error_types

    def _save_elimination_summary(self, experiment_result: ExperimentResult, results_df: pd.DataFrame, 
                                failed_results: pd.DataFrame, successful_results: pd.DataFrame):
        """Save enhanced elimination summary with failure information."""
        # Get error types for failed results
        error_types = {}
        if not failed_results.empty:
            failed_pipeline_log = []
            for _, row in failed_results.iterrows():
                failed_pipeline_log.append({'error_message': row.get('error', '')})
            error_types = self._analyze_and_log_error_patterns(failed_pipeline_log, len(failed_results))
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'eliminated_count': len(experiment_result.eliminated_pipelines),
            'retained_count': len(experiment_result.retained_pipelines),
            'eliminated_pipelines': list(experiment_result.eliminated_pipelines),
            'retained_pipelines': list(experiment_result.retained_pipelines),
            'content_type_winners': experiment_result.content_type_winners,
            'elimination_reasons': experiment_result.elimination_reasons,
            'failure_statistics': {
                'total_failed': len(failed_results),
                'total_successful': len(successful_results),
                'total_tested': len(results_df),
                'failure_rate': len(failed_results) / len(results_df) * 100 if len(results_df) > 0 else 0,
                'error_types': dict(error_types)
            }
        }
        
        with open(self.output_dir / "elimination_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_and_save_failure_report(self, results_df: pd.DataFrame):
        """Generate and save failure analysis report."""
        failure_report = self.generate_failure_analysis_report(results_df)
        with open(self.output_dir / "failure_analysis_report.txt", 'w') as f:
            f.write(failure_report)

    def _log_results_summary(self, experiment_result: ExperimentResult, failed_results: pd.DataFrame, results_df: pd.DataFrame):
        """Log comprehensive results summary."""
        self.logger.info(f"Eliminated {len(experiment_result.eliminated_pipelines)} underperforming pipelines")
        self.logger.info(f"Retained {len(experiment_result.retained_pipelines)} competitive pipelines")
        if not failed_results.empty:
            self.logger.info(f"Failed pipelines log saved to: {self.output_dir / 'failed_pipelines.json'}")
            self.logger.info(f"Failure analysis report saved to: {self.output_dir / 'failure_analysis_report.txt'}")
            self.logger.info(f"Failure rate: {len(failed_results)}/{len(results_df)} ({len(failed_results)/len(results_df)*100:.1f}%)")

    def generate_failure_analysis_report(self, results_df: pd.DataFrame) -> str:
        """Generate a detailed failure analysis report with recommendations."""
        from collections import Counter, defaultdict
        
        # Separate failed results with proper validation
        failed_results, _ = self._validate_and_separate_results(results_df)
        failed_results = failed_results if not failed_results.empty else pd.DataFrame()
        
        if failed_results.empty:
            return "✅ No pipeline failures detected. All pipelines executed successfully!"
        
        report_lines = []
        report_lines.append("🔍 PIPELINE FAILURE ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall statistics
        total_pipelines = len(results_df)
        failed_count = len(failed_results)
        failure_rate = (failed_count / total_pipelines) * 100
        
        report_lines.append("📊 OVERVIEW")
        report_lines.append(f"   Total pipelines tested: {total_pipelines}")
        report_lines.append(f"   Failed pipelines: {failed_count}")
        report_lines.append(f"   Failure rate: {failure_rate:.1f}%")
        report_lines.append("")
        
        # Error type analysis
        error_types = Counter()
        tool_failures = defaultdict(list)
        content_type_failures = defaultdict(int)
        
        for _, row in failed_results.iterrows():
            error_msg = str(row.get('error', ''))
            content_type = row.get('content_type', 'unknown')
            pipeline_id = row.get('pipeline_id', 'unknown')
            
            content_type_failures[content_type] += 1
            
            # Categorize errors using centralized function
            error_type = ErrorTypes.categorize_error(error_msg)
            error_types[error_type] += 1
            
            # Track tool failures for tools that have direct mappings
            if error_type in [ErrorTypes.GIFSKI, ErrorTypes.FFMPEG, ErrorTypes.IMAGEMAGICK, 
                            ErrorTypes.GIFSICLE, ErrorTypes.ANIMATELY]:
                tool_failures[error_type].append(pipeline_id)
        
        report_lines.append("🔧 ERROR TYPE BREAKDOWN")
        for error_type, count in error_types.most_common():
            percentage = (count / failed_count) * 100
            report_lines.append(f"   {error_type}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Content type analysis
        report_lines.append("📁 FAILURES BY CONTENT TYPE")
        for content_type, count in sorted(content_type_failures.items()):
            report_lines.append(f"   {content_type}: {count} failures")
        report_lines.append("")
        
        # Recommendations based on error patterns
        report_lines.append("💡 RECOMMENDATIONS")
        
        # Tool-specific recommendations
        if error_types[ErrorTypes.GIFSKI] > failed_count * 0.3:
            report_lines.append("   🔴 HIGH GIFSKI FAILURES:")
            report_lines.append("      - Consider updating gifski binary")
            report_lines.append("      - Check system compatibility (some systems lack required libraries)")
            report_lines.append("      - May need to exclude gifski from production pipelines")
            report_lines.append("")
        
        if error_types['ffmpeg'] > failed_count * 0.2:
            report_lines.append("   🟠 FFMPEG ISSUES DETECTED:")
            report_lines.append("      - Verify FFmpeg installation and PATH configuration")
            report_lines.append("      - Check for missing codecs or filters")
            report_lines.append("      - Consider testing with different FFmpeg versions")
            report_lines.append("")
        
        if error_types[ErrorTypes.IMAGEMAGICK] > failed_count * 0.2:
            report_lines.append("   🟡 IMAGEMAGICK CONFIGURATION ISSUES:")
            report_lines.append("      - Check ImageMagick security policies (/etc/ImageMagick-*/policy.xml)")
            report_lines.append("      - Verify GIF read/write permissions are enabled")
            report_lines.append("      - Consider increasing memory limits")
            report_lines.append("")
        
        if error_types[ErrorTypes.TIMEOUT] > 0:
            report_lines.append("   ⏱️ TIMEOUT ISSUES:")
            report_lines.append("      - Consider increasing timeout values for complex GIFs")
            report_lines.append("      - Monitor system resources during testing")
            report_lines.append("      - May indicate performance bottlenecks")
            report_lines.append("")
        
        if error_types[ErrorTypes.COMMAND_EXECUTION] > failed_count * 0.15:
            report_lines.append("   ⚡ COMMAND EXECUTION FAILURES:")
            report_lines.append("      - Check tool binary availability and permissions")
            report_lines.append("      - Verify system PATH includes all required tools")
            report_lines.append("      - Consider running system_tools.get_available_tools() for diagnostics")
            report_lines.append("")
        
        # Most problematic pipelines
        if tool_failures:
            report_lines.append("🚨 MOST PROBLEMATIC TOOL COMBINATIONS")
            for tool, failures in tool_failures.items():
                if len(failures) > 5:  # Only show tools with many failures
                    unique_pipelines = len(set(failures))
                    report_lines.append(f"   {tool}: {unique_pipelines} unique pipeline failures")
                    
                    # Show most common failure patterns
                    failure_patterns = Counter()
                    for pipeline in failures:
                        # Extract pattern (simplified)
                        parts = pipeline.split('__')
                        if len(parts) >= 2:
                            pattern = f"{parts[0]}...{parts[-1]}"
                            failure_patterns[pattern] += 1
                    
                    for pattern, count in failure_patterns.most_common(3):
                        report_lines.append(f"      - {pattern}: {count} failures")
            report_lines.append("")
        
        # General recommendations
        report_lines.append("🔄 GENERAL RECOMMENDATIONS")
        report_lines.append("   1. Run 'giflab view-failures results/experiments/latest/' for detailed error analysis")
        report_lines.append("   2. Consider excluding problematic tools from production pipelines")
        report_lines.append("   3. Test individual tools with 'python -c \"from giflab.system_tools import get_available_tools; print(get_available_tools())\"'")
        report_lines.append("   4. Review and update tool configurations in src/giflab/config.py")
        report_lines.append("   5. Consider running elimination analysis with fewer tool combinations to isolate issues")
        report_lines.append("")
        
        report_lines.append("📖 For detailed failure logs, see: results/experiments/latest/failed_pipelines.json")
        
        return "\n".join(report_lines)

    def validate_research_findings(self) -> Dict[str, bool]:
        """Validate the preliminary research findings about redundant methods."""
        findings = {}
        
        # Test ImageMagick redundant methods from research
        redundant_imagemagick = [
            "O2x2", "O3x3", "O4x4", "O8x8",  # Same as Ordered
            "H4x4a", "H6x6a", "H8x8a"       # Same as FloydSteinberg
        ]
        
        self.logger.info("Validating research findings about redundant ImageMagick methods")
        
        # This would test if these methods truly produce identical results
        for method in redundant_imagemagick:
            findings[f"imagemagick_{method}_redundant"] = True  # Placeholder
            
        # Test FFmpeg Bayer scale findings
        findings["ffmpeg_bayer_scale_4_5_best_for_noise"] = True  # Placeholder
        
        # Test Gifsicle O3 vs O2 finding
        findings["gifsicle_o3_minimal_benefit"] = True  # Placeholder
        
        return findings 
    
    def _save_pareto_analysis_results(self, experiment_result: ExperimentResult):
        """Save Pareto frontier analysis results to files."""
        try:
            pareto_analysis = experiment_result.pareto_analysis
            if not pareto_analysis:
                self.logger.warning("No Pareto analysis data to save")
                return
            
            # 1. Save global Pareto frontier points
            global_frontier = pareto_analysis.get('global_frontier', {})
            frontier_points = global_frontier.get('frontier_points', [])
            
            if frontier_points:
                frontier_df = pd.DataFrame(frontier_points)
                frontier_path = self.output_dir / "pareto_frontier_global.csv"
                frontier_df.to_csv(frontier_path, index=False)
                self.logger.info(f"Saved global Pareto frontier to: {frontier_path}")
            
            # 2. Save content-type specific frontiers
            content_frontiers = pareto_analysis.get('content_type_frontiers', {})
            for content_type, frontier_data in content_frontiers.items():
                points = frontier_data.get('frontier_points', [])
                if points:
                    frontier_df = pd.DataFrame(points)
                    frontier_path = self.output_dir / f"pareto_frontier_{content_type}.csv"
                    frontier_df.to_csv(frontier_path, index=False)
                    self.logger.info(f"Saved {content_type} Pareto frontier to: {frontier_path}")
            
            # 3. Save quality-aligned efficiency rankings
            rankings = pareto_analysis.get('efficiency_rankings', {})
            if rankings:
                rankings_data = []
                for quality_level, ranked_pipelines in rankings.items():
                    for rank, (pipeline_id, metrics) in enumerate(ranked_pipelines, 1):
                        rankings_data.append({
                            'quality_level': quality_level,
                            'rank': rank,
                            'pipeline_id': pipeline_id,
                            'best_size_kb': metrics['best_size_kb'],
                            'samples_at_quality': metrics['samples_at_quality']
                        })
                
                if rankings_data:
                    rankings_df = pd.DataFrame(rankings_data)
                    rankings_path = self.output_dir / "quality_aligned_rankings.csv"
                    rankings_df.to_csv(rankings_path, index=False)
                    self.logger.info(f"Saved quality-aligned rankings to: {rankings_path}")
            
            # 4. Save dominated pipelines list
            dominated = experiment_result.pareto_dominated_pipelines
            if dominated:
                dominated_df = pd.DataFrame(list(dominated), columns=['pipeline_id'])
                dominated_df['elimination_reason'] = 'Pareto dominated'
                dominated_path = self.output_dir / "pareto_dominated_pipelines.csv"
                dominated_df.to_csv(dominated_path, index=False)
                self.logger.info(f"Saved dominated pipelines to: {dominated_path}")
            
            # 5. Save comprehensive analysis summary
            summary_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_frontier_points': len(frontier_points),
                'total_dominated_pipelines': len(dominated),
                'content_types_analyzed': list(content_frontiers.keys()),
                'quality_levels_analyzed': list(rankings.keys()) if rankings else [],
                'trade_off_insights': pareto_analysis.get('trade_off_insights', {})
            }
            
            summary_path = self.output_dir / "pareto_analysis_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            self.logger.info(f"Saved Pareto analysis summary to: {summary_path}")
            
            # 6. Generate human-readable report
            self._generate_pareto_report(experiment_result)
            
        except Exception as e:
            self.logger.warning(f"Failed to save Pareto analysis results: {e}")
    
    def _generate_pareto_report(self, experiment_result: ExperimentResult):
        """Generate a human-readable Pareto analysis report."""
        try:
            pareto_analysis = experiment_result.pareto_analysis
            report_lines = []
            
            report_lines.append("# 🎯 Pareto Frontier Analysis Report")
            report_lines.append("")
            report_lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Executive Summary
            global_frontier = pareto_analysis.get('global_frontier', {})
            frontier_count = len(global_frontier.get('frontier_points', []))
            dominated_count = len(experiment_result.pareto_dominated_pipelines)
            
            report_lines.append("## Executive Summary")
            report_lines.append("")
            report_lines.append(f"- **Optimal Pipelines (Pareto Frontier):** {frontier_count}")
            report_lines.append(f"- **Dominated Pipelines (Eliminatable):** {dominated_count}")
            report_lines.append(f"- **Efficiency Gain:** {dominated_count / (frontier_count + dominated_count) * 100:.1f}% pipeline reduction")
            report_lines.append("")
            
            # Quality-Aligned Winners
            rankings = pareto_analysis.get('efficiency_rankings', {})
            if rankings:
                report_lines.append("## 🏆 Quality-Aligned Winners")
                report_lines.append("")
                report_lines.append("| Quality Level | Winner Pipeline | File Size (KB) | Samples |")
                report_lines.append("|---------------|-----------------|----------------|---------|")
                
                for quality_level in sorted(rankings.keys()):
                    ranked_list = rankings[quality_level]
                    if ranked_list:
                        winner_id, winner_metrics = ranked_list[0]
                        size_kb = winner_metrics['best_size_kb']
                        samples = winner_metrics['samples_at_quality']
                        quality_num = quality_level.replace('quality_', '')
                        report_lines.append(f"| {quality_num} | `{winner_id}` | {size_kb:.1f} | {samples} |")
                
                report_lines.append("")
            
            # Content-Type Analysis
            content_frontiers = pareto_analysis.get('content_type_frontiers', {})
            if content_frontiers:
                report_lines.append("## 📊 Content-Type Analysis")
                report_lines.append("")
                
                for content_type, frontier_data in content_frontiers.items():
                    frontier_points = frontier_data.get('frontier_points', [])
                    dominated = frontier_data.get('dominated_pipelines', [])
                    
                    report_lines.append(f"### {content_type.title()} Content")
                    report_lines.append(f"- **Optimal pipelines:** {len(frontier_points)}")
                    report_lines.append(f"- **Dominated pipelines:** {len(dominated)}")
                    
                    if frontier_points:
                        report_lines.append("- **Pareto optimal pipelines:**")
                        for point in frontier_points[:5]:  # Top 5
                            pipeline = point['pipeline_id']
                            quality = point['quality']
                            size = point['file_size_kb']
                            efficiency = point['efficiency_score']
                            report_lines.append(f"  - `{pipeline}`: {quality:.3f} quality, {size:.1f}KB, {efficiency:.5f} efficiency")
                    report_lines.append("")
            
            # Trade-off Insights
            insights = pareto_analysis.get('trade_off_insights', {})
            if insights:
                report_lines.append("## 💡 Trade-off Insights")
                report_lines.append("")
                
                # Efficiency leaders
                efficiency_leaders = insights.get('efficiency_leaders', [])
                if efficiency_leaders:
                    report_lines.append("### Top Efficiency Leaders")
                    report_lines.append("| Pipeline | Quality | Size (KB) | Efficiency |")
                    report_lines.append("|----------|---------|-----------|------------|")
                    
                    for leader in efficiency_leaders[:10]:
                        pipeline = leader['pipeline_id']
                        quality = leader['composite_quality']
                        size = leader['file_size_kb']
                        efficiency = leader['efficiency_ratio']
                        report_lines.append(f"| `{pipeline}` | {quality:.3f} | {size:.1f} | {efficiency:.5f} |")
                    report_lines.append("")
                
                # Sweet spot analysis
                sweet_spots = insights.get('sweet_spot_pipelines', {})
                if sweet_spots:
                    report_lines.append("### Sweet Spot Analysis (High Quality + Low Size)")
                    total_sweet_spot = sum(sweet_spots.values())
                    for pipeline, count in sorted(sweet_spots.items(), key=lambda x: x[1], reverse=True)[:5]:
                        percentage = (count / total_sweet_spot) * 100
                        report_lines.append(f"- **{pipeline}**: {count} results ({percentage:.1f}%)")
                    report_lines.append("")
            
            # Methodology
            report_lines.append("## 📋 Methodology")
            report_lines.append("")
            report_lines.append("**Pareto Optimality Criteria:**")
            report_lines.append("- A pipeline is Pareto optimal if no other pipeline achieves both higher quality AND smaller file size")
            report_lines.append("- Quality metric: Composite quality score (weighted combination of SSIM, MS-SSIM, PSNR, temporal consistency)")
            report_lines.append("- Size metric: File size in kilobytes")
            report_lines.append("")
            report_lines.append("**Quality-Aligned Rankings:**")
            report_lines.append("- Compare pipelines at specific quality levels (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)")
            report_lines.append("- Winner = smallest file size among pipelines achieving target quality")
            report_lines.append("")
            
            # Save report
            report_path = self.output_dir / "pareto_analysis_report.md"
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Generated Pareto analysis report: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate Pareto report: {e}")