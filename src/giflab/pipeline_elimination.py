"""Pipeline Elimination Framework

Systematically eliminates underperforming pipeline combinations through
competitive testing on synthetic GIFs with diverse characteristics.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

from .dynamic_pipeline import generate_all_pipelines, Pipeline
from .experiment import ExperimentalPipeline
from .metrics import calculate_comprehensive_metrics
from .meta import extract_gif_metadata


# Use enhanced error message cleaning from error_handling module
from .error_handling import clean_error_message


class ErrorTypes:
    """Constants for error type categorization."""
    GIFSKI = 'gifski'
    FFMPEG = 'ffmpeg'
    IMAGEMAGICK = 'imagemagick'
    GIFSICLE = 'gifsicle'
    ANIMATELY = 'animately'
    TIMEOUT = 'timeout'
    COMMAND_EXECUTION = 'command_execution'
    OTHER = 'other'
    
    @classmethod
    def all_types(cls) -> list[str]:
        """Return all error type constants."""
        return [cls.GIFSKI, cls.FFMPEG, cls.IMAGEMAGICK, cls.GIFSICLE, 
                cls.ANIMATELY, cls.TIMEOUT, cls.COMMAND_EXECUTION, cls.OTHER]
    
    @classmethod
    def categorize_error(cls, error_msg: str) -> str:
        """Categorize an error message into error type constants.
        
        Args:
            error_msg: Error message string to categorize
            
        Returns:
            Error type constant string
        """
        error_msg_lower = error_msg.lower()
        
        if cls.GIFSKI in error_msg_lower:
            return cls.GIFSKI
        elif cls.FFMPEG in error_msg_lower:
            return cls.FFMPEG
        elif cls.IMAGEMAGICK in error_msg_lower:
            return cls.IMAGEMAGICK
        elif cls.GIFSICLE in error_msg_lower:
            return cls.GIFSICLE
        elif cls.ANIMATELY in error_msg_lower:
            return cls.ANIMATELY
        elif 'command failed' in error_msg_lower:
            return cls.COMMAND_EXECUTION
        elif 'timeout' in error_msg_lower:
            return cls.TIMEOUT
        else:
            return cls.OTHER


class PipelineResultsCache:
    """SQLite-based cache for pipeline test results to avoid redundant testing."""
    
    def __init__(self, cache_db_path: Path, git_commit: Optional[str] = None):
        """Initialize the results cache.
        
        Args:
            cache_db_path: Path to SQLite database file
            git_commit: Current git commit hash for cache invalidation
        """
        self.cache_db_path = cache_db_path
        self.git_commit = git_commit or "unknown"
        self.logger = logging.getLogger(__name__)
        self._pending_results = []  # Batch storage
        self._pending_failures = []  # Batch failure storage
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database schema."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Results table (successful tests)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE NOT NULL,
                        pipeline_id TEXT NOT NULL,
                        gif_name TEXT NOT NULL,
                        test_colors INTEGER NOT NULL,
                        test_lossy INTEGER NOT NULL,
                        test_frame_ratio REAL NOT NULL,
                        git_commit TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        result_json TEXT NOT NULL
                    )
                """)
                
                # Failures table (for debugging and analysis)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_failures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pipeline_id TEXT NOT NULL,
                        gif_name TEXT NOT NULL,
                        test_colors INTEGER NOT NULL,
                        test_lossy INTEGER NOT NULL,
                        test_frame_ratio REAL NOT NULL,
                        git_commit TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        error_traceback TEXT,
                        pipeline_steps TEXT,
                        tools_used TEXT
                    )
                """)
                
                # Create indexes for faster lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                    ON pipeline_results(pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failure_lookup 
                    ON pipeline_failures(pipeline_id, error_type, created_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_failure_analysis 
                    ON pipeline_failures(error_type, git_commit, created_at)
                """)
                
                conn.commit()
                self.logger.debug(f"📁 Initialized cache database: {self.cache_db_path}")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize cache database: {e}")
    
    def _generate_cache_key(self, pipeline_id: str, gif_name: str, params: dict) -> str:
        """Generate a unique cache key for a pipeline test combination."""
        import hashlib
        
        # Create a deterministic key from all parameters
        key_data = f"{pipeline_id}|{gif_name}|{params['colors']}|{params['lossy']}|{params.get('frame_ratio', 1.0)}|{self.git_commit}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get_cached_result(self, pipeline_id: str, gif_name: str, params: dict) -> Optional[dict]:
        """Retrieve cached result if it exists and is valid.
        
        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name
            params: Test parameters dict
            
        Returns:
            Cached result dict or None if not found/invalid
        """
        cache_key = self._generate_cache_key(pipeline_id, gif_name, params)
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT result_json, git_commit, created_at 
                    FROM pipeline_results 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    result_json, cached_git_commit, created_at = row
                    
                    # Check if cache is still valid (same git commit)
                    if cached_git_commit == self.git_commit:
                        self.logger.debug(f"💾 Cache hit for {pipeline_id} on {gif_name}")
                        return json.loads(result_json)
                    else:
                        self.logger.debug(f"💾 Cache invalidated for {pipeline_id} on {gif_name} (git commit changed)")
                        # Could optionally delete invalidated entries here
                        return None
                else:
                    self.logger.debug(f"💾 Cache miss for {pipeline_id} on {gif_name}")
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Failed to retrieve cached result: {e}")
            return None
    
    def queue_result(self, pipeline_id: str, gif_name: str, params: dict, result: dict):
        """Queue a successful pipeline test result for batch storage.
        
        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name  
            params: Test parameters dict
            result: Test result dict to cache
        """
        cache_key = self._generate_cache_key(pipeline_id, gif_name, params)
        
        self._pending_results.append({
            'cache_key': cache_key,
            'pipeline_id': pipeline_id,
            'gif_name': gif_name,
            'test_colors': params['colors'],
            'test_lossy': params['lossy'],
            'test_frame_ratio': params.get('frame_ratio', 1.0),
            'git_commit': self.git_commit,
            'created_at': datetime.now().isoformat(),
            'result_json': json.dumps(result)
        })
        
        self.logger.debug(f"💾 Queued result for {pipeline_id} on {gif_name} (batch size: {len(self._pending_results)})")
    
    def queue_failure(self, pipeline_id: str, gif_name: str, params: dict, error_info: dict):
        """Queue a pipeline failure for batch storage and analysis.
        
        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name
            params: Test parameters dict
            error_info: Error information dict with keys: error, error_traceback, pipeline_steps, tools_used
        """
        error_type = ErrorTypes.categorize_error(error_info.get('error', ''))
        
        self._pending_failures.append({
            'pipeline_id': pipeline_id,
            'gif_name': gif_name,
            'test_colors': params['colors'],
            'test_lossy': params['lossy'],
            'test_frame_ratio': params.get('frame_ratio', 1.0),
            'git_commit': self.git_commit,
            'created_at': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_info.get('error', 'Unknown error'),
            'error_traceback': error_info.get('error_traceback', ''),
            'pipeline_steps': json.dumps(error_info.get('pipeline_steps', [])),
            'tools_used': json.dumps(error_info.get('tools_used', []))
        })
        
        self.logger.debug(f"💾 Queued failure for {pipeline_id} on {gif_name} (batch size: {len(self._pending_failures)})")
    
    def flush_batch(self, force: bool = False):
        """Flush pending results and failures to database.
        
        Args:
            force: If True, flush regardless of batch size
        """
        batch_size = 10  # Configurable batch size
        
        # Flush results if batch is ready or forced
        if force or len(self._pending_results) >= batch_size:
            self._flush_results_batch()
            
        # Flush failures if batch is ready or forced
        if force or len(self._pending_failures) >= batch_size:
            self._flush_failures_batch()
    
    def _flush_results_batch(self):
        """Flush pending successful results to database."""
        if not self._pending_results:
            return
            
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO pipeline_results 
                    (cache_key, pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio, 
                     git_commit, created_at, result_json)
                    VALUES (:cache_key, :pipeline_id, :gif_name, :test_colors, :test_lossy, 
                            :test_frame_ratio, :git_commit, :created_at, :result_json)
                """, self._pending_results)
                
                conn.commit()
                count = len(self._pending_results)
                self.logger.debug(f"💾 Flushed {count} cached results to database")
                self._pending_results.clear()
                
        except Exception as e:
            self.logger.warning(f"Failed to flush results batch: {e}")
    
    def _flush_failures_batch(self):
        """Flush pending failures to database."""
        if not self._pending_failures:
            return
            
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.executemany("""
                    INSERT INTO pipeline_failures 
                    (pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio, 
                     git_commit, created_at, error_type, error_message, error_traceback,
                     pipeline_steps, tools_used)
                    VALUES (:pipeline_id, :gif_name, :test_colors, :test_lossy, :test_frame_ratio,
                            :git_commit, :created_at, :error_type, :error_message, :error_traceback,
                            :pipeline_steps, :tools_used)
                """, self._pending_failures)
                
                conn.commit()
                count = len(self._pending_failures)
                self.logger.info(f"🔍 Stored {count} failures for debugging analysis")
                self._pending_failures.clear()
                
        except Exception as e:
            self.logger.warning(f"Failed to flush failures batch: {e}")
    
    def store_result(self, pipeline_id: str, gif_name: str, params: dict, result: dict):
        """Store a pipeline test result (legacy method - now uses batching).
        
        Args:
            pipeline_id: Pipeline identifier
            gif_name: Test GIF name  
            params: Test parameters dict
            result: Test result dict to cache
        """
        self.queue_result(pipeline_id, gif_name, params, result)
        self.flush_batch()  # Auto-flush when batch is ready
    
    def clear_cache(self):
        """Clear all cached results and failures."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
                results_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_failures")
                failures_count = cursor.fetchone()[0]
                
                conn.execute("DELETE FROM pipeline_results")
                conn.execute("DELETE FROM pipeline_failures")
                conn.commit()
                
                self.logger.info(f"🗑️ Cleared {results_count} cached results and {failures_count} stored failures")
                
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the current cache."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Results statistics
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_results")
                total_results = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT git_commit, COUNT(*) 
                    FROM pipeline_results 
                    GROUP BY git_commit 
                    ORDER BY COUNT(*) DESC
                """)
                results_by_commit = dict(cursor.fetchall())
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM pipeline_results 
                    WHERE git_commit = ?
                """, (self.git_commit,))
                current_commit_results = cursor.fetchone()[0]
                
                # Failures statistics
                cursor = conn.execute("SELECT COUNT(*) FROM pipeline_failures")
                total_failures = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT error_type, COUNT(*) 
                    FROM pipeline_failures 
                    WHERE git_commit = ?
                    GROUP BY error_type 
                    ORDER BY COUNT(*) DESC
                """, (self.git_commit,))
                current_failures_by_type = dict(cursor.fetchall())
                
                # Database file size
                db_size_bytes = self.cache_db_path.stat().st_size if self.cache_db_path.exists() else 0
                db_size_mb = db_size_bytes / (1024 * 1024)
                
                return {
                    'total_results': total_results,
                    'current_commit_results': current_commit_results,
                    'results_by_commit': results_by_commit,
                    'total_failures': total_failures,
                    'current_failures_by_type': current_failures_by_type,
                    'database_size_mb': round(db_size_mb, 2),
                    'current_git_commit': self.git_commit,
                    'pending_batch_size': len(self._pending_results),
                    'pending_failures_size': len(self._pending_failures)
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {
                'total_results': 0,
                'current_commit_results': 0,
                'results_by_commit': {},
                'total_failures': 0,
                'current_failures_by_type': {},
                'database_size_mb': 0,
                'current_git_commit': self.git_commit,
                'pending_batch_size': 0,
                'pending_failures_size': 0
            }
    
    def query_failures(self, error_type: Optional[str] = None, pipeline_id: Optional[str] = None, 
                      recent_hours: Optional[int] = None) -> List[dict]:
        """Query failures for debugging analysis.
        
        Args:
            error_type: Filter by specific error type (e.g., 'ffmpeg', 'timeout')
            pipeline_id: Filter by specific pipeline
            recent_hours: Only show failures from last N hours
            
        Returns:
            List of failure records
        """
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Build dynamic query
                where_conditions = ["git_commit = ?"]
                params = [self.git_commit]
                
                if error_type:
                    where_conditions.append("error_type = ?")
                    params.append(error_type)
                    
                if pipeline_id:
                    where_conditions.append("pipeline_id = ?")
                    params.append(pipeline_id)
                    
                if recent_hours:
                    from datetime import datetime, timedelta
                    cutoff_time = (datetime.now() - timedelta(hours=recent_hours)).isoformat()
                    where_conditions.append("created_at >= ?")
                    params.append(cutoff_time)
                
                where_clause = " AND ".join(where_conditions)
                
                cursor = conn.execute(f"""
                    SELECT pipeline_id, gif_name, test_colors, test_lossy, test_frame_ratio,
                           error_type, error_message, error_traceback, created_at,
                           pipeline_steps, tools_used
                    FROM pipeline_failures 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                """, params)
                
                columns = [desc[0] for desc in cursor.description]
                failures = []
                
                for row in cursor.fetchall():
                    failure = dict(zip(columns, row))
                    # Parse JSON fields
                    try:
                        failure['pipeline_steps'] = json.loads(failure['pipeline_steps'])
                        failure['tools_used'] = json.loads(failure['tools_used'])
                    except (json.JSONDecodeError, TypeError):
                        failure['pipeline_steps'] = []
                        failure['tools_used'] = []
                    
                    failures.append(failure)
                
                return failures
                
        except Exception as e:
            self.logger.warning(f"Failed to query failures: {e}")
            return []


@dataclass
class SyntheticGifSpec:
    """Specification for a synthetic test GIF."""
    name: str
    frames: int
    size: Tuple[int, int]
    content_type: str
    description: str
    

@dataclass 
class SamplingStrategy:
    """Configuration for intelligent sampling strategies."""
    name: str
    description: str
    sample_ratio: float  # Fraction of total pipelines to test
    min_samples_per_tool: int = 3  # Minimum samples per tool type

@dataclass
class EliminationResult:
    """Result of pipeline elimination analysis."""
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


class ParetoAnalyzer:
    """Advanced Pareto frontier analysis for pipeline efficiency comparison."""
    
    def __init__(self, results_df: pd.DataFrame, logger: logging.Logger = None):
        """Initialize the Pareto analyzer.
        
        Args:
            results_df: DataFrame with pipeline test results
            logger: Optional logger instance
        """
        self.results_df = results_df
        self.logger = logger or logging.getLogger(__name__)
        self.quality_metrics = ['composite_quality', 'ssim_mean', 'ms_ssim_mean']
        self.size_metrics = ['file_size_kb', 'compression_ratio']
    
    def generate_comprehensive_pareto_analysis(self) -> dict:
        """Generate complete Pareto analysis across all dimensions."""
        
        analysis = {
            'content_type_frontiers': {},
            'global_frontier': None,
            'pipeline_dominance_analysis': {},
            'efficiency_rankings': {},
            'trade_off_insights': {}
        }
        
        try:
            # 1. Per-content-type analysis
            for content_type in self.results_df['content_type'].unique():
                content_data = self.results_df[self.results_df['content_type'] == content_type]
                analysis['content_type_frontiers'][content_type] = self._compute_pareto_frontier(
                    content_data, quality_col='composite_quality', size_col='file_size_kb'
                )
            
            # 2. Global frontier across all content types
            analysis['global_frontier'] = self._compute_pareto_frontier(
                self.results_df, quality_col='composite_quality', size_col='file_size_kb'
            )
            
            # 3. Pipeline dominance analysis
            analysis['pipeline_dominance_analysis'] = self._analyze_pipeline_dominance()
            
            # 4. Efficiency rankings at quality targets
            analysis['efficiency_rankings'] = self._rank_pipelines_at_quality_targets()
            
            # 5. Trade-off insights
            analysis['trade_off_insights'] = self._generate_trade_off_insights()
            
        except Exception as e:
            self.logger.warning(f"Pareto analysis failed: {e}")
            
        return analysis
    
    def _compute_pareto_frontier(self, data: pd.DataFrame, 
                                quality_col: str, size_col: str) -> dict:
        """Compute Pareto frontier for quality vs size trade-off."""
        
        if data.empty or quality_col not in data.columns or size_col not in data.columns:
            return {'frontier_points': [], 'dominated_pipelines': []}
        
        # Filter out invalid data
        valid_data = data.dropna(subset=[quality_col, size_col])
        if valid_data.empty:
            return {'frontier_points': [], 'dominated_pipelines': []}
        
        # Step 1: For each pipeline, get its best performances
        pipeline_best_points = {}
        
        for pipeline_id in valid_data['pipeline_id'].unique():
            pipeline_data = valid_data[valid_data['pipeline_id'] == pipeline_id]
            
            # Get all quality-size points for this pipeline
            points = []
            for _, row in pipeline_data.iterrows():
                quality = row[quality_col]
                size = row[size_col]
                if pd.notna(quality) and pd.notna(size):
                    points.append((quality, size, dict(row)))
            
            # For this pipeline, find its own Pareto frontier
            pipeline_frontier = self._find_pareto_optimal_points(points)
            pipeline_best_points[pipeline_id] = pipeline_frontier
        
        # Step 2: Combine all pipeline best points
        all_candidate_points = []
        for pipeline_id, points in pipeline_best_points.items():
            for quality, size, row_data in points:
                all_candidate_points.append((quality, size, pipeline_id, row_data))
        
        # Step 3: Find global Pareto frontier
        global_frontier = self._find_pareto_optimal_points(all_candidate_points)
        
        # Step 4: Identify dominated pipelines
        frontier_pipelines = {point[2] for point in global_frontier}  # pipeline_ids in frontier
        all_pipelines = set(valid_data['pipeline_id'].unique())
        dominated_pipelines = all_pipelines - frontier_pipelines
        
        return {
            'frontier_points': [
                {
                    'quality': point[0],
                    'file_size_kb': point[1], 
                    'pipeline_id': point[2],
                    'efficiency_score': point[0] / point[1] if point[1] > 0 else 0,  # quality per KB
                    'row_data': point[3] if len(point) > 3 else None
                }
                for point in global_frontier
            ],
            'dominated_pipelines': list(dominated_pipelines),
            'pipeline_frontier_segments': pipeline_best_points
        }
    
    def _find_pareto_optimal_points(self, points: list) -> list:
        """Find Pareto optimal points from a list of (quality, size, ...) tuples.
        
        Higher quality is better, lower size is better.
        """
        if not points:
            return []
        
        # Sort by quality (descending) then by size (ascending) 
        sorted_points = sorted(points, key=lambda x: (-x[0], x[1]))
        
        pareto_frontier = []
        min_size_seen = float('inf')
        
        for point in sorted_points:
            quality, size = point[0], point[1]
            
            # This point is Pareto optimal if it has smaller size than any previous point
            # (since we're iterating in descending quality order)
            if size < min_size_seen:
                pareto_frontier.append(point)
                min_size_seen = size
        
        return pareto_frontier
    
    def _analyze_pipeline_dominance(self) -> dict:
        """Analyze which pipelines dominate others across different scenarios."""
        
        dominance_analysis = {}
        
        for content_type in self.results_df['content_type'].unique():
            content_data = self.results_df[self.results_df['content_type'] == content_type]
            
            pipeline_dominance = {}
            
            for pipeline_a in content_data['pipeline_id'].unique():
                data_a = content_data[content_data['pipeline_id'] == pipeline_a]
                dominated_by_a = []
                
                for pipeline_b in content_data['pipeline_id'].unique():
                    if pipeline_a == pipeline_b:
                        continue
                        
                    data_b = content_data[content_data['pipeline_id'] == pipeline_b]
                    
                    # Check if A dominates B (A has better/equal quality AND better/equal size)
                    if self._pipeline_dominates(data_a, data_b):
                        dominated_by_a.append(pipeline_b)
                
                pipeline_dominance[pipeline_a] = {
                    'dominates': dominated_by_a,
                    'dominance_count': len(dominated_by_a)
                }
            
            dominance_analysis[content_type] = pipeline_dominance
        
        return dominance_analysis
    
    def _pipeline_dominates(self, data_a: pd.DataFrame, data_b: pd.DataFrame) -> bool:
        """Check if pipeline A dominates pipeline B."""
        
        # Compare best performances
        best_a_quality = data_a['composite_quality'].max() if 'composite_quality' in data_a.columns else 0
        best_a_size = data_a['file_size_kb'].min() if 'file_size_kb' in data_a.columns else float('inf')
        
        best_b_quality = data_b['composite_quality'].max() if 'composite_quality' in data_b.columns else 0  
        best_b_size = data_b['file_size_kb'].min() if 'file_size_kb' in data_b.columns else float('inf')
        
        # A dominates B if A is better/equal in both dimensions and strictly better in at least one
        quality_better_or_equal = best_a_quality >= best_b_quality
        size_better_or_equal = best_a_size <= best_b_size
        
        strictly_better = (best_a_quality > best_b_quality) or (best_a_size < best_b_size)
        
        return quality_better_or_equal and size_better_or_equal and strictly_better
    
    def _rank_pipelines_at_quality_targets(self) -> dict:
        """Rank pipelines by file size at specific quality target levels."""
        
        target_qualities = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        rankings = {}
        
        for target_quality in target_qualities:
            # Find pipelines that can achieve this quality level
            capable_pipelines = {}
            
            for pipeline_id in self.results_df['pipeline_id'].unique():
                pipeline_data = self.results_df[self.results_df['pipeline_id'] == pipeline_id]
                
                # Find results at or above target quality
                quality_col = 'composite_quality'
                if quality_col in pipeline_data.columns:
                    quality_results = pipeline_data[pipeline_data[quality_col] >= target_quality]
                    
                    if not quality_results.empty:
                        # Best (smallest) file size at this quality level
                        best_size = quality_results['file_size_kb'].min()
                        capable_pipelines[pipeline_id] = {
                            'best_size_kb': best_size,
                            'samples_at_quality': len(quality_results)
                        }
            
            # Rank by file size
            ranked_pipelines = sorted(
                capable_pipelines.items(), 
                key=lambda x: x[1]['best_size_kb']
            )
            
            rankings[f"quality_{target_quality}"] = ranked_pipelines
        
        return rankings
    
    def _generate_trade_off_insights(self) -> dict:
        """Generate insights about quality-size trade-offs."""
        
        insights = {}
        
        try:
            # Calculate efficiency metrics
            if 'composite_quality' in self.results_df.columns and 'file_size_kb' in self.results_df.columns:
                # Avoid division by zero
                valid_sizes = self.results_df['file_size_kb'] > 0
                temp_df = self.results_df[valid_sizes].copy()
                temp_df['efficiency_ratio'] = temp_df['composite_quality'] / temp_df['file_size_kb']
                
                # Top efficiency leaders
                efficiency_leaders = temp_df.nlargest(10, 'efficiency_ratio')[
                    ['pipeline_id', 'composite_quality', 'file_size_kb', 'efficiency_ratio']
                ].to_dict('records')
                
                insights['efficiency_leaders'] = efficiency_leaders
                
                # Quality vs size correlation by pipeline
                pipeline_correlations = {}
                for pipeline_id in temp_df['pipeline_id'].unique():
                    pipeline_data = temp_df[temp_df['pipeline_id'] == pipeline_id]
                    if len(pipeline_data) > 2:
                        correlation = pipeline_data['composite_quality'].corr(pipeline_data['file_size_kb'])
                        if pd.notna(correlation):
                            pipeline_correlations[pipeline_id] = correlation
                
                insights['quality_size_correlations'] = pipeline_correlations
                
                # Sweet spot analysis (high quality, low size)
                quality_threshold = temp_df['composite_quality'].quantile(0.8)  # Top 20% quality
                size_threshold = temp_df['file_size_kb'].quantile(0.2)  # Bottom 20% size
                
                sweet_spot_results = temp_df[
                    (temp_df['composite_quality'] >= quality_threshold) & 
                    (temp_df['file_size_kb'] <= size_threshold)
                ]
                
                insights['sweet_spot_pipelines'] = sweet_spot_results['pipeline_id'].value_counts().to_dict()
                
        except Exception as e:
            self.logger.warning(f"Trade-off insights generation failed: {e}")
        
        return insights


class PipelineEliminator:
    """Systematic pipeline elimination through competitive testing."""
    
    # Available sampling strategies for efficient testing
    SAMPLING_STRATEGIES = {
        'full': SamplingStrategy(
            name="Full Brute Force",
            description="Test all pipeline combinations (slowest, most thorough)",
            sample_ratio=1.0,
        ),
        'representative': SamplingStrategy(
            name="Representative Sampling", 
            description="Test representative samples from each tool category",
            sample_ratio=0.15,  # ~15% of pipelines
            min_samples_per_tool=5,
        ),
        'factorial': SamplingStrategy(
            name="Factorial Design",
            description="Statistical design of experiments approach",
            sample_ratio=0.08,  # ~8% of pipelines
            min_samples_per_tool=3,
        ),
        'progressive': SamplingStrategy(
            name="Progressive Elimination",
            description="Multi-stage elimination with refinement",
            sample_ratio=0.25,  # Varies across stages
            min_samples_per_tool=4,
        ),
        'quick': SamplingStrategy(
            name="Quick Test",
            description="Fast test for development (least thorough)",
            sample_ratio=0.05,  # ~5% of pipelines
            min_samples_per_tool=2,
        ),
        'targeted': SamplingStrategy(
            name="Targeted Expansion",
            description="Strategic expansion focusing on high-value size and temporal variations",
            sample_ratio=0.12,  # ~12% of pipelines
            min_samples_per_tool=4,
        ),
    }
    
    # Constants for progress tracking and memory management
    PROGRESS_SAVE_INTERVAL = 100  # Save resume data every N jobs to prevent memory buildup
    BUFFER_FLUSH_INTERVAL = 50    # Flush results buffer every N jobs for performance
    
    def __init__(self, output_dir: Path = Path("elimination_results"), use_gpu: bool = False, use_cache: bool = True):
        self.base_output_dir = output_dir
        self.use_gpu = use_gpu
        self.use_cache = use_cache
        self.logger = logging.getLogger(__name__)
        
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
            git_commit = self._get_git_commit()
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
        
        # Define synthetic test cases based on research findings
        self.synthetic_specs = [
            # ORIGINAL RESEARCH-BASED CONTENT TYPES
            # From research: gradients benefit from dithering
            SyntheticGifSpec(
                "smooth_gradient", 8, (120, 120), "gradient",
                "Smooth color transitions - should benefit from Riemersma dithering"
            ),
            SyntheticGifSpec(
                "complex_gradient", 12, (150, 150), "complex_gradient", 
                "Multi-directional gradients with multiple hues"
            ),
            
            # From research: solid colors should NOT use dithering
            SyntheticGifSpec(
                "solid_blocks", 6, (100, 100), "solid",
                "Flat color blocks - dithering should provide no benefit"
            ),
            SyntheticGifSpec(
                "high_contrast", 10, (120, 120), "contrast",
                "Sharp edges and high contrast - no dithering benefit expected"
            ),
            
            # From research: complex/noise content where Bayer scales 4-5 excel
            SyntheticGifSpec(
                "photographic_noise", 8, (140, 140), "noise",
                "Photo-realistic with noise - good for testing Bayer dithering"
            ),
            SyntheticGifSpec(
                "texture_complex", 15, (130, 130), "texture",
                "Complex textures where dithering patterns can blend naturally"
            ),
            
            # Geometric patterns from research
            SyntheticGifSpec(
                "geometric_patterns", 10, (110, 110), "geometric",
                "Structured geometric shapes - test ordered dithering methods"
            ),
            
            # Edge cases for comprehensive testing
            SyntheticGifSpec(
                "few_colors", 6, (100, 100), "minimal",
                "Very few distinct colors - test edge behavior"
            ),
            SyntheticGifSpec(
                "many_colors", 20, (160, 160), "spectrum",
                "Full color spectrum - stress test palette reduction"
            ),
            SyntheticGifSpec(
                "animation_heavy", 30, (100, 100), "motion",
                "Rapid animation with temporal coherence requirements"
            ),
            
            # SIZE VARIATIONS - Test if dimensions affect pipeline performance
            SyntheticGifSpec(
                "gradient_small", 8, (50, 50), "gradient",
                "Small gradient - test compression behavior at minimum realistic size"
            ),
            SyntheticGifSpec(
                "gradient_medium", 8, (200, 200), "gradient", 
                "Medium gradient - standard web size testing"
            ),
            SyntheticGifSpec(
                "gradient_large", 8, (500, 500), "gradient",
                "Large gradient - test performance on bigger files"
            ),
            SyntheticGifSpec(
                "gradient_xlarge", 8, (1000, 1000), "gradient",
                "Extra large gradient - maximum realistic size testing"
            ),
            SyntheticGifSpec(
                "noise_small", 8, (50, 50), "noise",
                "Small noisy content - test Bayer dithering on small dimensions"
            ),
            SyntheticGifSpec(
                "noise_large", 8, (500, 500), "noise",
                "Large noisy content - test Bayer scale performance on large files"
            ),
            
            # FRAME COUNT VARIATIONS - Test temporal processing differences
            SyntheticGifSpec(
                "minimal_frames", 2, (120, 120), "gradient",
                "Minimal animation - test behavior with very few frames"
            ),
            SyntheticGifSpec(
                "long_animation", 50, (120, 120), "motion",
                "Long animation - test frame processing efficiency"
            ),
            SyntheticGifSpec(
                "very_long_animation", 100, (120, 120), "motion",
                "Very long animation - stress test temporal optimization"
            ),
            
            # MISSING CONTENT TYPES - Real-world patterns not covered
            SyntheticGifSpec(
                "mixed_content", 12, (200, 150), "mixed",
                "Text + graphics + photo elements - common real-world combination"
            ),
            SyntheticGifSpec(
                "data_visualization", 8, (300, 200), "charts",
                "Charts and graphs - technical/scientific content"
            ),
            SyntheticGifSpec(
                "transitions", 15, (150, 150), "morph",
                "Complex transitions and morphing - advanced animation patterns"
            ),
            
            # EDGE CASES - Extreme but realistic scenarios
            SyntheticGifSpec(
                "single_pixel_anim", 10, (100, 100), "micro_detail",
                "Single pixel changes - minimal motion detection"
            ),
            SyntheticGifSpec(
                "static_minimal_change", 20, (150, 150), "static_plus",
                "Mostly static with tiny changes - frame reduction opportunities"
            ),
            SyntheticGifSpec(
                "high_frequency_detail", 12, (200, 200), "detail",
                "High frequency details - test aliasing and quality preservation"
            )
        ]
    
    def generate_synthetic_gifs(self) -> List[Path]:
        """Generate all synthetic test GIFs."""
        self.logger.info(f"Generating {len(self.synthetic_specs)} synthetic test GIFs")
        
        gif_paths = []
        for spec in self.synthetic_specs:
            gif_path = self.output_dir / f"{spec.name}.gif"
            if not gif_path.exists():
                self._create_synthetic_gif(gif_path, spec)
            gif_paths.append(gif_path)
            
        return gif_paths
    
    def _create_synthetic_gif(self, path: Path, spec: SyntheticGifSpec):
        """Create a synthetic GIF based on specification."""
        images = []
        
        for frame_idx in range(spec.frames):
            if spec.content_type == "gradient":
                img = self._create_gradient_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "complex_gradient":
                img = self._create_complex_gradient_frame(spec.size, frame_idx, spec.frames) 
            elif spec.content_type == "solid":
                img = self._create_solid_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "contrast":
                img = self._create_contrast_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "noise":
                img = self._create_noise_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "texture":
                img = self._create_texture_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "geometric":
                img = self._create_geometric_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "minimal":
                img = self._create_minimal_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "spectrum":
                img = self._create_spectrum_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "motion":
                img = self._create_motion_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "mixed":
                img = self._create_mixed_content_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "charts":
                img = self._create_data_visualization_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "morph":
                img = self._create_transitions_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "micro_detail":
                img = self._create_single_pixel_anim_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "static_plus":
                img = self._create_static_minimal_change_frame(spec.size, frame_idx, spec.frames)
            elif spec.content_type == "detail":
                img = self._create_high_frequency_detail_frame(spec.size, frame_idx, spec.frames)
            else:
                img = self._create_simple_frame(spec.size, frame_idx, spec.frames)
                
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
    
    def _create_gradient_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Create smooth gradient frame - should benefit from Riemersma dithering."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Animated gradient shift
        offset = (frame / total_frames) * 255
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Create smooth diagonal gradient with temporal shift
                r = int((x / size[0]) * 255)
                g = int((y / size[1]) * 255) 
                b = int(((x + y + offset) / (size[0] + size[1])) * 255) % 255
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_complex_gradient_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Multi-directional gradients with multiple hues."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Time-based rotation
        phase = (frame / total_frames) * 2 * np.pi
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Distance and angle from center
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx) + phase
                
                # Complex gradient based on polar coordinates
                r = int(127 + 127 * np.sin(angle))
                g = int(127 + 127 * np.cos(angle * 1.5))
                b = int(127 + 127 * np.sin(distance / 20 + phase))
                
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_solid_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Solid color blocks - dithering should provide no benefit."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Simple color blocks that change over time
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        block_size = 20
        
        for x in range(0, size[0], block_size):
            for y in range(0, size[1], block_size):
                color_idx = ((x // block_size) + (y // block_size) + frame) % len(colors)
                draw.rectangle([x, y, x + block_size - 1, y + block_size - 1], 
                             fill=colors[color_idx])
                
        return img
    
    def _create_contrast_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """High contrast patterns - no dithering benefit expected."""
        img = Image.new('RGB', size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Moving high contrast pattern
        offset = int((frame / total_frames) * 20)
        
        for x in range(0, size[0], 10):
            for y in range(0, size[1], 10):
                if ((x + y + offset) // 10) % 2 == 0:
                    draw.rectangle([x, y, x + 9, y + 9], fill=(255, 255, 255))
                    
        return img
    
    def _create_noise_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Photo-realistic noise - good for testing Bayer dithering."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Reproducible noise with temporal coherence
        np.random.seed(frame * 42)
        
        # Base image with noise
        base_color = int(128 + 50 * np.sin(frame / total_frames * 2 * np.pi))
        
        for x in range(size[0]):
            for y in range(size[1]):
                noise_r = np.random.randint(-50, 50)
                noise_g = np.random.randint(-50, 50) 
                noise_b = np.random.randint(-50, 50)
                
                r = max(0, min(255, base_color + noise_r))
                g = max(0, min(255, base_color + noise_g))
                b = max(0, min(255, base_color + noise_b))
                
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_texture_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Complex textures where dithering patterns blend naturally."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # Procedural texture with animation
        phase = (frame / total_frames) * 4 * np.pi
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Multi-frequency texture pattern
                val1 = np.sin(x * 0.1 + phase) * np.cos(y * 0.1)
                val2 = np.sin(x * 0.05 + y * 0.05 + phase * 0.5)
                val3 = np.sin((x + y) * 0.02 + phase * 0.3)
                
                r = int(127 + 60 * val1)
                g = int(127 + 60 * val2) 
                b = int(127 + 60 * val3)
                
                draw.point((x, y), (r, g, b))
                
        return img
    
    def _create_geometric_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Structured geometric shapes - test ordered dithering."""
        img = Image.new('RGB', size, (50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        # Animated geometric patterns
        rotation = (frame / total_frames) * 360
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Draw rotating polygons
        for radius in range(20, min(size) // 2, 15):
            vertices = []
            for i in range(6):  # Hexagon
                angle = rotation + i * 60
                x = center_x + radius * np.cos(np.radians(angle))
                y = center_y + radius * np.sin(np.radians(angle))
                vertices.append((x, y))
            
            color_intensity = int(100 + 100 * (radius / (min(size) // 2)))
            draw.polygon(vertices, fill=(color_intensity, color_intensity // 2, 255 - color_intensity))
            
        return img
    
    def _create_minimal_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Very few colors - test edge behavior."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        current_color = colors[frame % len(colors)]
        return Image.new('RGB', size, current_color)
    
    def _create_spectrum_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Full color spectrum - stress test palette reduction."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        # HSV color wheel with animation
        phase = (frame / total_frames) * 360
        
        center_x, center_y = size[0] // 2, size[1] // 2
        max_radius = min(size) // 2
        
        for x in range(size[0]):
            for y in range(size[1]):
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.degrees(np.arctan2(dy, dx)) + phase
                
                if distance <= max_radius:
                    # HSV to RGB conversion
                    hue = angle % 360
                    saturation = distance / max_radius
                    value = 1.0
                    
                    h_i = int(hue / 60) % 6
                    f = hue / 60 - h_i
                    p = value * (1 - saturation)
                    q = value * (1 - f * saturation)
                    t = value * (1 - (1 - f) * saturation)
                    
                    if h_i == 0: r, g, b = value, t, p
                    elif h_i == 1: r, g, b = q, value, p
                    elif h_i == 2: r, g, b = p, value, t
                    elif h_i == 3: r, g, b = p, q, value
                    elif h_i == 4: r, g, b = t, p, value
                    else: r, g, b = value, p, q
                    
                    draw.point((x, y), (int(r*255), int(g*255), int(b*255)))
                    
        return img
    
    def _create_motion_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Rapid animation with temporal coherence requirements."""
        img = Image.new('RGB', size, (0, 0, 50))
        draw = ImageDraw.Draw(img)
        
        # Moving objects with trails
        progress = frame / total_frames
        
        # Primary moving object
        obj_x = int(progress * (size[0] - 20))
        obj_y = int(size[1] // 2 + 30 * np.sin(progress * 6 * np.pi))
        
        draw.ellipse([obj_x, obj_y, obj_x + 20, obj_y + 20], fill=(255, 200, 100))
        
        # Secondary bouncing object
        bounce_x = int(size[0] // 2 + (size[0] // 3) * np.sin(progress * 4 * np.pi))
        bounce_y = int(abs(np.sin(progress * 8 * np.pi)) * (size[1] - 30))
        
        draw.rectangle([bounce_x, bounce_y, bounce_x + 15, bounce_y + 15], fill=(100, 255, 150))
        
        return img
    
    def _create_simple_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Fallback simple frame."""
        img = Image.new('RGB', size, (100, 100, 100))
        draw = ImageDraw.Draw(img)
        draw.rectangle([frame * 5, frame * 5, frame * 5 + 20, frame * 5 + 20], fill=(255, 255, 255))
        return img

    # NEW FRAME GENERATION METHODS FOR EXPANDED SYNTHETIC DATASET

    def _create_mixed_content_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Mixed content: text + graphics + photo elements - common real-world combination."""
        img = Image.new('RGB', size, (240, 240, 245))  # Light gray background
        draw = ImageDraw.Draw(img)
        
        # Photo-like gradient region (top third)
        gradient_height = size[1] // 3
        for y in range(gradient_height):
            intensity = int(180 + 50 * (y / gradient_height))
            color_shift = int(20 * np.sin(frame / total_frames * 2 * np.pi))
            color = (intensity + color_shift, intensity - color_shift//2, intensity + color_shift//3)
            draw.line([(0, y), (size[0], y)], fill=color)
        
        # Text-like blocks (middle third)  
        text_y_start = gradient_height
        text_y_end = 2 * gradient_height
        block_width = size[0] // 8
        for i in range(8):
            x = i * block_width
            block_height = 5 + (frame + i) % 8
            y = text_y_start + 20 + i * 3
            if y + block_height < text_y_end:
                draw.rectangle([x, y, x + block_width - 2, y + block_height], fill=(50, 50, 50))
        
        # Graphics elements (bottom third)
        graphics_y_start = text_y_end
        circle_x = int((frame / total_frames) * (size[0] - 40) + 20)
        circle_y = graphics_y_start + 20
        draw.ellipse([circle_x - 15, circle_y - 15, circle_x + 15, circle_y + 15], 
                    fill=(255, 100, 100))
        
        # Add some sharp graphic lines
        for i in range(3):
            y_pos = graphics_y_start + 50 + i * 10
            line_length = int((frame / total_frames + i * 0.3) * size[0]) % size[0]
            draw.line([(0, y_pos), (line_length, y_pos)], 
                     fill=(0, 150, 200), width=2)
        
        return img

    def _create_data_visualization_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Charts and graphs - technical/scientific content."""
        img = Image.new('RGB', size, (255, 255, 255))  # White background
        draw = ImageDraw.Draw(img)
        
        # Draw axes
        margin = 30
        chart_width = size[0] - 2 * margin
        chart_height = size[1] - 2 * margin
        
        # X and Y axes
        draw.line([(margin, size[1] - margin), (size[0] - margin, size[1] - margin)], 
                 fill=(0, 0, 0), width=2)  # X-axis
        draw.line([(margin, margin), (margin, size[1] - margin)], 
                 fill=(0, 0, 0), width=2)  # Y-axis
        
        # Animated data points
        num_points = 10
        for i in range(num_points):
            x = margin + (i / (num_points - 1)) * chart_width
            
            # Animated sine wave data
            base_height = 0.5 + 0.3 * np.sin(i / 2 + frame / total_frames * 2 * np.pi)
            y = size[1] - margin - base_height * chart_height
            
            # Data point
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(200, 50, 50))
            
            # Connect lines
            if i > 0:
                prev_x = margin + ((i-1) / (num_points - 1)) * chart_width
                prev_base = 0.5 + 0.3 * np.sin((i-1) / 2 + frame / total_frames * 2 * np.pi)
                prev_y = size[1] - margin - prev_base * chart_height
                draw.line([(prev_x, prev_y), (x, y)], fill=(100, 150, 200), width=2)
        
        # Grid lines
        for i in range(1, 5):
            grid_y = margin + (i / 5) * chart_height
            draw.line([(margin, grid_y), (size[0] - margin, grid_y)], 
                     fill=(200, 200, 200), width=1)
        
        # Bar chart in corner
        bar_start_x = size[0] - 100
        bar_values = [0.3, 0.7, 0.5, 0.9]
        for i, val in enumerate(bar_values):
            animated_val = val + 0.1 * np.sin(frame / total_frames * 2 * np.pi + i)
            bar_height = max(5, int(animated_val * 50))
            bar_x = bar_start_x + i * 15
            bar_y = size[1] - margin - 10
            draw.rectangle([bar_x, bar_y - bar_height, bar_x + 10, bar_y], 
                         fill=(150, 100, 250))
        
        return img

    def _create_transitions_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Complex transitions and morphing - advanced animation patterns."""
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        
        progress = frame / total_frames
        
        # Morphing between circle and square
        if progress < 0.5:
            # Circle to square transition
            morph_progress = progress * 2  # 0 to 1
            
            center_x, center_y = size[0] // 2, size[1] // 2
            radius = min(size) // 4
            
            # Interpolate between circle and square points
            points = []
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                
                # Circle position
                circle_x = center_x + radius * np.cos(rad)
                circle_y = center_y + radius * np.sin(rad)
                
                # Square position (approximate)
                if -45 <= angle < 45 or 315 <= angle < 360:  # Right side
                    square_x = center_x + radius
                    square_y = center_y + radius * np.tan(rad)
                elif 45 <= angle < 135:  # Top side
                    square_y = center_y - radius
                    square_x = center_x - radius * np.tan(rad - np.pi/2)
                elif 135 <= angle < 225:  # Left side
                    square_x = center_x - radius
                    square_y = center_y - radius * np.tan(rad - np.pi)
                else:  # Bottom side
                    square_y = center_y + radius
                    square_x = center_x + radius * np.tan(rad - 3*np.pi/2)
                
                # Interpolate
                x = circle_x + morph_progress * (square_x - circle_x)
                y = circle_y + morph_progress * (square_y - circle_y)
                points.append((x, y))
            
            if len(points) > 2:
                # Smooth color transition
                color_r = int(255 * (1 - morph_progress))
                color_g = int(255 * morph_progress)
                draw.polygon(points, fill=(color_r, color_g, 100))
        
        else:
            # Square to triangle transition
            morph_progress = (progress - 0.5) * 2  # 0 to 1
            
            center_x, center_y = size[0] // 2, size[1] // 2
            size_val = min(size) // 4
            
            # Square corners
            square_points = [
                (center_x - size_val, center_y - size_val),
                (center_x + size_val, center_y - size_val),
                (center_x + size_val, center_y + size_val),
                (center_x - size_val, center_y + size_val)
            ]
            
            # Triangle corners
            triangle_points = [
                (center_x, center_y - size_val),
                (center_x + size_val, center_y + size_val),
                (center_x - size_val, center_y + size_val)
            ]
            
            # Interpolate (square has 4 points, triangle has 3)
            final_points = []
            for i in range(3):
                sq_x, sq_y = square_points[i]
                tri_x, tri_y = triangle_points[i]
                x = sq_x + morph_progress * (tri_x - sq_x)
                y = sq_y + morph_progress * (tri_y - sq_y)
                final_points.append((x, y))
            
            color_g = int(255 * (1 - morph_progress))
            color_b = int(255 * morph_progress)
            draw.polygon(final_points, fill=(100, color_g, color_b))
        
        return img

    def _create_single_pixel_anim_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Single pixel changes - minimal motion detection."""
        img = Image.new('RGB', size, (128, 128, 128))  # Gray background
        
        # Only change a few pixels each frame
        pixels_to_change = [(
            (frame * 7 + i * 13) % size[0],
            (frame * 5 + i * 11) % size[1]
        ) for i in range(3)]
        
        for x, y in pixels_to_change:
            # Subtle color changes
            color_shift = (frame * 17 + x + y) % 64  # Small variation
            img.putpixel((x, y), (128 + color_shift, 128 - color_shift//2, 128 + color_shift//3))
        
        # Add one more obvious but tiny moving element
        moving_x = (frame * 2) % size[0]
        moving_y = (frame) % size[1]
        img.putpixel((moving_x, moving_y), (255, 255, 255))
        
        return img

    def _create_static_minimal_change_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """Mostly static with tiny changes - frame reduction opportunities."""
        # Base static image
        img = Image.new('RGB', size, (200, 210, 220))
        draw = ImageDraw.Draw(img)
        
        # Static background pattern
        for x in range(0, size[0], 20):
            for y in range(0, size[1], 20):
                draw.rectangle([x, y, x + 18, y + 18], outline=(180, 190, 200))
        
        # Very minimal animated element - only changes every few frames
        if frame % 5 == 0 and size[0] > 10:  # Only change every 5th frame, if size allows
            change_x = (frame // 5) % (size[0] - 10)
            change_y = size[1] // 2
            draw.ellipse([change_x, change_y, change_x + 8, change_y + 8], 
                        fill=(220, 100, 100))
        
        # Tiny blinking element
        if frame % 8 < 2 and size[0] > 20 and size[1] > 20:  # Blink pattern, if size allows
            blink_x = size[0] - 20
            blink_y = 20
            draw.ellipse([blink_x, blink_y, blink_x + 4, blink_y + 4], 
                        fill=(100, 220, 100))
        
        return img

    def _create_high_frequency_detail_frame(self, size: Tuple[int, int], frame: int, total_frames: int) -> Image.Image:
        """High frequency details - test aliasing and quality preservation."""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Fine grid pattern
        for x in range(0, size[0], 2):
            for y in range(0, size[1], 2):
                if (x + y + frame) % 4 == 0:
                    draw.point((x, y), (0, 0, 0))
        
        # Moiré patterns - high frequency interference
        center_x, center_y = size[0] // 2, size[1] // 2
        for x in range(size[0]):
            for y in range(size[1]):
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # High frequency radial pattern
                freq = 0.5 + frame / total_frames * 0.3
                if int(distance * freq) % 2 == 0:
                    img.putpixel((x, y), (200, 200, 200))
        
        # Fine diagonal lines that create aliasing
        line_spacing = 3
        offset = frame % (line_spacing * 2)
        for i in range(-size[0], size[0], line_spacing):
            x1, y1 = i + offset, 0
            x2, y2 = i + offset + size[1], size[1]
            if 0 <= x1 < size[0] and 0 <= x2 < size[0]:
                draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=1)
        
        return img

    def select_pipelines_intelligently(self, all_pipelines: List, strategy: str = 'representative') -> List:
        """Select pipelines using intelligent sampling strategies to reduce testing time."""
        sampling_config = self.SAMPLING_STRATEGIES.get(strategy, self.SAMPLING_STRATEGIES['representative'])
        
        self.logger.info(f"🧠 Using sampling strategy: {sampling_config.name}")
        self.logger.info(f"📋 Description: {sampling_config.description}")
        self.logger.info(f"📊 Target sample ratio: {sampling_config.sample_ratio:.1%}")
        
        if strategy == 'factorial':
            return self._factorial_design_sampling(all_pipelines, sampling_config)
        elif strategy == 'progressive':
            return self._progressive_elimination_sampling(all_pipelines, sampling_config)
        elif strategy == 'targeted':
            return self._targeted_expansion_sampling(all_pipelines, sampling_config)
        else:
            return self._representative_sampling(all_pipelines, sampling_config)
    
    def _representative_sampling(self, all_pipelines: List, config: SamplingStrategy) -> List:
        """Sample representative pipelines from each tool category."""
        from collections import defaultdict
        import random
        
        # Handle empty pipeline list
        if not all_pipelines:
            self.logger.warning("⚠️ No pipelines provided for sampling")
            return []
        
        # Validate pipeline objects - ensure they have the expected structure
        valid_pipelines = []
        for pipeline in all_pipelines:
            if hasattr(pipeline, 'steps') and hasattr(pipeline.steps, '__iter__'):
                try:
                    # Test if we can access tool names (this will fail for invalid objects)
                    _ = [step.tool_cls.NAME for step in pipeline.steps]
                    valid_pipelines.append(pipeline)
                except (AttributeError, TypeError):
                    self.logger.warning(f"⚠️ Invalid pipeline object detected: {type(pipeline)}. Skipping.")
            else:
                self.logger.warning(f"⚠️ Pipeline object missing 'steps' attribute: {type(pipeline)}. Skipping.")
        
        if not valid_pipelines:
            self.logger.warning("⚠️ No valid pipeline objects found for sampling")
            return []
        
        # Group pipelines by tool categories and dithering methods
        tool_groups = defaultdict(list)
        
        for pipeline in valid_pipelines:
            # Extract tool signatures from pipeline steps
            tool_signature = "_".join([step.tool_cls.NAME for step in pipeline.steps])
            tool_groups[tool_signature].append(pipeline)
        
        selected_pipelines = []
        total_target = int(len(all_pipelines) * config.sample_ratio)
        
        # Handle case where no tool groups found
        if not tool_groups:
            self.logger.warning("⚠️ No tool groups found in pipelines")
            return []
            
        samples_per_group = max(config.min_samples_per_tool, total_target // len(tool_groups))
        
        self.logger.info(f"🔧 Tool groups found: {len(tool_groups)}")
        self.logger.info(f"📈 Target samples per group: {samples_per_group}")
        
        for tool_sig, pipelines in tool_groups.items():
            # Sample from this group - prioritize diversity in parameters
            group_samples = min(samples_per_group, len(pipelines))
            
            if group_samples == len(pipelines):
                selected_pipelines.extend(pipelines)
            else:
                # Ensure diversity by sampling across different parameter ranges
                sampled = self._diverse_parameter_sampling(pipelines, group_samples)
                selected_pipelines.extend(sampled)
        
        # If we're under target, add random samples from high-potential pipelines
        if len(selected_pipelines) < total_target:
            # Use list comprehension instead of set operations on non-hashable objects
            selected_ids = {id(p) for p in selected_pipelines}
            remaining = [p for p in all_pipelines if id(p) not in selected_ids]
            additional = min(total_target - len(selected_pipelines), len(remaining))
            if remaining and additional > 0:
                selected_pipelines.extend(random.sample(remaining, additional))
        
        actual_count = min(total_target, len(selected_pipelines))
        
        # Safe percentage calculation
        percentage = (actual_count / len(all_pipelines) * 100) if all_pipelines else 0
        self.logger.info(f"✅ Selected {actual_count} pipelines from {len(all_pipelines)} total ({percentage:.1f}%)")
        return selected_pipelines[:actual_count]
    
    def _diverse_parameter_sampling(self, pipelines: List, n_samples: int) -> List:
        """Sample pipelines with diverse parameter combinations."""
        import random
        
        if n_samples >= len(pipelines):
            return pipelines
        
        # TODO: Enhanced parameter space analysis
        # For now, use stratified random sampling
        return random.sample(pipelines, n_samples)
    
    def _factorial_design_sampling(self, all_pipelines: List, config: SamplingStrategy) -> List:
        """Use statistical design of experiments for efficient sampling."""
        self.logger.info("🧪 Using factorial design approach")
        
        # Identify key factors for factorial design:
        # Factor 1: Tool family (ImageMagick, FFmpeg, Gifsicle, etc.)
        # Factor 2: Dithering method category (None, Floyd-Steinberg, Bayer, etc.) 
        # Factor 3: Color reduction level (Low: 8-16, Medium: 32-64, High: 128+)
        # Factor 4: Lossy compression (None: 0, Light: 20-40, Heavy: 60+)
        
        tool_families = set()
        dithering_categories = set()
        
        for pipeline in all_pipelines:
            for step in pipeline.steps:
                if hasattr(step.tool_cls, 'NAME'):
                    tool_name = step.tool_cls.NAME
                    tool_families.add(tool_name.split('_')[0])  # Get base tool name
                    
                    # Categorize dithering methods
                    if 'None' in tool_name or 'none' in tool_name.lower():
                        dithering_categories.add('none')
                    elif 'floyd' in tool_name.lower() or 'FloydSteinberg' in tool_name:
                        dithering_categories.add('floyd_steinberg')
                    elif 'bayer' in tool_name.lower() or 'Bayer' in tool_name:
                        dithering_categories.add('bayer')
                    elif 'riemersma' in tool_name.lower():
                        dithering_categories.add('riemersma')
                    else:
                        dithering_categories.add('other')
        
        # Create factorial combinations
        target_count = int(len(all_pipelines) * config.sample_ratio)
        combinations_needed = min(target_count, len(tool_families) * len(dithering_categories) * 3 * 2)  # 3 color levels, 2 lossy levels
        
        self.logger.info(f"🔬 Factorial design: {len(tool_families)} tools × {len(dithering_categories)} dithering methods")
        self.logger.info(f"🎯 Target factorial combinations: {combinations_needed}")
        
        # For now, fall back to representative sampling with factorial weighting
        return self._representative_sampling(all_pipelines, config)
    
    def _progressive_elimination_sampling(self, all_pipelines: List, config: SamplingStrategy) -> List:
        """Multi-stage progressive elimination to focus on promising pipelines."""
        self.logger.info("📈 Using progressive elimination strategy")
        
        # Stage 1: Quick screening (5% of pipelines)
        stage1_config = SamplingStrategy("stage1", "Initial screening", 0.05, min_samples_per_tool=2)
        stage1_pipelines = self._representative_sampling(all_pipelines, stage1_config)
        
        self.logger.info(f"🔍 Stage 1: Screening {len(stage1_pipelines)} pipelines for initial assessment")
        
        # In a full implementation, we would:
        # 1. Run Stage 1 testing
        # 2. Identify top-performing tool families/dithering methods
        # 3. Run Stage 2 with more comprehensive testing of promising categories
        # 4. Run Stage 3 with full parameter sweeps of the best candidates
        
        # For now, return stage 1 selection with expanded promising categories
        expanded_target = int(len(all_pipelines) * config.sample_ratio)
        if len(stage1_pipelines) < expanded_target:
            # Add more samples from promising categories (would be data-driven in full implementation)
            stage1_ids = {id(p) for p in stage1_pipelines}
            remaining = [p for p in all_pipelines if id(p) not in stage1_ids] 
            additional_needed = expanded_target - len(stage1_pipelines)
            if remaining:
                additional_samples = remaining[:additional_needed]
                stage1_pipelines.extend(additional_samples)
        
        self.logger.info(f"📊 Progressive sampling selected {len(stage1_pipelines)} pipelines")
        return stage1_pipelines

    def _targeted_expansion_sampling(self, all_pipelines: List, config: SamplingStrategy) -> List:
        """Strategic sampling focused on high-value expanded dataset testing."""
        self.logger.info("🎯 Using targeted expansion strategy")
        
        # Validate input - delegate to representative sampling which has validation
        selected_pipelines = self._representative_sampling(all_pipelines, config)
        
        self.logger.info(f"📊 Targeted expansion selected {len(selected_pipelines)} pipelines")
        self.logger.info("🎯 Will test on strategically selected GIF subset (17 vs 25 GIFs)")
        
        return selected_pipelines

    def get_targeted_synthetic_gifs(self) -> List[Path]:
        """Generate a strategically reduced set of synthetic GIFs for targeted testing."""
        self.logger.info("🎯 Generating targeted synthetic GIF subset")
        
        # Define high-value subset: Original + Size variations + 1 frame variation + 1 content type
        targeted_specs = []
        
        # Keep all original research-based content (10 GIFs)
        original_names = [
            'smooth_gradient', 'complex_gradient', 'solid_blocks', 'high_contrast',
            'photographic_noise', 'texture_complex', 'geometric_patterns', 
            'few_colors', 'many_colors', 'animation_heavy'
        ]
        
        # Add high-value size variations (4 GIFs) - skip medium, keep key sizes
        size_variation_names = [
            'gradient_small',    # 50x50 - minimum realistic
            'gradient_large',    # 500x500 - big file performance  
            'gradient_xlarge',   # 1000x1000 - maximum realistic
            'noise_large'        # 500x500 - test Bayer on large files
        ]
        
        # Add key frame variation (2 GIFs) - most informative extremes
        frame_variation_names = [
            'minimal_frames',    # 2 frames - edge case
            'long_animation'     # 50 frames - extended animation (skip 100 frame extreme)
        ]
        
        # Add most valuable new content type (1 GIF)
        new_content_names = [
            'mixed_content'      # Real-world mixed content (skip data viz and transitions initially)
        ]
        
        # Combine all targeted names
        targeted_names = original_names + size_variation_names + frame_variation_names + new_content_names
        
        # Generate only the targeted GIFs
        targeted_specs = [spec for spec in self.synthetic_specs if spec.name in targeted_names]
        
        self.logger.info(f"🎯 Selected {len(targeted_specs)} high-value GIFs:")
        self.logger.info(f"   📋 Original research-based: {len(original_names)}")
        self.logger.info(f"   📏 Key size variations: {len(size_variation_names)}")  
        self.logger.info(f"   🎬 Key frame variations: {len(frame_variation_names)}")
        self.logger.info(f"   🔄 Essential new content: {len(new_content_names)}")
        
        gif_paths = []
        for spec in targeted_specs:
            gif_path = self.output_dir / f"{spec.name}.gif"
            if not gif_path.exists():
                self._create_synthetic_gif(gif_path, spec)
            gif_paths.append(gif_path)
            
        return gif_paths

    def run_elimination_analysis(self, 
                                test_pipelines: List[Pipeline] = None,
                                elimination_threshold: float = 0.05,
                                use_targeted_gifs: bool = False) -> EliminationResult:
        """Run competitive elimination analysis on synthetic GIFs.
        
        Args:
            test_pipelines: Specific pipelines to test (None = all pipelines)
            elimination_threshold: SSIM threshold for elimination (lower = stricter)
            
        Returns:
            EliminationResult with eliminated and retained pipelines
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
        elimination_result = self._analyze_and_eliminate(results_df, elimination_threshold)
        
        # Save results
        self._save_results(elimination_result, results_df)
        
        return elimination_result
    
    def _run_comprehensive_testing(self, gif_paths: List[Path], pipelines: List[Pipeline]) -> pd.DataFrame:
        """Run comprehensive testing of all pipeline combinations with streaming results to disk."""
        from .metrics import calculate_comprehensive_metrics, DEFAULT_METRICS_CONFIG
        import tempfile
        import time
        import csv
        
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm is not available
            class tqdm:
                def __init__(self, total=None, initial=0, desc="", unit="", bar_format=""):
                    self.total = total
                    self.n = initial
                    self.desc = desc
                
                def update(self, n=1):
                    self.n += n
                    if hasattr(self, 'total') and self.total:
                        percentage = (self.n / self.total) * 100
                        print(f"\r{self.desc}: {self.n}/{self.total} ({percentage:.1f}%)", end="", flush=True)
                
                def close(self):
                    print()  # New line after progress
        
        # Calculate total jobs for progress tracking
        total_jobs = len(gif_paths) * len(pipelines) * len(self.test_params)
        self.logger.info(f"Starting comprehensive testing: {total_jobs:,} total pipeline combinations")
        
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
            'gif_name', 'content_type', 'pipeline_id', 'success',
            'file_size_kb', 'original_size_kb', 'compression_ratio',
            'ssim_mean', 'ssim_std', 'ssim_min', 'ssim_max',
            'ms_ssim_mean', 'psnr_mean', 'temporal_consistency',
            'mse_mean', 'rmse_mean', 'fsim_mean', 'gmsd_mean',
            'chist_mean', 'edge_similarity_mean', 'texture_similarity_mean',
            'sharpness_similarity_mean', 'composite_quality',
            'render_time_ms', 'total_processing_time_ms',
            'pipeline_steps', 'tools_used', 'test_colors', 'test_lossy', 'test_frame_ratio',
            'error', 'error_traceback', 'error_timestamp'
        ]
        
        # Initialize streaming CSV with headers
        write_csv_header = not streaming_csv_path.exists()
        csv_file = open(streaming_csv_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        if write_csv_header:
            csv_writer.writeheader()
        
        # Estimate remaining time
        remaining_jobs = total_jobs - len(completed_job_ids)
        estimated_time = self._estimate_execution_time(remaining_jobs)
        self.logger.info(f"Estimated completion time: {estimated_time}")
        
        # Setup progress bar
        progress = tqdm(total=total_jobs, initial=len(completed_job_ids), 
                       desc="Testing pipelines", unit="jobs",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        # Use small memory buffer for batching, not unlimited accumulation
        results_buffer = []
        buffer_size = 100  # Process in batches of 100 results
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
                            self.logger.error(error_msg)
                            self.cache.queue_failure(
                                pipeline_id=pipeline.identifier(),
                                gif_name=gif_path.name,
                                test_colors=params.get("colors", 0),
                                test_lossy=params.get("lossy", 0),
                                test_frame_ratio=params.get("frame_ratio", 1.0),
                                error_type="validation",
                                error_message=error_msg,
                                error_traceback="",
                                pipeline_steps=str([step.name() for step in pipeline.steps]),
                                tools_used=str([step.tool_cls.NAME for step in pipeline.steps])
                            )
                            # Add validation failure to completed jobs and continue
                            failed_result = {
                                'gif_name': gif_path.stem,
                                'content_type': content_type,
                                'pipeline_id': pipeline.identifier(),
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
                                self.logger.error(error_msg)
                                self.cache.queue_failure(
                                    pipeline_id=pipeline.identifier(),
                                    gif_name=gif_path.name,
                                    test_colors=params.get("colors", 0),
                                    test_lossy=params.get("lossy", 0),
                                    test_frame_ratio=params.get("frame_ratio", 1.0),
                                    error_type="validation",
                                    error_message=error_msg,
                                    error_traceback="",
                                    pipeline_steps=str([step.name() for step in pipeline.steps]),
                                    tools_used=str([step.tool_cls.NAME for step in pipeline.steps])
                                )
                                # Add validation failure to completed jobs and skip this pipeline
                                failed_result = {
                                    'gif_name': gif_path.stem,
                                    'content_type': content_type,
                                    'pipeline_id': pipeline.identifier(),
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
                            'error': clean_error_message(str(e)),  # Clean error message for CSV
                            'error_traceback': traceback.format_exc().replace('\n', ' | '),  # Preserve full traceback
                            'error_timestamp': datetime.now().isoformat(),
                            'success': False,
                            'pipeline_steps': [step.name for step in pipeline.steps] if hasattr(pipeline, 'steps') else [],
                            'tools_used': pipeline.tools_used() if hasattr(pipeline, 'tools_used') else [],
                            'test_colors': params.get('colors', None),
                            'test_lossy': params.get('lossy', None),
                            'test_frame_ratio': params.get('frame_ratio', None),
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
        
        # Log cache performance statistics
        if self.cache:
            total_cache_operations = cache_hits + cache_misses
            if total_cache_operations > 0:
                cache_hit_rate = (cache_hits / total_cache_operations) * 100
                self.logger.info(f"💾 Cache performance: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
                
                # Estimate time saved
                estimated_time_per_test = 2.5  # seconds (conservative estimate)
                time_saved_minutes = (cache_hits * estimated_time_per_test) / 60
                if time_saved_minutes > 1:
                    self.logger.info(f"⏱️ Estimated time saved by cache: {time_saved_minutes:.1f} minutes")
            else:
                self.logger.info("💾 Cache statistics: No cache operations performed")
        
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
            results_df = pd.read_csv(streaming_csv_path)
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
            
        if force or len(buffer) >= 50:  # Configurable flush threshold
            try:
                # Write all buffered results to CSV
                for result in buffer:
                    # Ensure all required fields are present with safe defaults
                    safe_result = {
                        'gif_name': result.get('gif_name', ''),
                        'content_type': result.get('content_type', ''),
                        'pipeline_id': result.get('pipeline_id', ''),
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
                        'render_time_ms': result.get('render_time_ms', 0),
                        'total_processing_time_ms': result.get('total_processing_time_ms', 0),
                        'pipeline_steps': str(result.get('pipeline_steps', [])),
                        'tools_used': str(result.get('tools_used', [])),
                        'test_colors': result.get('test_colors', 0),
                        'test_lossy': result.get('test_lossy', 0),
                        'test_frame_ratio': result.get('test_frame_ratio', 1.0),
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
        from .metrics import calculate_comprehensive_metrics
        from .dynamic_pipeline import Pipeline
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
                    
                    # Pass PNG sequence to gifski if available from previous step
                    if png_sequence_dir and "Gifski" in wrapper.__class__.__name__:
                        step_params["png_sequence_dir"] = str(png_sequence_dir)
                
                # Check if next step is gifski 
                next_step_is_gifski = False
                if i + 1 < len(pipeline.steps):
                    next_wrapper_name = pipeline.steps[i + 1].tool_cls.__name__
                    next_step_is_gifski = "Gifski" in next_wrapper_name
                
                # Always run the current step first
                step_result = wrapper.apply(current_input, step_output, params=step_params)
                pipeline_metadata.update(step_result)
                
                # Export PNG sequence AFTER step is applied if next step is gifski
                # Also verify gifski tool is actually available to prevent race conditions
                wrapper_name = wrapper.__class__.__name__
                supports_png_export = (
                    "FFmpegFrameReducer" in wrapper_name or "FFmpegColorReducer" in wrapper_name or
                    "ImageMagickFrameReducer" in wrapper_name or "ImageMagickColorReducer" in wrapper_name
                )
                if (next_step_is_gifski and supports_png_export and self._is_gifski_available()):
                    png_sequence_dir = tmpdir_path / f"png_sequence_{step.variable}"
                    
                    # Use appropriate export function based on tool
                    # Export from step_output (processed result) not current_input (raw input)
                    if "FFmpeg" in wrapper.__class__.__name__:
                        from .external_engines.ffmpeg import export_png_sequence
                        png_result = export_png_sequence(step_output, png_sequence_dir)
                    else:  # ImageMagick
                        from .external_engines.imagemagick import export_png_sequence
                        png_result = export_png_sequence(step_output, png_sequence_dir)
                    
                    # Merge PNG export metadata
                    pipeline_metadata.update({f"png_export_{step.variable}": png_result})
                else:
                    png_sequence_dir = None  # Reset if not using PNG sequence optimization
                
                current_input = step_output
            
            # Copy final result to output path
            import shutil
            shutil.copy(current_input, output_path)
            
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
            result = {
                'gif_name': gif_path.stem,
                'content_type': content_type,
                'pipeline_id': pipeline.identifier(),
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
                
                # Composite quality score (weighted combination)
                'composite_quality': quality_metrics.get('composite_quality', 0.0),
                
                # Performance metrics
                'render_time_ms': quality_metrics.get('render_ms', 0),
                'total_processing_time_ms': int((time.perf_counter() - start_time) * 1000),
                
                # Pipeline details
                'pipeline_steps': len(pipeline.steps),
                'tools_used': [step.tool_cls.NAME for step in pipeline.steps],
                
                # Test parameters
                'test_colors': params["colors"],
                'test_lossy': params["lossy"],
                'test_frame_ratio': params.get("frame_ratio", 1.0),
            }
            
            return result
    
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
            from .metrics import calculate_comprehensive_metrics
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
                from .metrics import calculate_comprehensive_metrics
                return calculate_comprehensive_metrics(original_path, compressed_path)
            
            self.logger.debug("🚀 Computing quality metrics using GPU acceleration")
            
            # Use GPU-accelerated version
            return self._calculate_cuda_metrics(original_path, compressed_path)
            
        except ImportError:
            # OpenCV CUDA not available, fall back to CPU with clear explanation
            self.logger.warning("🔄 OpenCV CUDA support lost during processing")
            self.logger.warning("🔄 Falling back to CPU for quality metrics calculation")
            self.logger.info("💡 Install opencv-python with CUDA support for better performance")
            from .metrics import calculate_comprehensive_metrics
            return calculate_comprehensive_metrics(original_path, compressed_path)
    
    def _calculate_cuda_metrics(self, original_path: Path, compressed_path: Path) -> dict:
        """GPU-accelerated quality metrics calculation using CUDA."""
        import cv2
        import numpy as np
        import time
        from .metrics import extract_gif_frames, resize_to_common_dimensions, align_frames
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
        
        # Calculate composite quality using traditional metrics
        composite_quality = (
            config.SSIM_WEIGHT * result['ssim'] +
            config.MS_SSIM_WEIGHT * result['ms_ssim'] +
            config.PSNR_WEIGHT * result['psnr'] +
            config.TEMPORAL_WEIGHT * result['temporal_consistency']
        )
        result['composite_quality'] = float(composite_quality)
        
        # Add system metrics
        result['kilobytes'] = float(compressed_path.stat().st_size / 1024)
        
        # Calculate processing time
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        result['render_ms'] = min(int(elapsed_seconds * 1000), 86400000)
        
        return result
    
    def _gpu_ssim(self, gpu_img1: 'cv2.cuda_GpuMat', gpu_img2: 'cv2.cuda_GpuMat') -> float:
        """GPU-accelerated SSIM calculation (simplified version)."""
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
        except Exception as e:
            self.logger.warning(f"GPU color conversion failed: {e}")
            return 0.0
        
        try:
            # Convert to float32 for calculations
            gpu_float1 = cv2.cuda.convertTo(gpu_gray1, cv2.CV_32F)
            gpu_float2 = cv2.cuda.convertTo(gpu_gray2, cv2.CV_32F)
            
            # SSIM constants
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # Mean calculations using GPU blur
            kernel_size = (11, 11)
            sigma = 1.5
            
            mu1 = cv2.cuda.GaussianBlur(gpu_float1, kernel_size, sigma)
            mu2 = cv2.cuda.GaussianBlur(gpu_float2, kernel_size, sigma)
        except Exception as e:
            self.logger.warning(f"GPU SSIM calculation failed: {e}")
            return 0.0
        
        mu1_sq = cv2.cuda.multiply(mu1, mu1)
        mu2_sq = cv2.cuda.multiply(mu2, mu2)
        mu1_mu2 = cv2.cuda.multiply(mu1, mu2)
        
        # Variance calculations
        sigma1_sq = cv2.cuda.GaussianBlur(cv2.cuda.multiply(gpu_float1, gpu_float1), kernel_size, sigma)
        sigma1_sq = cv2.cuda.subtract(sigma1_sq, mu1_sq)
        
        sigma2_sq = cv2.cuda.GaussianBlur(cv2.cuda.multiply(gpu_float2, gpu_float2), kernel_size, sigma)
        sigma2_sq = cv2.cuda.subtract(sigma2_sq, mu2_sq)
        
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
        
        FOCUSED: Consistent 50% frame reduction with systematic testing of:
        - Lossy: 60%, 100% compression levels (engine-specific mapping)
        - Colors: 32, 128 palette sizes
        """
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
    
    def _analyze_and_eliminate(self, results_df: pd.DataFrame, threshold: float) -> EliminationResult:
        """Analyze results using comprehensive quality metrics and eliminate underperforming pipelines."""
        elimination_result = EliminationResult()
        
        # Filter out failed jobs for analysis
        if 'success' in results_df.columns:
            successful_results = results_df[results_df['success'] == True].copy()
        else:
            # If no success column, assume all results are successful
            successful_results = results_df.copy()
        
        if successful_results.empty:
            self.logger.warning("No successful pipeline results to analyze")
            return elimination_result
        
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
            elimination_result.content_type_winners[content_type] = all_content_winners
            
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
        
        elimination_result.performance_matrix = performance_matrix
        
        # Find pipelines that never win in any content type or criteria
        all_winners = set()
        for winners in elimination_result.content_type_winners.values():
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
            
            if 'composite_quality' in pipeline_results.columns:
                avg_composite_quality = pipeline_results['composite_quality'].mean()
                max_composite_quality = pipeline_results['composite_quality'].max()
                
                if pd.isna(avg_composite_quality) or pd.isna(max_composite_quality):
                    self.logger.warning(f"Invalid composite_quality metrics for pipeline: {pipeline_id}")
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
            if 'composite_quality' not in pipeline_results.columns and 'ssim_mean' not in pipeline_results.columns:
                numeric_cols = pipeline_results.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    primary_metric = pipeline_results[numeric_cols[0]].mean()
                    if pd.isna(primary_metric) or primary_metric < threshold:
                        should_eliminate = True
            
            if should_eliminate:
                underperformers.add(pipeline_id)
        
        elimination_result.eliminated_pipelines = never_winners.union(underperformers)
        elimination_result.retained_pipelines = all_winners - underperformers
        
        # Add detailed elimination reasons
        for pipeline in elimination_result.eliminated_pipelines:
            if pipeline in never_winners:
                elimination_result.elimination_reasons[pipeline] = "Never achieved top-3 performance in any content type or criteria"
            elif pipeline in underperformers:
                pipeline_results = successful_results[successful_results['pipeline_id'] == pipeline]
                if not pipeline_results.empty:
                    # Use available quality metrics for elimination reason
                    if 'composite_quality' in pipeline_results.columns:
                        avg_quality = pipeline_results['composite_quality'].mean()
                        if not pd.isna(avg_quality):
                            elimination_result.elimination_reasons[pipeline] = f"Consistently poor performance (avg composite quality: {avg_quality:.3f})"
                        else:
                            elimination_result.elimination_reasons[pipeline] = "Invalid/missing composite quality metrics"
                    elif 'ssim_mean' in pipeline_results.columns:
                        avg_ssim = pipeline_results['ssim_mean'].mean()
                        if not pd.isna(avg_ssim):
                            elimination_result.elimination_reasons[pipeline] = f"Consistently poor performance (avg SSIM: {avg_ssim:.3f})"
                        else:
                            elimination_result.elimination_reasons[pipeline] = "Invalid/missing SSIM metrics"
                    else:
                        elimination_result.elimination_reasons[pipeline] = "Poor performance across available metrics"
                else:
                    elimination_result.elimination_reasons[pipeline] = "No successful test results"
        
        # Pareto frontier analysis for quality-aligned comparison
        self.logger.info("Running Pareto frontier analysis...")
        try:
            pareto_analyzer = ParetoAnalyzer(successful_results, self.logger)
            pareto_analysis = pareto_analyzer.generate_comprehensive_pareto_analysis()
            elimination_result.pareto_analysis = pareto_analysis
            
            # Extract dominated pipelines from Pareto analysis
            all_dominated = set()
            for content_type, frontier_data in pareto_analysis['content_type_frontiers'].items():
                all_dominated.update(frontier_data.get('dominated_pipelines', []))
            
            elimination_result.pareto_dominated_pipelines = all_dominated
            
            # Extract quality-aligned rankings
            elimination_result.quality_aligned_rankings = pareto_analysis.get('efficiency_rankings', {})
            
            # Update elimination reasons for Pareto-dominated pipelines
            for pipeline in all_dominated:
                if pipeline not in elimination_result.elimination_reasons:
                    elimination_result.elimination_reasons[pipeline] = "Pareto dominated (always better alternatives available)"
            
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
            elimination_result.pareto_analysis = {}
            elimination_result.pareto_dominated_pipelines = set()
            elimination_result.quality_aligned_rankings = {}
        
        # Log elimination statistics
        self.logger.info(f"Analysis complete:")
        self.logger.info(f"  - Total pipelines tested: {len(all_pipelines)}")
        self.logger.info(f"  - Eliminated: {len(elimination_result.eliminated_pipelines)}")
        self.logger.info(f"  - Retained: {len(elimination_result.retained_pipelines)}")
        
        return elimination_result
    
    def _save_results(self, elimination_result: EliminationResult, results_df: pd.DataFrame):
        """Save elimination analysis results."""
        failed_results, successful_results = self._validate_and_separate_results(results_df)
        
        self._save_csv_results(results_df)
        
        if not failed_results.empty:
            self._save_failed_pipelines_log(failed_results)
        
        self._save_elimination_summary(elimination_result, results_df, failed_results, successful_results)
        self._save_pareto_analysis_results(elimination_result)
        self._generate_and_save_failure_report(results_df)
        self._log_results_summary(elimination_result, failed_results, results_df)

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
    
    def _save_failed_pipelines_log(self, failed_results: pd.DataFrame):
        """Create and save detailed failed pipelines log."""
        failed_pipeline_log = []
        for _, row in failed_results.iterrows():
            failed_entry = {
                'timestamp': datetime.now().isoformat(),
                'gif_name': row.get('gif_name', 'unknown'),
                'content_type': row.get('content_type', 'unknown'),
                'pipeline_id': row.get('pipeline_id', 'unknown'),
                'error_message': row.get('error', 'No error message'),
                'pipeline_steps': row.get('pipeline_steps', []),
                'tools_used': row.get('tools_used', []),
                'test_parameters': {
                    'colors': row.get('test_colors', None),
                    'lossy': row.get('test_lossy', None),
                    'frame_ratio': row.get('test_frame_ratio', None)
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

    def _save_elimination_summary(self, elimination_result: EliminationResult, results_df: pd.DataFrame, 
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
            'eliminated_count': len(elimination_result.eliminated_pipelines),
            'retained_count': len(elimination_result.retained_pipelines),
            'eliminated_pipelines': list(elimination_result.eliminated_pipelines),
            'retained_pipelines': list(elimination_result.retained_pipelines),
            'content_type_winners': elimination_result.content_type_winners,
            'elimination_reasons': elimination_result.elimination_reasons,
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

    def _log_results_summary(self, elimination_result: EliminationResult, failed_results: pd.DataFrame, results_df: pd.DataFrame):
        """Log comprehensive results summary."""
        self.logger.info(f"Eliminated {len(elimination_result.eliminated_pipelines)} underperforming pipelines")
        self.logger.info(f"Retained {len(elimination_result.retained_pipelines)} competitive pipelines")
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
        report_lines.append("   1. Run 'giflab view-failures elimination_results/' for detailed error analysis")
        report_lines.append("   2. Consider excluding problematic tools from production pipelines")
        report_lines.append("   3. Test individual tools with 'python -c \"from giflab.system_tools import get_available_tools; print(get_available_tools())\"'")
        report_lines.append("   4. Review and update tool configurations in src/giflab/config.py")
        report_lines.append("   5. Consider running elimination analysis with fewer tool combinations to isolate issues")
        report_lines.append("")
        
        report_lines.append("📖 For detailed failure logs, see: elimination_results/failed_pipelines.json")
        
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
    
    def _save_pareto_analysis_results(self, elimination_result: EliminationResult):
        """Save Pareto frontier analysis results to files."""
        try:
            pareto_analysis = elimination_result.pareto_analysis
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
            dominated = elimination_result.pareto_dominated_pipelines
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
            self._generate_pareto_report(elimination_result)
            
        except Exception as e:
            self.logger.warning(f"Failed to save Pareto analysis results: {e}")
    
    def _generate_pareto_report(self, elimination_result: EliminationResult):
        """Generate a human-readable Pareto analysis report."""
        try:
            pareto_analysis = elimination_result.pareto_analysis
            report_lines = []
            
            report_lines.append("# 🎯 Pareto Frontier Analysis Report")
            report_lines.append("")
            report_lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Executive Summary
            global_frontier = pareto_analysis.get('global_frontier', {})
            frontier_count = len(global_frontier.get('frontier_points', []))
            dominated_count = len(elimination_result.pareto_dominated_pipelines)
            
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