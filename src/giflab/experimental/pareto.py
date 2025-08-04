"""Pareto frontier analysis for pipeline efficiency comparison.

This module provides advanced Pareto frontier analysis capabilities for comparing
pipeline efficiency across multiple dimensions (quality vs size trade-offs).
"""

import logging
from typing import Any, Dict, List

import pandas as pd


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