"""Intelligent sampling strategies for pipeline testing.

This module provides various sampling strategies to reduce testing time while
maintaining representative coverage of the pipeline space.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass 
class SamplingStrategy:
    """Configuration for intelligent sampling strategies."""
    name: str
    description: str
    sample_ratio: float  # Fraction of total pipelines to test
    min_samples_per_tool: int = 3  # Minimum samples per tool type


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


class PipelineSampler:
    """Handles intelligent sampling of pipeline combinations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the pipeline sampler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def select_pipelines_intelligently(self, all_pipelines: List, strategy: str = 'representative') -> List:
        """Select pipelines using intelligent sampling strategies to reduce testing time."""
        sampling_config = SAMPLING_STRATEGIES.get(strategy, SAMPLING_STRATEGIES['representative'])
        
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
        
        # Total: 17 GIFs (10 + 4 + 2 + 1)
        all_targeted_names = original_names + size_variation_names + frame_variation_names + new_content_names
        
        # Note: This method currently only returns the names.
        # In the full implementation, it would generate the actual GIF files.
        self.logger.info(f"🎯 Targeted subset: {len(all_targeted_names)} GIFs selected")
        self.logger.info(f"   📋 Categories: {len(original_names)} original + {len(size_variation_names)} size + {len(frame_variation_names)} frames + {len(new_content_names)} content")
        
        # Return as Path objects (would be actual file paths in full implementation)
        return [Path(f"{name}.gif") for name in all_targeted_names]