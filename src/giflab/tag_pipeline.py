"""Comprehensive tagging pipeline for adding content analysis to existing compression results.

This pipeline adds 25 continuous scores (0.0-1.0) to compression results:
- 6 content classification scores (CLIP)  
- 4 quality/artifact assessment scores (Classical CV)
- 5 technical characteristic scores (Classical CV)
- 10 temporal motion analysis scores (Classical CV)

CRITICAL: Tagging runs ONCE on original GIFs only, scores inherited by all variants.
"""

import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import csv

import pandas as pd
from tqdm import tqdm

from .tagger import HybridCompressionTagger, TaggingResult
from .io import read_csv_as_dicts, setup_logging


class TaggingPipeline:
    """Pipeline for adding comprehensive tagging scores to existing compression results.
    
    Adds 25 continuous scores per original GIF:
    - Content classification (CLIP): 6 confidence scores  
    - Quality assessment (Classical CV): 4 artifact scores
    - Technical characteristics (Classical CV): 5 feature scores
    - Temporal motion analysis (Classical CV): 10 motion scores
    """
    
    # Define the 25 tagging columns that will be added to CSV
    TAGGING_COLUMNS = [
        # Content classification (CLIP) - 6 columns
        'screen_capture_confidence', 'vector_art_confidence', 'photography_confidence',
        'hand_drawn_confidence', '3d_rendered_confidence', 'pixel_art_confidence',
        
        # Quality assessment (Classical CV) - 4 columns  
        'blocking_artifacts', 'ringing_artifacts', 'quantization_noise', 'overall_quality',
        
        # Technical characteristics (Classical CV) - 5 columns
        'text_density', 'edge_density', 'color_complexity', 'contrast_score', 'gradient_smoothness',
        
        # Temporal motion analysis (Classical CV) - 10 columns
        'frame_similarity', 'motion_intensity', 'motion_smoothness', 'static_region_ratio',
        'scene_change_frequency', 'fade_transition_presence', 'cut_sharpness', 
        'temporal_entropy', 'loop_detection_confidence', 'motion_complexity'
    ]
    
    def __init__(self, workers: int = 1):
        """Initialize the comprehensive tagging pipeline.
        
        Args:
            workers: Number of worker processes for parallel tagging
        """
        self.tagger = HybridCompressionTagger()
        self.workers = max(1, workers)  
        self.logger = setup_logging(Path("logs"))
        
        self.logger.info(f"Initialized HybridCompressionTagger with {self.workers} workers")
        self.logger.info(f"Will add {len(self.TAGGING_COLUMNS)} tagging columns to CSV")
    
    def load_existing_results(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load existing compression results from CSV.
        
        Args:
            csv_path: Path to compression results CSV
            
        Returns:
            List of result dictionaries
            
        Raises:
            IOError: If CSV cannot be read
        """
        try:
            results = read_csv_as_dicts(csv_path)
            self.logger.info(f"Loaded {len(results)} existing results from {csv_path}")
            
            # Validate expected columns exist
            if results:
                required_cols = {'gif_sha', 'orig_filename', 'engine'}
                missing_cols = required_cols - set(results[0].keys())
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to load CSV {csv_path}: {e}")
            raise
    
    def identify_unique_gifs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify unique original GIFs from compression results.
        
        CRITICAL: Only select original engine records (engine='original'), 
        not compressed variants.
        
        Args:
            results: List of compression result dictionaries
            
        Returns:
            List of unique original GIF records (one per gif_sha)
        """
        original_gifs = {}
        
        for result in results:
            gif_sha = result.get("gif_sha")
            engine = result.get("engine", "").lower()
            
            # Only process original GIFs, not compressed variants
            if gif_sha and engine == "original" and gif_sha not in original_gifs:
                original_gifs[gif_sha] = result
        
        unique_list = list(original_gifs.values())
        self.logger.info(f"Found {len(unique_list)} unique original GIFs to tag")
        
        if not unique_list:
            self.logger.warning("No original GIFs found (engine='original'). Tagging requires original records.")
        
        return unique_list
    
    def find_original_gif_path(self, result: Dict[str, Any], raw_dir: Path) -> Optional[Path]:
        """Find the original GIF file path for a result record.
        
        Args:
            result: Original compression result dictionary
            raw_dir: Directory containing original GIF files
            
        Returns:
            Path to original GIF file, or None if not found
        """
        orig_filename = result.get("orig_filename")
        if not orig_filename:
            return None
        
        gif_path = raw_dir / orig_filename
        if gif_path.exists() and gif_path.is_file():
            return gif_path
        
        # Try case-insensitive search
        try:
            for file_path in raw_dir.iterdir():
                if file_path.is_file() and file_path.name.lower() == orig_filename.lower():
                    return file_path
        except Exception:
            pass
        
        return None
    
    def tag_single_gif(self, gif_path: Path, gif_sha: str) -> TaggingResult:
        """Generate comprehensive tagging scores for a single GIF file.
        
        Args:
            gif_path: Path to GIF file
            gif_sha: SHA hash of the GIF
            
        Returns:
            TaggingResult with 25 continuous scores
            
        Raises:
            RuntimeError: If tagging fails
        """
        try:
            return self.tagger.tag_gif(gif_path, gif_sha=gif_sha)
        except Exception as e:
            self.logger.error(f"Failed to tag {gif_path}: {e}")
            raise RuntimeError(f"Tagging failed for {gif_path}: {e}")
    
    def update_results_with_tags(
        self,
        results: List[Dict[str, Any]], 
        tagging_results: Dict[str, TaggingResult]
    ) -> List[Dict[str, Any]]:
        """Update compression results with generated tagging scores.
        
        CRITICAL: Tagging scores are inherited by ALL variants of the same gif_sha.
        
        Args:
            results: Original compression results
            tagging_results: Dictionary mapping gif_sha to TaggingResult
            
        Returns:
            Updated results with 25 tagging columns added
        """
        updated_results = []
        
        for result in results:
            gif_sha = result.get("gif_sha")
            result_copy = result.copy()
            
            if gif_sha in tagging_results:
                # Add all 25 tagging scores
                tagging_result = tagging_results[gif_sha]
                for column in self.TAGGING_COLUMNS:
                    score = tagging_result.scores.get(column, 0.0)
                    result_copy[column] = f"{score:.6f}"  # High precision for ML use
            else:
                # Add empty scores if tagging failed
                for column in self.TAGGING_COLUMNS:
                    result_copy[column] = "0.000000"
            
            updated_results.append(result_copy)
        
        return updated_results
    
    def write_tagged_csv(
        self, 
        updated_results: List[Dict[str, Any]], 
        output_csv_path: Path
    ) -> None:
        """Write updated results with tagging scores to new CSV file.
        
        Args:
            updated_results: Results with tagging scores added
            output_csv_path: Path for output CSV file
            
        Raises:
            IOError: If CSV cannot be written
        """
        if not updated_results:
            raise ValueError("No results to write")
        
        # Determine fieldnames - original columns + tagging columns
        original_fieldnames = [k for k in updated_results[0].keys() if k not in self.TAGGING_COLUMNS]
        fieldnames = original_fieldnames + self.TAGGING_COLUMNS
        
        try:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in updated_results:
                    writer.writerow(result)
            
            self.logger.info(f"Wrote {len(updated_results)} results to {output_csv_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write CSV {output_csv_path}: {e}")
            raise
    
    def run(
        self,
        csv_path: Path,
        raw_dir: Path,
        output_csv_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run the complete comprehensive tagging pipeline.
        
        Args:
            csv_path: Path to existing compression results CSV
            raw_dir: Directory containing original GIF files
            output_csv_path: Path for output CSV with tags (defaults to timestamped)
            
        Returns:
            Dictionary with tagging pipeline statistics
            
        Raises:
            IOError: If files cannot be read/written
            RuntimeError: If tagging process fails
        """
        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_path = csv_path.parent / f"results_tagged_{timestamp}.csv"
        
        self.logger.info(f"Starting comprehensive tagging pipeline")
        self.logger.info(f"Input CSV: {csv_path}")
        self.logger.info(f"Raw GIFs directory: {raw_dir}")
        self.logger.info(f"Output CSV: {output_csv_path}")
        
        # Load existing results
        results = self.load_existing_results(csv_path)
        if not results:
            return {"status": "no_results", "tagged": 0}
        
        # Find unique original GIFs to tag
        original_gifs = self.identify_unique_gifs(results)
        if not original_gifs:
            return {"status": "no_original_gifs", "tagged": 0}
        
        # Generate comprehensive tags for each unique original GIF
        tagging_results = {}
        tagged_count = 0
        failed_count = 0
        
        self.logger.info(f"Processing {len(original_gifs)} original GIFs...")
        
        # Process GIFs with progress bar
        for gif_record in tqdm(original_gifs, desc="Tagging GIFs", unit="gif"):
            gif_sha = gif_record["gif_sha"]
            
            # Find original GIF path
            gif_path = self.find_original_gif_path(gif_record, raw_dir)
            if not gif_path:
                self.logger.warning(f"Could not find original GIF for {gif_sha}: {gif_record.get('orig_filename', 'unknown')}")
                failed_count += 1
                continue
            
            try:
                # Generate comprehensive tagging scores
                tagging_result = self.tag_single_gif(gif_path, gif_sha)
                tagging_results[gif_sha] = tagging_result
                tagged_count += 1
                
                # Log key scores for monitoring
                content_classification = tagging_result.content_classification
                if content_classification:
                    content_type = max(content_classification.items(), key=lambda x: x[1])
                    content_type_str = f"{content_type[0]}={content_type[1]:.3f}"
                else:
                    content_type_str = "no_content_classification"
                
                motion_intensity = tagging_result.scores.get('motion_intensity', 0)
                overall_quality = tagging_result.scores.get('overall_quality', 0)
                
                self.logger.info(
                    f"Tagged {gif_path.name}: {content_type_str}, "
                    f"motion={motion_intensity:.3f}, quality={overall_quality:.3f}, "
                    f"time={tagging_result.processing_time_ms}ms"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to tag {gif_path}: {e}")
                failed_count += 1
        
        if tagged_count == 0:
            return {
                "status": "no_successful_tags",
                "total_results": len(results),
                "original_gifs": len(original_gifs),
                "tagged_successfully": 0,
                "tagging_failures": failed_count,
            }
        
        # Update all results with tagging scores (inheritance)
        self.logger.info("Updating all compression results with inherited tagging scores...")
        updated_results = self.update_results_with_tags(results, tagging_results)
        
        # Write updated results to new CSV
        self.logger.info("Writing tagged results to CSV...")
        self.write_tagged_csv(updated_results, output_csv_path)
        
        return {
            "status": "completed",
            "total_results": len(results),
            "original_gifs": len(original_gifs), 
            "tagged_successfully": tagged_count,
            "tagging_failures": failed_count,
            "output_path": str(output_csv_path),
            "tagging_columns_added": len(self.TAGGING_COLUMNS)
        }


def create_tagging_pipeline(workers: int = 1) -> TaggingPipeline:
    """Factory function to create a comprehensive tagging pipeline.
    
    Args:
        workers: Number of worker processes
        
    Returns:
        Configured TaggingPipeline instance
    """
    return TaggingPipeline(workers=workers)


def validate_tagged_csv(csv_path: Path) -> Dict[str, Any]:
    """Validate that a CSV contains the expected tagging columns.
    
    Args:
        csv_path: Path to tagged CSV file
        
    Returns:
        Validation report dictionary
    """
    try:
        # Load first few rows to check structure
        df = pd.read_csv(csv_path, nrows=5)
        
        # Check for tagging columns
        missing_columns = []
        present_columns = []
        
        for col in TaggingPipeline.TAGGING_COLUMNS:
            if col in df.columns:
                present_columns.append(col)
            else:
                missing_columns.append(col)
        
        return {
            "valid": not missing_columns,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "tagging_columns_present": len(present_columns),
            "tagging_columns_missing": len(missing_columns),
            "missing_columns": missing_columns,
            "csv_path": str(csv_path)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "csv_path": str(csv_path)
        } 