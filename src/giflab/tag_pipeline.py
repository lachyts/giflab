"""AI tagging pipeline for adding content tags to existing compression results."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .tagger import GifTagger, TaggingResult
from .io import read_csv_as_dicts, append_csv_row, setup_logging


class TaggingPipeline:
    """Pipeline for adding AI-generated tags to existing compression results."""
    
    def __init__(
        self,
        model_name: str = "default",
        workers: int = 1,
        max_tags_per_gif: int = 5
    ):
        """Initialize the tagging pipeline.
        
        Args:
            model_name: AI model to use for tagging
            workers: Number of worker processes for parallel tagging
            max_tags_per_gif: Maximum tags to generate per GIF
        """
        self.tagger = GifTagger(model_name)
        self.workers = workers  
        self.max_tags_per_gif = max_tags_per_gif
        self.logger = setup_logging(Path("logs"))
    
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
            return results
        except Exception as e:
            self.logger.error(f"Failed to load CSV {csv_path}: {e}")
            raise
    
    def identify_unique_gifs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify unique GIFs from compression results (one per gif_sha).
        
        Args:
            results: List of compression result dictionaries
            
        Returns:
            List of unique GIF records (one per gif_sha)
        """
        unique_gifs = {}
        
        for result in results:
            gif_sha = result.get("gif_sha")
            if gif_sha and gif_sha not in unique_gifs:
                unique_gifs[gif_sha] = result
        
        unique_list = list(unique_gifs.values())
        self.logger.info(f"Found {len(unique_list)} unique GIFs to tag")
        return unique_list
    
    def find_original_gif_path(self, result: Dict[str, Any], raw_dir: Path) -> Optional[Path]:
        """Find the original GIF file path for a result record.
        
        Args:
            result: Compression result dictionary
            raw_dir: Directory containing original GIF files
            
        Returns:
            Path to original GIF file, or None if not found
        """
        orig_filename = result.get("orig_filename")
        if not orig_filename:
            return None
        
        gif_path = raw_dir / orig_filename
        if gif_path.exists():
            return gif_path
        
        # Try case-insensitive search
        for file_path in raw_dir.iterdir():
            if file_path.name.lower() == orig_filename.lower():
                return file_path
        
        return None
    
    def tag_single_gif(self, gif_path: Path) -> TaggingResult:
        """Generate tags for a single GIF file.
        
        Args:
            gif_path: Path to GIF file
            
        Returns:
            Tagging result with generated tags
            
        Raises:
            RuntimeError: If tagging fails
        """
        try:
            return self.tagger.tag_gif(gif_path, max_tags=self.max_tags_per_gif)
        except Exception as e:
            self.logger.error(f"Failed to tag {gif_path}: {e}")
            raise RuntimeError(f"Tagging failed for {gif_path}: {e}")
    
    def update_results_with_tags(
        self,
        results: List[Dict[str, Any]], 
        tagging_results: Dict[str, TaggingResult]
    ) -> List[Dict[str, Any]]:
        """Update compression results with generated tags.
        
        Args:
            results: Original compression results
            tagging_results: Dictionary mapping gif_sha to TaggingResult
            
        Returns:
            Updated results with tags column
        """
        updated_results = []
        
        for result in results:
            gif_sha = result.get("gif_sha")
            result_copy = result.copy()
            
            if gif_sha in tagging_results:
                tagging_result = tagging_results[gif_sha]
                # Join tags with semicolon separator
                result_copy["tags"] = ";".join(tagging_result.tags)
            else:
                result_copy["tags"] = ""
            
            updated_results.append(result_copy)
        
        return updated_results
    
    def run(
        self,
        csv_path: Path,
        raw_dir: Path,
        output_csv_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run the complete tagging pipeline.
        
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
        
        self.logger.info(f"Starting tagging pipeline: {csv_path} -> {output_csv_path}")
        
        # Load existing results
        results = self.load_existing_results(csv_path)
        if not results:
            return {"status": "no_results", "tagged": 0}
        
        # Find unique GIFs to tag
        unique_gifs = self.identify_unique_gifs(results)
        if not unique_gifs:
            return {"status": "no_unique_gifs", "tagged": 0}
        
        # Generate tags for each unique GIF
        tagging_results = {}
        tagged_count = 0
        failed_count = 0
        
        for gif_record in unique_gifs:
            gif_sha = gif_record["gif_sha"]
            
            # Find original GIF path
            gif_path = self.find_original_gif_path(gif_record, raw_dir)
            if not gif_path:
                self.logger.warning(f"Could not find original GIF for {gif_sha}")
                failed_count += 1
                continue
            
            try:
                # Generate tags
                tagging_result = self.tag_single_gif(gif_path)
                tagging_results[gif_sha] = tagging_result
                tagged_count += 1
                
                self.logger.info(f"Tagged {gif_path.name}: {';'.join(tagging_result.tags)}")
                
            except Exception as e:
                self.logger.error(f"Failed to tag {gif_path}: {e}")
                failed_count += 1
        
        # Update all results with tags
        updated_results = self.update_results_with_tags(results, tagging_results)
        
        # Write updated results to new CSV
        fieldnames = list(results[0].keys()) if results else []
        if "tags" not in fieldnames:
            fieldnames.append("tags")
        
        # TODO: Write all updated results to CSV
        # This will be implemented in Stage 9 (S9)
        self.logger.info(f"Would write {len(updated_results)} results to {output_csv_path}")
        
        return {
            "status": "completed",
            "total_results": len(results),
            "unique_gifs": len(unique_gifs),
            "tagged_successfully": tagged_count,
            "tagging_failures": failed_count,
            "output_path": str(output_csv_path)
        }


def create_tagging_pipeline(
    model_name: str = "default",
    workers: int = 1,
    max_tags: int = 5
) -> TaggingPipeline:
    """Factory function to create a tagging pipeline.
    
    Args:
        model_name: AI model to use for tagging
        workers: Number of worker processes
        max_tags: Maximum tags per GIF
        
    Returns:
        Configured TaggingPipeline instance
    """
    return TaggingPipeline(
        model_name=model_name,
        workers=workers,
        max_tags_per_gif=max_tags
    ) 