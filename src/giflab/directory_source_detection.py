"""Directory-based source detection for GIF collection.

This module provides functionality to automatically detect GIF sources based on
their directory structure within the data/raw/ folder.

Directory Structure:
    data/raw/
    ├── tenor/           # GIFs from Tenor (search queries can be in metadata)
    ├── animately/       # GIFs from Animately platform uploads
    ├── tgif_dataset/    # GIFs from TGIF dataset
    └── unknown/         # Ungrouped GIFs (default)

The pipeline will automatically detect the source based on which subdirectory
a GIF is located in, making it easy to organize and manage collections.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .source_tracking import SourcePlatform


def detect_source_from_directory(gif_path: Path, raw_dir: Path) -> Tuple[str, Dict[str, Any] | None]:
    """Detect source platform and metadata from directory structure.
    
    Args:
        gif_path: Path to the GIF file
        raw_dir: Path to the raw data directory (e.g., data/raw/)
        
    Returns:
        Tuple of (source_platform, source_metadata)
        
    Examples:
        # data/raw/tenor/love_search/cute_cat.gif
        platform, metadata = detect_source_from_directory(gif_path, raw_dir)
        # Returns: ("tenor", {"query": "love_search", "detected_from": "directory"})
        
        # data/raw/animately/user_uploads/animation.gif  
        platform, metadata = detect_source_from_directory(gif_path, raw_dir)
        # Returns: ("animately", {"collection_context": "user_uploads", "detected_from": "directory"})
    """
    try:
        # Get the relative path from raw_dir to gif_path
        relative_path = gif_path.relative_to(raw_dir)
        path_parts = relative_path.parts
        
        if len(path_parts) == 1:
            # File is directly in raw_dir (e.g., data/raw/file.gif)
            return SourcePlatform.UNKNOWN, None
        
        # First directory indicates the source platform
        source_dir = path_parts[0].lower()
        
        # Map directory names to platforms
        platform_mapping = {
            "tenor": SourcePlatform.TENOR,
            "animately": SourcePlatform.ANIMATELY,
            "tgif_dataset": SourcePlatform.TGIF_DATASET,
            "tgif": SourcePlatform.TGIF_DATASET,  # Alternative name
            "unknown": SourcePlatform.UNKNOWN,
        }
        
        platform = platform_mapping.get(source_dir, SourcePlatform.UNKNOWN)
        
        # Create metadata based on directory structure
        metadata = {
            "detected_from": "directory",
            "directory_path": str(relative_path.parent),
        }
        
        # Add platform-specific metadata based on directory structure
        if platform == SourcePlatform.TENOR:
            metadata.update(_extract_tenor_metadata_from_path(path_parts))
        elif platform == SourcePlatform.ANIMATELY:
            metadata.update(_extract_animately_metadata_from_path(path_parts))
        elif platform == SourcePlatform.TGIF_DATASET:
            metadata.update(_extract_tgif_metadata_from_path(path_parts))
        
        return platform, metadata
        
    except ValueError:
        # gif_path is not within raw_dir
        return SourcePlatform.UNKNOWN, None


def _extract_tenor_metadata_from_path(path_parts: Tuple[str, ...]) -> Dict[str, Any]:
    """Extract Tenor-specific metadata from directory path.
    
    Expected structure: tenor/query_name/file.gif
    """
    metadata = {}
    
    if len(path_parts) >= 2:
        # Second part is the query/search term
        query_dir = path_parts[1]
        
        # Convert directory name to readable query
        # Replace underscores with spaces and decode common patterns
        query = query_dir.replace("_", " ").replace("-", " ")
        
        metadata["query"] = query
        
        # If there's a third level, it might be a collection context
        if len(path_parts) >= 3:
            metadata["collection_context"] = path_parts[2]
    
    return metadata


def _extract_animately_metadata_from_path(path_parts: Tuple[str, ...]) -> Dict[str, Any]:
    """Extract Animately-specific metadata from directory path.
    
    Expected structure: animately/context/file.gif
    """
    metadata = {}
    
    if len(path_parts) >= 2:
        # Second part is the collection context
        context_dir = path_parts[1]
        
        # Map common directory names to intents
        intent_mapping = {
            "uploads": "compression",
            "user_uploads": "compression",
            "test_data": "analysis",
            "samples": "analysis",
            "demos": "demonstration",
        }
        
        metadata["collection_context"] = context_dir
        metadata["upload_intent"] = intent_mapping.get(context_dir, "unknown")
    
    return metadata


def _extract_tgif_metadata_from_path(path_parts: Tuple[str, ...]) -> Dict[str, Any]:
    """Extract TGIF dataset metadata from directory path.
    
    Expected structure: tgif_dataset/category/file.gif
    """
    metadata = {}
    
    if len(path_parts) >= 2:
        # Second part is the category
        category_dir = path_parts[1]
        metadata["category"] = category_dir
    
    return metadata


def create_directory_structure(raw_dir: Path) -> None:
    """Create the recommended directory structure for source organization.
    
    Args:
        raw_dir: Path to the raw data directory
        
    Creates:
        raw_dir/tenor/
        raw_dir/animately/
        raw_dir/tgif_dataset/
        raw_dir/unknown/
    """
    directories = [
        raw_dir / "tenor",
        raw_dir / "animately", 
        raw_dir / "tgif_dataset",
        raw_dir / "unknown",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
        # Create a README file explaining the directory
        readme_path = directory / "README.md"
        if not readme_path.exists():
            readme_content = _generate_readme_content(directory.name)
            readme_path.write_text(readme_content)


def _generate_readme_content(directory_name: str) -> str:
    """Generate README content for source directories."""
    
    content_map = {
        "tenor": """# Tenor GIFs

This directory contains GIFs collected from Tenor.

## Organization

Organize GIFs by search query:
- `love/` - GIFs from "love" search
- `marketing/` - GIFs from "marketing" search
- `email_campaign/` - GIFs for email campaigns

## Metadata

GIFs in this directory will automatically have:
- `source_platform`: "tenor"
- `query`: Based on subdirectory name
- `collection_context`: Based on deeper subdirectory structure
""",
        "animately": """# Animately Platform GIFs

This directory contains GIFs uploaded to the Animately platform.

## Organization

Organize GIFs by context:
- `user_uploads/` - Regular user uploads
- `test_data/` - Test/development data
- `samples/` - Sample GIFs for demos

## Metadata

GIFs in this directory will automatically have:
- `source_platform`: "animately"
- `collection_context`: Based on subdirectory name
- `upload_intent`: Automatically inferred from context
""",
        "tgif_dataset": """# TGIF Dataset GIFs

This directory contains GIFs from the TGIF dataset.

## Organization

Organize GIFs by category:
- `human_action/` - Human activities
- `animal_action/` - Animal activities
- `object_motion/` - Object movements

## Metadata

GIFs in this directory will automatically have:
- `source_platform`: "tgif_dataset"
- `category`: Based on subdirectory name
""",
        "unknown": """# Unknown Source GIFs

This directory contains GIFs with unknown or unspecified sources.

## Usage

Place GIFs here when:
- Source is unknown
- Mixed sources that don't fit other categories
- Temporary storage before organization

## Metadata

GIFs in this directory will have:
- `source_platform`: "unknown"
- No additional metadata
""",
    }
    
    return content_map.get(directory_name, "# Source Directory\n\nGIFs organized by source.")


def get_directory_organization_help() -> str:
    """Get help text for directory organization."""
    
    return """
🗂️  Directory-Based Source Detection

GifLab automatically detects GIF sources based on directory structure:

📁 data/raw/
├── 📁 tenor/
│   ├── 📁 love/              # GIFs from "love" search
│   ├── 📁 marketing/         # GIFs from "marketing" search
│   └── 📁 email_campaign/    # GIFs for email campaigns
├── 📁 animately/
│   ├── 📁 user_uploads/      # Regular user uploads
│   ├── 📁 test_data/         # Test/development data
│   └── 📁 samples/           # Sample GIFs
├── 📁 tgif_dataset/
│   ├── 📁 human_action/      # Human activities
│   └── 📁 animal_action/     # Animal activities
└── 📁 unknown/               # Ungrouped GIFs

✨ Benefits:
• Automatic source detection
• Visual organization
• Easy to manage collections
• Survives pipeline restarts
• No manual metadata entry required

💡 Usage:
1. Run: giflab organize-directories data/raw/
2. Move GIFs to appropriate directories
3. Run: giflab run data/raw/ --detect-source-from-directory
""" 