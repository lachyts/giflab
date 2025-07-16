"""Source tracking utilities for GIF collection metadata."""

from datetime import datetime
from typing import Any


def create_source_metadata(
    platform: str,
    query: str | None = None,
    collection_context: str | None = None,
    **kwargs: Any
) -> tuple[str, dict[str, Any]]:
    """Create standardized source metadata for GIF collection.
    
    Args:
        platform: Platform identifier (e.g., "tenor", "animately", "tgif_dataset")
        query: Search query used (if any)
        collection_context: Broader collection context (e.g., "email_marketing_campaign")
        **kwargs: Platform-specific metadata
        
    Returns:
        Tuple of (platform, metadata_dict)
        
    Example:
        platform, metadata = create_source_metadata(
            "tenor",
            query="love",
            collection_context="email_marketing", 
            tenor_id="xyz123",
            popularity=0.85
        )
    """
    metadata = {
        "collected_at": datetime.now().isoformat(),
        **kwargs
    }
    
    if query:
        metadata["query"] = query
        
    if collection_context:
        metadata["collection_context"] = collection_context
    
    return platform, metadata


def create_tenor_metadata(
    query: str,
    collection_context: str | None = None,
    tenor_id: str | None = None,
    popularity: float | None = None,
    **kwargs: Any
) -> tuple[str, dict[str, Any]]:
    """Create standardized metadata for Tenor GIF collection.
    
    Args:
        query: Search query used on Tenor
        collection_context: Broader collection context
        tenor_id: Tenor's internal GIF ID
        popularity: Tenor popularity score (0.0-1.0)
        **kwargs: Additional Tenor-specific metadata
        
    Returns:
        Tuple of (platform, metadata_dict)
    """
    metadata_kwargs = {}
    
    if tenor_id:
        metadata_kwargs["tenor_id"] = tenor_id
    if popularity is not None:
        metadata_kwargs["popularity"] = popularity
    
    metadata_kwargs.update(kwargs)
    
    return create_source_metadata(
        platform="tenor",
        query=query,
        collection_context=collection_context,
        **metadata_kwargs
    )





def create_animately_metadata(
    user_id: str | None = None,
    upload_intent: str | None = None,
    original_size_kb: float | None = None,
    user_agent: str | None = None,
    **kwargs: Any
) -> tuple[str, dict[str, Any]]:
    """Create standardized metadata for Animately platform uploads.
    
    Args:
        user_id: User identifier (if available)
        upload_intent: Purpose of upload ("compression", "analysis", etc.)
        original_size_kb: Original file size before compression
        user_agent: Browser/client user agent
        **kwargs: Additional platform-specific metadata
        
    Returns:
        Tuple of (platform, metadata_dict)
    """
    metadata_kwargs = {}
    
    if user_id:
        metadata_kwargs["user_id"] = user_id
    if upload_intent:
        metadata_kwargs["upload_intent"] = upload_intent
    if original_size_kb is not None:
        metadata_kwargs["original_size_kb"] = original_size_kb
    if user_agent:
        metadata_kwargs["user_agent"] = user_agent
        
    metadata_kwargs.update(kwargs)
    
    return create_source_metadata(
        platform="animately",
        **metadata_kwargs
    )


def create_tgif_metadata(
    tgif_id: str | None = None,
    description: str | None = None,
    category: str | None = None,
    **kwargs: Any
) -> tuple[str, dict[str, Any]]:
    """Create standardized metadata for TGIF dataset.
    
    Args:
        tgif_id: TGIF dataset identifier
        description: Human-readable description from dataset
        category: Content category from dataset
        **kwargs: Additional TGIF-specific metadata
        
    Returns:
        Tuple of (platform, metadata_dict)
    """
    metadata_kwargs = {}
    
    if tgif_id:
        metadata_kwargs["tgif_id"] = tgif_id
    if description:
        metadata_kwargs["description"] = description
    if category:
        metadata_kwargs["category"] = category
        
    metadata_kwargs.update(kwargs)
    
    return create_source_metadata(
        platform="tgif_dataset",
        **metadata_kwargs
    )


# Common platform constants for easy reference
class SourcePlatform:
    """Constants for common source platforms."""
    TENOR = "tenor"
    ANIMATELY = "animately"
    TGIF_DATASET = "tgif_dataset"
    UNKNOWN = "unknown" 