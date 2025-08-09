#!/usr/bin/env python3
"""
Example usage of GifLab's source tracking features.

This script demonstrates how to use the new source tracking functionality
to collect GIFs from different sources and maintain metadata about their origin.
"""

from pathlib import Path

from src.giflab.directory_source_detection import (
    create_directory_structure,
    detect_source_from_directory,
)
from src.giflab.meta import extract_gif_metadata
from src.giflab.source_tracking import (
    create_animately_metadata,
    create_tenor_metadata,
    create_tgif_metadata,
)


def example_tenor_collection():
    """Example of collecting GIFs from Tenor with source tracking."""

    # Simulate collecting a GIF from Tenor
    gif_path = Path("data/raw/love_gif.gif")

    if gif_path.exists():
        # Create Tenor-specific metadata
        platform, metadata = create_tenor_metadata(
            query="love",
            collection_context="email_marketing_campaign",
            tenor_id="12345",
            popularity=0.85,
            search_rank=1,
        )

        # Extract GIF metadata with source tracking
        gif_metadata = extract_gif_metadata(
            gif_path, source_platform=platform, source_metadata=metadata
        )

        print("✅ Tenor GIF processed:")
        print(f"   Platform: {gif_metadata.source_platform}")
        print(f"   Metadata: {gif_metadata.source_metadata}")
        print(f"   Query: {gif_metadata.source_metadata.get('query', 'N/A')}")
        print(f"   Tenor ID: {gif_metadata.source_metadata.get('tenor_id', 'N/A')}")
        print()

        return gif_metadata
    else:
        print(f"⚠️  Example file not found: {gif_path}")
        return None


def example_animately_upload():
    """Example of processing user uploads from Animately platform."""

    # Simulate processing a user upload
    gif_path = Path("data/raw/user_upload.gif")

    if gif_path.exists():
        # Create Animately metadata
        platform, metadata = create_animately_metadata(
            user_id="user_456",
            upload_intent="compression",
            original_size_kb=1024.5,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        )

        # Extract GIF metadata with source tracking
        gif_metadata = extract_gif_metadata(
            gif_path, source_platform=platform, source_metadata=metadata
        )

        print("✅ Animately Upload processed:")
        print(f"   Platform: {gif_metadata.source_platform}")
        print(f"   User ID: {gif_metadata.source_metadata.get('user_id', 'N/A')}")
        print(f"   Intent: {gif_metadata.source_metadata.get('upload_intent', 'N/A')}")
        print(
            f"   Original Size: {gif_metadata.source_metadata.get('original_size_kb', 'N/A')} KB"
        )
        print()

        return gif_metadata
    else:
        print(f"⚠️  Example file not found: {gif_path}")
        return None


def example_tgif_dataset():
    """Example of processing GIFs from the TGIF dataset."""

    # Simulate processing a TGIF dataset GIF
    gif_path = Path("data/raw/tgif_sample.gif")

    if gif_path.exists():
        # Create TGIF dataset metadata
        platform, metadata = create_tgif_metadata(
            tgif_id="tgif_001", description="a man is dancing", category="human_action"
        )

        # Extract GIF metadata with source tracking
        gif_metadata = extract_gif_metadata(
            gif_path, source_platform=platform, source_metadata=metadata
        )

        print("✅ TGIF Dataset GIF processed:")
        print(f"   Platform: {gif_metadata.source_platform}")
        print(f"   TGIF ID: {gif_metadata.source_metadata.get('tgif_id', 'N/A')}")
        print(
            f"   Description: {gif_metadata.source_metadata.get('description', 'N/A')}"
        )
        print(f"   Category: {gif_metadata.source_metadata.get('category', 'N/A')}")
        print()

        return gif_metadata
    else:
        print(f"⚠️  Example file not found: {gif_path}")
        return None


def example_directory_based_detection():
    """Example of using directory-based source detection."""

    print("🗂️ Directory-Based Source Detection Example")
    print("=" * 50)

    # Create a temporary directory structure for demonstration
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"

        # Create directory structure
        print("📁 Creating directory structure...")
        create_directory_structure(raw_dir)

        # Create some example GIF files
        test_files = [
            raw_dir / "tenor" / "love" / "cute_cat.gif",
            raw_dir / "tenor" / "marketing" / "business_chart.gif",
            raw_dir / "animately" / "user_uploads" / "animation.gif",
            raw_dir / "tgif_dataset" / "human_action" / "dancing.gif",
            raw_dir / "unknown" / "misc.gif",
        ]

        print("📄 Creating test GIF files...")
        for file_path in test_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()  # Create empty file for demo

        # Test directory detection
        print("🔍 Testing directory-based detection:")
        for file_path in test_files:
            platform, metadata = detect_source_from_directory(file_path, raw_dir)
            print(f"   {file_path.relative_to(raw_dir)}")
            print(f"      Platform: {platform}")
            print(f"      Metadata: {metadata}")
            print()

        print("✅ Directory-based detection working correctly!")
        print()
        print("💡 To use this in your pipeline:")
        print("   1. Run: python -m giflab organize-directories data/raw/")
        print("   2. Move your GIFs to the appropriate directories")
        print("   3. Run: python -m giflab run data/raw/")
        print("   4. Source will be automatically detected from directory structure")


def example_mixed_dataset_analysis():
    """Example of analyzing a mixed dataset with different sources."""

    print("🔍 Mixed Dataset Analysis Example")
    print("=" * 50)

    # Collect GIFs from different sources
    all_gifs = []

    # Process different sources
    tenor_gif = example_tenor_collection()
    if tenor_gif:
        all_gifs.append(tenor_gif)

    upload_gif = example_animately_upload()
    if upload_gif:
        all_gifs.append(upload_gif)

    tgif_gif = example_tgif_dataset()
    if tgif_gif:
        all_gifs.append(tgif_gif)

    # Analyze the collection
    if all_gifs:
        print("📊 Dataset Analysis:")
        print(f"   Total GIFs: {len(all_gifs)}")

        # Group by source platform
        platform_counts = {}
        for gif in all_gifs:
            platform = gif.source_platform
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        print("   Source Distribution:")
        for platform, count in platform_counts.items():
            print(f"     {platform}: {count} GIFs")

        # Show search queries
        queries = set()
        for gif in all_gifs:
            if gif.source_metadata and "query" in gif.source_metadata:
                queries.add(gif.source_metadata["query"])

        if queries:
            print(f"   Search Queries: {', '.join(sorted(queries))}")

        print()
        print("💡 This metadata will help you analyze:")
        print("   • Platform-specific compression performance")
        print("   • Query-specific content characteristics")
        print("   • User upload patterns vs. searched content")
        print("   • Dataset bias detection and correction")

    else:
        print("⚠️  No GIFs found. Add some test GIFs to data/raw/ to run this example.")


def example_csv_output():
    """Example of how the CSV output will look with source tracking."""

    print("📄 CSV Output Example")
    print("=" * 50)

    # Example CSV row with source tracking
    csv_example = {
        "gif_sha": "abc123def456...",
        "orig_filename": "love_animation.gif",
        "engine": "gifsicle",
        "lossy": 40,
        "frame_keep_ratio": 0.8,
        "color_keep_count": 64,
        "kilobytes": 245.3,
        "ssim": 0.936,
        "render_ms": 1250,
        "orig_kilobytes": 612.1,
        "orig_width": 480,
        "orig_height": 270,
        "orig_frames": 24,
        "orig_fps": 24.0,
        "orig_n_colors": 156,
        "entropy": 4.2,
        "source_platform": "tenor",
        "source_metadata": '{"query":"love","collection_context":"email_marketing","tenor_id":"12345","popularity":0.85,"detected_from":"directory","collected_at":"2024-01-15T10:30:00Z"}',
        "timestamp": "2024-01-15T10:30:15Z",
    }

    print("Sample CSV row with source tracking:")
    for key, value in csv_example.items():
        print(f"   {key}: {value}")

    print()
    print("📊 SQL Analysis Examples:")
    print("   # Platform performance comparison")
    print(
        "   SELECT source_platform, AVG(kilobytes) as avg_size, AVG(ssim) as avg_quality"
    )
    print("   FROM results GROUP BY source_platform;")
    print()
    print("   # Query-specific analysis")
    print(
        "   SELECT JSON_EXTRACT(source_metadata, '$.query') as query, COUNT(*) as count"
    )
    print("   FROM results WHERE source_platform = 'tenor' GROUP BY query;")


if __name__ == "__main__":
    print("🎞️ GifLab Source Tracking Examples")
    print("=" * 50)
    print()

    example_directory_based_detection()
    print()
    example_mixed_dataset_analysis()
    print()
    example_csv_output()

    print()
    print("✅ Examples completed!")
    print("💡 Next steps:")
    print("   1. Add some test GIFs to data/raw/")
    print("   2. Run this script to see source tracking in action")
    print("   3. Use the helper functions in your data collection scripts")
    print("   4. Run the full pipeline to see source tracking in CSV output")
