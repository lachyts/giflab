# TGIF Dataset GIFs

This directory contains GIFs from the TGIF dataset.

## Organization

All GIFs go directly in this directory:
- No subdirectories needed - all files are from the same research dataset
- Content categories (human/animal/object) are better captured in metadata than directory structure
- Flat structure simplifies management and processing

## Metadata

GIFs in this directory will automatically have:
- `source_platform`: "tgif_dataset"
- `collection_context`: "research_dataset"
- Additional content categorization can be added via AI tagging

## Example
```
data/raw/tgif_dataset/
├── dancing_action.gif
├── cat_playing.gif
└── car_moving.gif
```
