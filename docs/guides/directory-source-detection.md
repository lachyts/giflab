# 🗂️ Directory-Based Source Detection Guide

## Overview

GifLab automatically detects GIF sources based on directory structure, eliminating the need for manual metadata entry and providing a persistent, visual way to organize your GIF collections.

## Directory Structure

### Recommended Structure
```
data/raw/
├── tenor/              # GIFs from Tenor platform
│   ├── love/           # "love" search results
│   ├── marketing/      # "marketing" search results
│   ├── email_campaign/ # Email campaign GIFs
│   └── reactions/      # Reaction GIFs
├── animately/          # GIFs from Animately platform (all user uploads)
│   ├── user_upload_1.gif
│   ├── user_upload_2.gif
│   └── user_upload_3.gif
├── tgif_dataset/       # GIFs from TGIF research dataset
│   ├── research_gif_1.gif
│   ├── research_gif_2.gif
│   └── research_gif_3.gif
└── unknown/            # Ungrouped/unclassified GIFs
```

### Platform Naming Convention

| Platform | Directory Name | Type | Notes |
|----------|----------------|------|-------|
| Tenor | `tenor/` | Live Platform | Google's GIF search platform |
| Animately | `animately/` | Live Platform | Your compression platform |
| TGIF Dataset | `tgif_dataset/` | Research Dataset | Academic research dataset |
| Unknown | `unknown/` | Fallback | Unclassified or mixed sources |

**Why "tgif_dataset"?**
- **Clarity**: Makes it clear this is a research dataset, not a live platform
- **Distinction**: Separates research datasets from live platforms
- **Flexibility**: Both `tgif/` and `tgif_dataset/` are supported (alternative names)

**Why flat structure for some platforms?** Animately and TGIF files have uniform characteristics within each platform, so subdirectories don't add meaningful organization. Tenor search queries, however, create content differences worth preserving in the directory structure.

## Setup and Usage

### 1. Create Directory Structure

```bash
# Create the organized directory structure
python -m giflab organize-directories data/raw/

# This creates:
# - data/raw/tenor/
# - data/raw/animately/
# - data/raw/tgif_dataset/
# - data/raw/unknown/
# - README.md files in each directory
```

### 2. Organize Your GIFs

Move your GIFs to the appropriate directories:

```bash
# Example organization
data/raw/
├── tenor/
│   ├── love/
│   │   ├── heart_eyes.gif
│   │   ├── love_hearts.gif
│   │   └── romantic_kiss.gif
│   └── marketing/
│       ├── growth_chart.gif
│       ├── success_thumbs_up.gif
│       └── handshake.gif
├── animately/
│   ├── custom_animation.gif
│   ├── logo_animation.gif
│   ├── compression_test.gif
│   └── quality_reference.gif
└── tgif_dataset/
    ├── dancing.gif
    ├── running.gif
    ├── jumping.gif
    ├── cat_playing.gif
    ├── dog_running.gif
    └── bird_flying.gif
```

### 3. Run Analysis

```bash
# Run with automatic source detection (default)
python -m giflab run data/raw/

# Or explicitly enable source detection
python -m giflab run data/raw/ --detect-source-from-directory

# Disable source detection if needed
python -m giflab run data/raw/ --no-detect-source-from-directory
```

## Metadata Extraction

### Platform-Specific Metadata

#### Tenor Platform
- **Query**: Extracted from directory name (`tenor/love/` → `"love"`)
- **Collection Context**: Extracted from subdirectory (`tenor/love/reactions/` → `"reactions"`)

#### Animately Platform
- **Collection Context**: All files are `"user_uploads"`
- **Upload Intent**: All files are `"compression"`

#### TGIF Dataset
- **Collection Context**: All files are `"research_dataset"`
- **Content Categorization**: Added via AI tagging rather than directory structure

### Example Metadata Output

```json
// tenor/love/cute_cat.gif
{
  "detected_from": "directory",
  "directory_path": "tenor/love",
  "query": "love",
  "collection_context": "cute_cat.gif"
}

// animately/animation.gif
{
  "detected_from": "directory", 
  "directory_path": "animately",
  "collection_context": "user_uploads",
  "upload_intent": "compression"
}

// tgif_dataset/dancing.gif
{
  "detected_from": "directory",
  "directory_path": "tgif_dataset", 
  "collection_context": "research_dataset"
}
```

## CSV Output

### New Columns Added
- `source_platform`: Platform identifier (`tenor`, `animately`, `tgif_dataset`, `unknown`)
- `source_metadata`: JSON string containing platform-specific metadata

### Example CSV Output
```csv
gif_sha,orig_filename,source_platform,source_metadata,timestamp
abc123,heart_eyes.gif,tenor,"{\"query\":\"love\",\"detected_from\":\"directory\",\"directory_path\":\"tenor/love\"}",2024-01-15T10:30:15Z
def456,animation.gif,animately,"{\"collection_context\":\"user_uploads\",\"upload_intent\":\"compression\",\"detected_from\":\"directory\",\"directory_path\":\"animately\"}",2024-01-15T10:30:20Z
ghi789,dancing.gif,tgif_dataset,"{\"collection_context\":\"research_dataset\",\"detected_from\":\"directory\",\"directory_path\":\"tgif_dataset\"}",2024-01-15T10:30:25Z
```

## Analysis Examples

### Platform-Specific Analysis
```sql
-- Compare compression performance by platform
SELECT 
    source_platform,
    AVG(kilobytes) as avg_compressed_size,
    AVG(ssim) as avg_quality,
    COUNT(*) as gif_count
FROM results 
GROUP BY source_platform;
```

### Query-Specific Analysis
```sql
-- Analyze Tenor search results
SELECT 
    JSON_EXTRACT(source_metadata, '$.query') as search_query,
    AVG(orig_kilobytes) as avg_original_size,
    AVG(kilobytes) as avg_compressed_size,
    COUNT(*) as gif_count
FROM results 
WHERE source_platform = 'tenor'
GROUP BY JSON_EXTRACT(source_metadata, '$.query');
```

### Context-Specific Analysis
```sql
-- Analyze Animately upload patterns
SELECT 
    JSON_EXTRACT(source_metadata, '$.collection_context') as context,
    JSON_EXTRACT(source_metadata, '$.upload_intent') as intent,
    AVG(orig_kilobytes) as avg_size,
    COUNT(*) as gif_count
FROM results 
WHERE source_platform = 'animately'
GROUP BY context, intent;
```

## Benefits

### 1. **Automatic Source Detection**
- No manual metadata entry required
- Consistent classification across runs
- Reduces human error

### 2. **Visual Organization**
- Easy to see what you have
- Simple to reorganize collections
- Clear directory structure

### 3. **Persistent Structure**
- Survives pipeline restarts
- No data loss on interruption
- Easy backup and sync

### 4. **Flexible Analysis**
- Query by platform, search term, or context
- Rich metadata for ML training
- Easy bias detection

### 5. **Scalable Organization**
- Easy to add new platforms
- Simple to reorganize existing collections
- Supports complex hierarchies

## Best Practices

### 1. **Directory Naming**
- Use descriptive names for search queries
- Keep directory names filesystem-safe (no special characters)
- Use underscores or hyphens instead of spaces

### 2. **Organization Strategy**
- Group by platform first, then by context
- Use consistent naming conventions
- Document your organization scheme

### 3. **Maintenance**
- Regularly review and reorganize
- Remove empty directories
- Update README files as needed

### 4. **Backup**
- Directory structure is easy to backup
- Version control your organization
- Document changes and reasoning

## Alternative Approaches

### Manual Metadata (if needed)
```python
from src.giflab.source_tracking import create_tenor_metadata
from src.giflab.meta import extract_gif_metadata

# Manual metadata creation (bypasses directory detection)
platform, metadata = create_tenor_metadata(
    query="love",
    collection_context="valentine_campaign",
    tenor_id="12345"
)

gif_metadata = extract_gif_metadata(
    gif_path,
    source_platform=platform,
    source_metadata=metadata
)
```

### Mixed Approach
```bash
# Use directory detection for most files
python -m giflab run data/raw/ --detect-source-from-directory

# Disable for specific manual cases
python -m giflab run data/raw/special_cases/ --no-detect-source-from-directory
```

## Troubleshooting

### Common Issues

1. **Files in wrong directories**
   - Solution: Move files to correct directories and re-run
   - Detection is based on current location

2. **Unexpected metadata**
   - Check directory structure matches expected format
   - Verify directory names are correct

3. **Missing metadata**
   - Files directly in `data/raw/` get `unknown` source
   - Move to appropriate subdirectories

4. **Performance concerns**
   - Directory detection is very fast
   - No performance impact on large collections

### Debug Commands
```bash
# Check directory structure
python -m giflab organize-directories data/raw/

# Test detection without processing
python -m giflab run data/raw/ --dry-run

# Disable detection for debugging
python -m giflab run data/raw/ --no-detect-source-from-directory
```

## Migration Guide

### From Manual Metadata
1. Organize existing GIFs into directory structure
2. Run `organize-directories` to create structure
3. Move files to appropriate locations
4. Re-run analysis with directory detection

### From Mixed Sources
1. Identify current sources and patterns
2. Create directory structure
3. Write scripts to move files based on existing metadata
4. Verify organization before running analysis

This directory-based approach provides a robust, scalable solution for managing multi-source GIF collections while maintaining rich metadata for analysis and ML training. 