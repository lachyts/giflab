# 🏷️ GifLab Tagging Approach — Technical Documentation

---

## Executive Summary

**Problem**: Predicting optimal compression parameters requires understanding GIF content characteristics and existing compression artifacts, not generic content tagging.

**Solution**: Compression-aware tagging system using classical computer vision to classify content types and detect artifacts that directly inform compression parameter selection.

**Key Results**: 
- Zero API costs with local processing
- 8-12 meaningful tags that directly impact compression decisions
- Fast execution suitable for large datasets
- 70-80% accuracy target for compression parameter prediction

---

## 1. The Compression Prediction Problem

### Why Generic Tagging Falls Short
Traditional content tagging focuses on semantic understanding ("cat", "sunset", "person"), but compression optimization depends on technical characteristics:

- **Vector art** compresses differently than **photography**
- **Screen captures** with text need different parameters than **natural images**
- **Already-compressed** GIFs require different handling than **pristine** sources
- **High-contrast** content behaves differently under lossy compression

**Core Insight**: We need compression-relevant categorization, not semantic description.

---

## 2. Compression-Relevant Tag Categories

### 2.1 Content Type Categories
These categories directly impact optimal compression parameters:

```python
CONTENT_TYPE_CATEGORIES = {
    "vector-art",      # Clean geometric shapes, solid colors
                      # → Benefits from high color reduction, low lossy
    
    "screen-capture",  # Text, UI elements, sharp edges
                      # → Needs careful lossy settings to preserve text
    
    "photography",     # Natural images, complex textures  
                      # → Can handle more lossy compression
    
    "hand-drawn",      # Artwork with organic lines
                      # → Moderate compression settings
    
    "3d-rendered",     # Computer graphics, smooth gradients
                      # → Responds well to frame reduction
    
    "pixel-art",       # Low-resolution, limited palette
                      # → Minimal color reduction needed
    
    "mixed-content"    # Combination of above types
                      # → Conservative compression settings
}
```

### 2.2 Quality/Artifact Assessment
Existing compression level affects how aggressive we can be:

```python
QUALITY_CATEGORIES = {
    "pristine",           # No visible compression artifacts
                         # → Can apply full compression range
    
    "lightly-compressed", # Minor artifacts present
                         # → Moderate additional compression safe
    
    "heavily-compressed", # Significant artifacts visible  
                         # → Minimal additional compression
    
    "low-quality"        # Poor source material
                         # → Focus on size reduction over quality
}
```

### 2.3 Technical Characteristics
Additional flags that influence compression decisions:

```python
TECHNICAL_TAGS = {
    "text-heavy",      # Contains significant text content
    "high-contrast",   # Sharp edges and transitions
    "low-color-count", # Limited color palette
    "high-detail",     # Complex textures and patterns
    "smooth-gradients" # Gradual color transitions
}
```

---

## 3. Implementation Phases

### 3.1 Phase 1: Classical Computer Vision (Recommended Start)

**Advantages:**
- Zero cost - no API fees
- Fast execution - runs locally  
- Privacy-friendly - no data leaves system
- Good accuracy - 70-80% sufficient for compression prediction
- Easy to implement - builds on existing CV knowledge

**Core Implementation:**
```python
class CompressionTagger:
    def __init__(self):
        # No external dependencies - pure classical CV
        pass
    
    def analyze_gif(self, gif_path: Path) -> Dict[str, float]:
        """Analyze GIF and return continuous scores for all metrics."""
        frames = extract_representative_frames(gif_path, max_frames=3)
        
        # Calculate all continuous scores
        scores = {
            'text_density': self.calculate_text_density(frames[0]),
            'edge_density': self.calculate_edge_density(frames[0]),
            'artifact_score': self.calculate_artifact_score(frames[0]),
            'color_complexity': self.calculate_color_complexity(frames[0]),
            'contrast_score': self.calculate_contrast_score(frames[0]),
            'gradient_smoothness': self.calculate_gradient_smoothness(frames[0])
        }
        
        return scores
    
    def get_content_type(self, scores: Dict[str, float]) -> str:
        """Derive content type from continuous scores."""
        # Use scores to classify content type
        if scores['text_density'] > 0.3 and scores['edge_density'] > 0.1:
            return "screen-capture"
        elif scores['color_complexity'] < 0.2 and scores['edge_density'] > 0.15:
            return "vector-art"
        # ... other classification logic
        else:
            return "photography"
```

### 3.2 Phase 2: Local CLIP Enhancement (If Needed)

**When to Use:** If Phase 1 accuracy is insufficient (< 70%)

**Implementation:**
```python
import clip
import torch

class CLIPEnhancedTagger(CompressionTagger):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def classify_content_type_clip(self, image):
        """Enhanced content classification using local CLIP."""
        text_queries = [
            "a screenshot of computer software",
            "vector art with geometric shapes", 
            "a photograph or realistic image",
            "hand drawn artwork or illustration",
            "3D rendered computer graphics",
            "pixel art or low resolution graphics"
        ]
        
        # Process through CLIP
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(text_queries).to(self.device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = self.clip_model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        categories = ["screen-capture", "vector-art", "photography", 
                     "hand-drawn", "3d-rendered", "pixel-art"]
        return categories[probs.argmax()]
```

### 3.3 Phase 3: Custom Lightweight Models (Future)

**When to Use:** Need 90%+ accuracy for production deployment

**Approach:** Train small CNNs on curated compression-specific datasets

```python
class CustomCompressionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Lightweight CNN for compression-type classification
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)
```

---

## 4. Classical Computer Vision Implementation

### 4.1 Content Type Classification

```python
def classify_content_type(self, frame):
    """Classify content type using classical CV heuristics."""
    
    # Calculate key metrics
    unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
    edge_density = self.calculate_edge_density(frame)
    color_variance = np.var(frame)
    text_likelihood = self.detect_text_patterns(frame)
    
    # Decision tree based on characteristics
    if text_likelihood > 0.3 and edge_density > 0.1:
        return "screen-capture"
    elif unique_colors < 50 and edge_density > 0.15:
        return "vector-art"
    elif unique_colors < 16 and edge_density > 0.2:
        return "pixel-art"
    elif edge_density < 0.05 and color_variance < 100:
        return "3d-rendered"
    elif self.has_organic_lines(frame):
        return "hand-drawn"
    else:
        return "photography"

def calculate_edge_density(self, frame):
    """Calculate density of edges in frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size

def detect_text_patterns(self, frame):
    """Detect text-like patterns using morphological operations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Morphological operations to detect text-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Look for rectangular text-like regions
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_like_regions = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Text typically has certain aspect ratios
        if 0.1 < aspect_ratio < 10 and w > 10 and h > 5:
            text_like_regions += 1
    
    return min(text_like_regions / 100, 1.0)  # Normalize
```

### 4.2 Compression Artifact Detection

```python
def detect_compression_artifacts(self, frame):
    """Detect compression artifacts using classical CV techniques."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Detect blocking artifacts (8x8 DCT blocks)
    blocking_score = self.detect_blocking_artifacts(gray)
    
    # Detect ringing artifacts around edges
    ringing_score = self.detect_ringing_artifacts(gray)
    
    # Detect quantization noise
    noise_score = self.detect_quantization_noise(gray)
    
    # Combine scores
    total_artifact_score = (blocking_score + ringing_score + noise_score) / 3
    
    if total_artifact_score < 0.1:
        return "pristine"
    elif total_artifact_score < 0.3:
        return "lightly-compressed"
    elif total_artifact_score < 0.6:
        return "heavily-compressed"
    else:
        return "low-quality"

def detect_blocking_artifacts(self, gray_image):
    """Detect 8x8 DCT blocking artifacts."""
    h, w = gray_image.shape
    
    # Calculate differences at 8-pixel intervals
    h_diffs = []
    v_diffs = []
    
    for i in range(8, h-8, 8):
        for j in range(w-1):
            h_diffs.append(abs(int(gray_image[i, j]) - int(gray_image[i-1, j])))
    
    for i in range(h-1):
        for j in range(8, w-8, 8):
            v_diffs.append(abs(int(gray_image[i, j]) - int(gray_image[i, j-1])))
    
    # Higher values at block boundaries indicate blocking
    boundary_strength = np.mean(h_diffs + v_diffs) if h_diffs or v_diffs else 0
    return min(boundary_strength / 255.0, 1.0)

def detect_ringing_artifacts(self, gray_image):
    """Detect ringing artifacts around edges."""
    # Edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Gaussian blur to simulate ringing
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    diff = cv2.absdiff(gray_image, blurred)
    
    # Focus on edge areas
    edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    ringing_areas = cv2.bitwise_and(diff, diff, mask=edge_mask)
    
    return np.mean(ringing_areas) / 255.0
```

### 4.3 Continuous Score Calculation

```python
def calculate_text_density(self, frame):
    """Calculate text density score (0.0-1.0)."""
    return self.detect_text_patterns(frame)  # Already returns 0.0-1.0

def calculate_edge_density(self, frame):
    """Calculate edge density score (0.0-1.0)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size

def calculate_contrast_score(self, frame):
    """Calculate contrast variation score (0.0-1.0)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return min(np.std(gray) / 128.0, 1.0)  # Normalize to 0-1

def calculate_color_complexity(self, frame):
    """Calculate color complexity score (0.0-1.0)."""
    unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
    # Normalize: assume max ~16M colors (256^3), but cap at reasonable level
    return min(unique_colors / 1000.0, 1.0)

def calculate_gradient_smoothness(self, frame):
    """Calculate gradient smoothness score (0.0-1.0)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Calculate local gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Smooth gradients have low, consistent gradients
    # Invert so higher score = smoother gradients
    gradient_variation = np.std(grad_magnitude)
    return max(0, 1.0 - (gradient_variation / 50.0))

def calculate_artifact_score(self, frame):
    """Calculate compression artifact score (0.0-1.0)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Combine multiple artifact detection methods
    blocking_score = self.detect_blocking_artifacts(gray)
    ringing_score = self.detect_ringing_artifacts(gray)
    noise_score = self.detect_quantization_noise(gray)
    
    return (blocking_score + ringing_score + noise_score) / 3
```

---

## 5. Performance Characteristics

### 5.1 Phase Comparison

| Phase | Accuracy | Speed | Cost | Setup | Privacy |
|-------|----------|-------|------|-------|---------|
| **Phase 1: Classical CV** | 70-80% | Fast | $0 | Low | Perfect |
| **Phase 2: Local CLIP** | 85-90% | Medium | $0 | Medium | Perfect |
| **Phase 3: Custom CNN** | 90-95%* | Fast | $0* | High | Perfect |
| **Cloud APIs** | 95%+ | Slow | $50-500 | Low | Poor |

*\*After training investment*

### 5.2 Expected Processing Times

**Classical CV (Phase 1):**
- Single GIF: ~10-50ms
- 1,000 GIFs: ~30 seconds
- 10,000 GIFs: ~5 minutes

**With Local CLIP (Phase 2):**
- Single GIF: ~100-200ms  
- 1,000 GIFs: ~3 minutes
- 10,000 GIFs: ~30 minutes

---

## 6. Integration with Existing Pipeline

### 6.1 TaggingPipeline Enhancement

```python
class TaggingPipeline:
    def __init__(self, model_name: str = "classical-cv"):
        if model_name == "classical-cv":
            self.analyzer = CompressionTagger()
        elif model_name == "clip-enhanced":
            self.analyzer = CLIPEnhancedTagger()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def analyze_single_gif(self, gif_path: Path) -> AnalysisResult:
        """Generate compression-aware scores for a single GIF."""
        
        try:
            start_time = time.time()
            
            # Calculate continuous scores
            scores = self.analyzer.analyze_gif(gif_path)
            
            # Optional: derive content type for logging/debugging
            content_type = self.analyzer.get_content_type(scores)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return AnalysisResult(
                gif_sha=calculate_sha(gif_path),
                scores=scores,
                content_type=content_type,  # For reference only
                model_version=self.analyzer.__class__.__name__,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {gif_path}: {e}")
            return AnalysisResult(
                gif_sha=calculate_sha(gif_path),
                scores={},
                content_type="error",
                model_version="failed",
                processing_time_ms=0
            )
    
    def derive_tags(self, scores: Dict[str, float], 
                   thresholds: Dict[str, float] = None) -> List[str]:
        """Convert continuous scores to tags using configurable thresholds."""
        if thresholds is None:
            thresholds = self.get_default_thresholds()
        
        tags = []
        
        # Content type classification
        content_type = self.analyzer.get_content_type(scores)
        tags.append(content_type)
        
        # Quality assessment
        if scores.get('artifact_score', 0) < thresholds['pristine_threshold']:
            tags.append('pristine')
        elif scores.get('artifact_score', 0) < thresholds['light_compression_threshold']:
            tags.append('lightly-compressed')
        else:
            tags.append('heavily-compressed')
        
        # Technical characteristics
        if scores.get('text_density', 0) > thresholds['text_heavy_threshold']:
            tags.append('text-heavy')
        if scores.get('contrast_score', 0) > thresholds['high_contrast_threshold']:
            tags.append('high-contrast')
        if scores.get('color_complexity', 0) < thresholds['low_color_threshold']:
            tags.append('low-color-count')
            
        return tags
```

### 6.2 CSV Output Format

Instead of storing tags, we store continuous scores that can be used directly by ML models and converted to tags on-demand:

```python
# New CSV columns (added to existing schema)
text_density = 0.67        # 0.0-1.0, higher = more text content
edge_density = 0.23        # 0.0-1.0, higher = more sharp edges  
artifact_score = 0.15      # 0.0-1.0, higher = more compression artifacts
color_complexity = 0.12    # 0.0-1.0, higher = more unique colors
contrast_score = 0.89      # 0.0-1.0, higher = more contrast variation
gradient_smoothness = 0.34 # 0.0-1.0, higher = more smooth gradients
```

**Advantages:**
- **ML-ready**: Continuous values directly usable by models
- **Flexible**: Tags derived on-demand with tunable thresholds
- **Clean CSV**: No bloated tag strings
- **Analyzable**: Direct correlation with compression parameters

### 6.3 Updated CSV Schema

The following columns will be added to the existing CSV schema:

| Column | Type | Range | Description | Compression Relevance |
|--------|------|-------|-------------|----------------------|
| `text_density` | float | 0.0-1.0 | Text content density | Affects lossy compression tolerance |
| `edge_density` | float | 0.0-1.0 | Sharp edge density | Impacts color reduction effectiveness |
| `artifact_score` | float | 0.0-1.0 | Compression artifacts present | Determines additional compression headroom |
| `color_complexity` | float | 0.0-1.0 | Unique color count (normalized) | Informs color palette reduction strategy |
| `contrast_score` | float | 0.0-1.0 | Contrast variation | Affects dithering and lossy settings |
| `gradient_smoothness` | float | 0.0-1.0 | Smooth gradient presence | Frame reduction tolerance |

**Example Row:**
```csv
gif_sha,orig_filename,engine,lossy,frame_keep_ratio,color_keep_count,kilobytes,ssim,text_density,edge_density,artifact_score,color_complexity,contrast_score,gradient_smoothness,timestamp
6c54c899...,example.gif,gifsicle,40,0.80,64,413.72,0.936,0.67,0.23,0.15,0.12,0.89,0.34,2024-01-15T10:30:00Z
```

---

## 7. Validation & Quality Assurance

### 7.1 Test Dataset Requirements

**Manual Validation Set (Recommended):**
- 100-200 representative GIFs across all categories
- Manual tagging by domain expert
- Balanced distribution of content types and quality levels

**Validation Metrics:**
- Per-category accuracy (target: >70% for Phase 1)
- Overall classification accuracy  
- Confusion matrix analysis
- Processing time benchmarks

### 7.2 Continuous Improvement

**Feedback Loop:**
1. Monitor compression parameter prediction accuracy
2. Identify misclassified GIFs affecting compression decisions
3. Refine heuristics or consider Phase 2 upgrade
4. Re-validate against test dataset

---

## 8. Recommended Implementation Path

### 8.1 Immediate Implementation (Phase 1)

```python
# Start with classical CV approach
tagger = CompressionTagger()

# Focus on core categories first
INITIAL_CATEGORIES = [
    "screen-capture", "photography", "vector-art",  # Content types
    "pristine", "lightly-compressed", "heavily-compressed",  # Quality
    "text-heavy", "high-contrast"  # Technical flags
]
```

### 8.2 Success Criteria

**Phase 1 Success Metrics:**
- ✅ 70%+ accuracy in compression parameter prediction
- ✅ < 50ms processing time per GIF
- ✅ Zero external dependencies or costs
- ✅ 6 continuous scores covering key compression factors
- ✅ Scores correlate with optimal compression parameters (R² > 0.5)
- ✅ Clean CSV integration without data bloat

### 8.3 Threshold Configuration

**Default Tag Derivation Thresholds:**
```python
DEFAULT_THRESHOLDS = {
    # Quality assessment
    'pristine_threshold': 0.1,           # artifact_score < 0.1 = pristine
    'light_compression_threshold': 0.3,  # artifact_score < 0.3 = lightly-compressed
    
    # Technical characteristics  
    'text_heavy_threshold': 0.3,         # text_density > 0.3 = text-heavy
    'high_contrast_threshold': 0.7,      # contrast_score > 0.7 = high-contrast
    'low_color_threshold': 0.2,          # color_complexity < 0.2 = low-color-count
    'smooth_gradient_threshold': 0.6,    # gradient_smoothness > 0.6 = smooth-gradients
    'high_edge_threshold': 0.4,          # edge_density > 0.4 = high-edge-density
}
```

**Tuning Process:**
1. Analyze score distributions across GIF dataset
2. Correlate scores with compression effectiveness  
3. Adjust thresholds for optimal parameter prediction
4. Validate against held-out test set

### 8.4 Decision Points

**When to consider Phase 2:**
- Phase 1 score accuracy < 70% correlation with optimal parameters
- Specific content types consistently misclassified
- Compression parameter prediction not improving

**When to consider Phase 3:**
- Need 90%+ accuracy for production
- Have budget for model training
- Dataset large enough (>10,000 manually labeled examples)

---

## 9. Future Enhancements

### 9.1 Advanced Artifact Detection
- **Temporal artifacts**: Frame stuttering, motion blur
- **Color banding**: Gradient quantization issues  
- **Moiré patterns**: Aliasing in screen captures
- **JPEG artifacts**: Block boundaries, ringing

### 9.2 Content-Specific Optimizations
- **Text detection**: OCR-based text density analysis
- **Face detection**: Portrait-specific compression
- **Logo detection**: Brand preservation requirements
- **Animation analysis**: Motion complexity assessment

### 9.3 Machine Learning Integration
- **Clustering**: Discover new content categories automatically
- **Anomaly detection**: Identify unusual GIF characteristics
- **Transfer learning**: Adapt pre-trained models for compression
- **Ensemble methods**: Combine multiple classification approaches

---

## 10. Critical Success Factors

✅ **Focus on compression relevance** - Tags directly inform parameter selection
✅ **Start simple and iterate** - Classical CV provides solid foundation  
✅ **Zero external dependencies** - No API costs or privacy concerns
✅ **Fast local processing** - Suitable for large dataset analysis
✅ **Clear upgrade path** - Phases 2 and 3 ready when needed

**Bottom Line**: This approach provides practical, cost-effective tagging that directly supports compression parameter optimization, with a clear path for accuracy improvements as needed. 