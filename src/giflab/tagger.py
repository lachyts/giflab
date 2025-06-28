"""Comprehensive hybrid tagging system for GIF content analysis and compression optimization.

This module implements a hybrid approach combining:
- CLIP for semantic content classification (6 metrics)
- Classical computer vision for technical analysis (9 metrics) 
- Comprehensive temporal motion analysis (10 metrics)

Total: 25 continuous scores (0.0-1.0) for ML-ready compression parameter prediction.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import cv2
from PIL import Image

# Conditional imports for CLIP dependencies
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    torch = None
    clip = None

from .meta import extract_gif_metadata


@dataclass
class TaggingResult:
    """Result of comprehensive hybrid tagging for a GIF."""
    
    gif_sha: str
    scores: Dict[str, float]  # 25 continuous scores 0.0-1.0
    model_version: str
    processing_time_ms: int
    
    @property
    def content_classification(self) -> Dict[str, float]:
        """Get CLIP content classification scores."""
        content_keys = {
            'screen_capture_confidence', 'vector_art_confidence', 'photography_confidence',
            'hand_drawn_confidence', '3d_rendered_confidence', 'pixel_art_confidence'
        }
        return {k: v for k, v in self.scores.items() if k in content_keys}
    
    @property 
    def quality_assessment(self) -> Dict[str, float]:
        """Get quality/artifact assessment scores."""
        quality_keys = {
            'blocking_artifacts', 'ringing_artifacts', 
            'quantization_noise', 'overall_quality'
        }
        return {k: v for k, v in self.scores.items() if k in quality_keys}
    
    @property
    def technical_characteristics(self) -> Dict[str, float]:
        """Get technical characteristic scores."""
        tech_keys = {
            'text_density', 'edge_density', 'color_complexity',
            'contrast_score', 'gradient_smoothness'
        }
        return {k: v for k, v in self.scores.items() if k in tech_keys}
    
    @property
    def temporal_analysis(self) -> Dict[str, float]:
        """Get temporal motion analysis scores."""
        temporal_keys = {
            'frame_similarity', 'motion_intensity', 'motion_smoothness',
            'static_region_ratio', 'scene_change_frequency', 'fade_transition_presence',
            'cut_sharpness', 'temporal_entropy', 'loop_detection_confidence',
            'motion_complexity'
        }
        return {k: v for k, v in self.scores.items() if k in temporal_keys}


class HybridCompressionTagger:
    """Hybrid tagger combining CLIP content classification with classical CV analysis 
    and comprehensive temporal motion analysis.
    
    Generates 25 continuous scores for compression parameter prediction:
    - 6 content classification scores (CLIP)
    - 4 quality/artifact assessment scores (Classical CV)
    - 5 technical characteristic scores (Classical CV)
    - 10 temporal motion analysis scores (Classical CV)
    """
    
    def __init__(self):
        """Initialize the hybrid tagger with CLIP model and content queries."""
        if not CLIP_AVAILABLE:
            raise RuntimeError(
                "CLIP dependencies not available. Install with: pip install torch clip-by-openai"
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        
        # Initialize CLIP for content classification
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.logger.info(f"CLIP model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP initialization failed: {e}")
        
        # Content type queries for CLIP semantic classification
        self.content_queries = [
            "a screenshot of computer software with text and UI elements",
            "vector art with clean geometric shapes and solid colors", 
            "a photograph or realistic image with natural textures",
            "hand drawn artwork or digital illustration",
            "3D rendered computer graphics with smooth surfaces",
            "pixel art or low resolution retro graphics"
        ]
        
        self.content_types = [
            "screen_capture", "vector_art", "photography", 
            "hand_drawn", "3d_rendered", "pixel_art"
        ]
        
        self.model_version = "hybrid_v1.0_comprehensive"
    
    def tag_gif(self, gif_path: Path, gif_sha: Optional[str] = None) -> TaggingResult:
        """Generate comprehensive tagging scores for a GIF.
        
        Args:
            gif_path: Path to the GIF file to analyze
            gif_sha: Optional pre-computed SHA hash
            
        Returns:
            TaggingResult with 25 continuous scores and metadata
            
        Raises:
            IOError: If GIF file cannot be read
            RuntimeError: If analysis fails
        """
        start_time = time.time()
        
        try:
            # Get GIF metadata if SHA not provided
            if gif_sha is None:
                metadata = extract_gif_metadata(gif_path)
                gif_sha = metadata.gif_sha
            
            # Extract frames for analysis (max 10 for robust temporal analysis)
            frames = self._extract_representative_frames(gif_path, max_frames=10)
            if not frames:
                raise RuntimeError("Could not extract frames from GIF")
            
            representative_frame = frames[0]  # Use first frame for CLIP analysis
            
            # Get CLIP content classification scores (6 metrics)
            content_scores = self._classify_content_with_clip(representative_frame)
            
            # Get classical CV technical scores including temporal analysis (19 metrics)
            technical_scores = self._analyze_comprehensive_characteristics(frames)
            
            # Combine all scores (6 + 19 = 25 total)
            all_scores = {**content_scores, **technical_scores}
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return TaggingResult(
                gif_sha=gif_sha,
                scores=all_scores,
                model_version=self.model_version,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Tagging failed for {gif_path}: {e}")
            raise RuntimeError(f"Tagging analysis failed: {e}")
    
    def _extract_representative_frames(self, gif_path: Path, max_frames: int = 10) -> List[np.ndarray]:
        """Extract representative frames from GIF for analysis."""
        try:
            # Use cv2 to read GIF frames
            cap = cv2.VideoCapture(str(gif_path))
            frames = []
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= max_frames:
                # Read all frames if small GIF
                frame_indices = range(frame_count)
            else:
                # Sample frames evenly across the GIF
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed for {gif_path}: {e}")
            raise
    
    def _classify_content_with_clip(self, image: np.ndarray) -> Dict[str, float]:
        """Use CLIP to get confidence scores for each content type."""
        try:
            # Convert to PIL Image for CLIP preprocessing
            pil_image = Image.fromarray(image)
            
            # Preprocess image and text
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            text_inputs = clip.tokenize(self.content_queries).to(self.device)
            
            # Get CLIP predictions
            with torch.no_grad():
                logits_per_image, _ = self.clip_model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Return confidence scores for each content type
            return {
                f"{content_type}_confidence": float(prob) 
                for content_type, prob in zip(self.content_types, probs)
            }
            
        except Exception as e:
            self.logger.error(f"CLIP classification failed: {e}")
            # Return zero scores if CLIP fails
            return {f"{content_type}_confidence": 0.0 for content_type in self.content_types}
    
    def _analyze_comprehensive_characteristics(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Comprehensive analysis including static technical metrics and temporal motion analysis."""
        representative_frame = frames[0]  # Use first frame for static analysis
        
        return {
            # Quality/artifact assessment (4 metrics)
            'blocking_artifacts': self._calculate_blocking_artifacts(representative_frame),
            'ringing_artifacts': self._calculate_ringing_artifacts(representative_frame),
            'quantization_noise': self._calculate_quantization_noise(representative_frame),
            'overall_quality': self._calculate_overall_quality(representative_frame),
            
            # Technical characteristics - static (5 metrics)
            'text_density': self._calculate_text_density(representative_frame),
            'edge_density': self._calculate_edge_density(representative_frame),
            'color_complexity': self._calculate_color_complexity(representative_frame),
            'contrast_score': self._calculate_contrast_score(representative_frame),
            'gradient_smoothness': self._calculate_gradient_smoothness(representative_frame),
            
            # Temporal motion analysis (10 metrics)
            'frame_similarity': self._calculate_frame_similarity(frames),
            'motion_intensity': self._calculate_motion_intensity(frames),
            'motion_smoothness': self._calculate_motion_smoothness(frames),
            'static_region_ratio': self._calculate_static_region_ratio(frames),
            'scene_change_frequency': self._calculate_scene_change_frequency(frames),
            'fade_transition_presence': self._calculate_fade_transition_presence(frames),
            'cut_sharpness': self._calculate_cut_sharpness(frames),
            'temporal_entropy': self._calculate_temporal_entropy(frames),
            'loop_detection_confidence': self._calculate_loop_detection_confidence(frames),
            'motion_complexity': self._calculate_motion_complexity(frames),
        }
    
    # Quality/Artifact Assessment Methods
    
    def _calculate_blocking_artifacts(self, frame: np.ndarray) -> float:
        """Detect DCT blocking artifacts (8x8 patterns)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Calculate differences at 8-pixel intervals (DCT block boundaries)
            h_diffs = []
            v_diffs = []
            
            for i in range(8, h-8, 8):
                for j in range(w-1):
                    h_diffs.append(abs(int(gray[i, j]) - int(gray[i-1, j])))
            
            for i in range(h-1):
                for j in range(8, w-8, 8):
                    v_diffs.append(abs(int(gray[i, j]) - int(gray[i, j-1])))
            
            # Higher boundary differences indicate blocking
            boundary_strength = np.mean(h_diffs + v_diffs) if h_diffs or v_diffs else 0
            return min(boundary_strength / 255.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_ringing_artifacts(self, frame: np.ndarray) -> float:
        """Detect ringing artifacts around edges."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply Laplacian to detect oscillations near edges
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            
            # Calculate ringing as high Laplacian response near edges
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            
            edge_mask = edges_dilated > 0
            ringing_strength = np.mean(laplacian_abs[edge_mask]) if np.any(edge_mask) else 0
            
            return min(ringing_strength / 100.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_quantization_noise(self, frame: np.ndarray) -> float:
        """Calculate color banding and posterization artifacts."""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Calculate histogram for value channel
            hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Look for sharp peaks indicating quantization
            hist_smooth = cv2.GaussianBlur(hist.reshape(-1, 1), (0, 0), 2).flatten()
            hist_diff = np.abs(hist.flatten() - hist_smooth)
            quantization_score = np.mean(hist_diff) / np.max(hist.flatten()) if np.max(hist.flatten()) > 0 else 0
            
            return min(quantization_score * 10, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_overall_quality(self, frame: np.ndarray) -> float:
        """Combined quality degradation score."""
        try:
            blocking = self._calculate_blocking_artifacts(frame)
            ringing = self._calculate_ringing_artifacts(frame)
            quantization = self._calculate_quantization_noise(frame)
            
            # Weighted combination of artifact measures
            overall = 0.4 * blocking + 0.3 * ringing + 0.3 * quantization
            return min(overall, 1.0)
            
        except Exception:
            return 0.5  # Default moderate quality if analysis fails
    
    # Technical Characteristics Methods
    
    def _calculate_text_density(self, frame: np.ndarray) -> float:
        """Detect text-like patterns using morphological operations."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Morphological operations to detect text structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find text-like rectangular regions
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like_regions = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text typically has specific aspect ratios and sizes
                if 0.1 < aspect_ratio < 10 and w > 10 and h > 5:
                    text_like_regions += 1
            
            return min(text_like_regions / 100, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_edge_density(self, frame: np.ndarray) -> float:
        """Calculate density of sharp edges."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return min(edge_density * 10, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_color_complexity(self, frame: np.ndarray) -> float:
        """Calculate normalized unique color count."""
        try:
            unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
            return min(unique_colors / 1000.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_contrast_score(self, frame: np.ndarray) -> float:
        """Calculate contrast variation."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray) / 255.0
            return min(contrast * 2, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_gradient_smoothness(self, frame: np.ndarray) -> float:
        """Calculate presence of smooth gradients."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Smooth gradients have low variance in gradient direction
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_smoothness = 1.0 - min(np.std(grad_magnitude) / 255.0, 1.0)
            
            return max(0.0, grad_smoothness)
            
        except Exception:
            return 0.5
    
    # Temporal Motion Analysis Methods
    
    def _calculate_frame_similarity(self, frames: List[np.ndarray]) -> float:
        """Calculate how similar consecutive frames are (higher = more similar)."""
        if len(frames) < 2:
            return 1.0  # Single frame GIF is perfectly "similar"
        
        try:
            similarities = []
            for i in range(len(frames) - 1):
                # Convert frames to grayscale for comparison
                frame1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Calculate normalized cross-correlation
                correlation = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF_NORMED)[0, 0]
                similarities.append(max(0, correlation))  # Ensure non-negative
            
            return np.mean(similarities)
            
        except Exception:
            return 0.5
    
    def _calculate_motion_intensity(self, frames: List[np.ndarray]) -> float:
        """Calculate overall motion level across frames (higher = more motion)."""
        if len(frames) < 2:
            return 0.0  # Single frame GIF has no motion
        
        try:
            motion_scores = []
            for i in range(len(frames) - 1):
                # Convert frames to grayscale
                frame1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(frame1, frame2)
                
                # Calculate motion as mean absolute difference normalized by max possible
                motion = np.mean(diff) / 255.0
                motion_scores.append(motion)
            
            return np.mean(motion_scores)
            
        except Exception:
            return 0.5
    
    def _calculate_motion_smoothness(self, frames: List[np.ndarray]) -> float:
        """Calculate motion smoothness (linear vs chaotic motion patterns)."""
        if len(frames) < 3:
            return 1.0
        
        try:
            motion_vectors = []
            for i in range(len(frames) - 1):
                # Calculate optical flow between consecutive frames
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Simple motion estimation using frame difference
                diff = cv2.absdiff(gray1, gray2)
                motion_magnitude = np.mean(diff)
                motion_vectors.append(motion_magnitude)
            
            # Smoothness = inverse of motion vector variance
            if len(motion_vectors) > 1:
                variance = np.var(motion_vectors)
                return max(0, 1.0 - min(variance / 100.0, 1.0))
            return 1.0
            
        except Exception:
            return 0.5
    
    def _calculate_static_region_ratio(self, frames: List[np.ndarray]) -> float:
        """Calculate percentage of image area that remains static."""
        if len(frames) < 2:
            return 1.0
        
        try:
            # Create motion mask by accumulating frame differences
            motion_mask = np.zeros(frames[0][:,:,0].shape, dtype=np.float32)
            
            for i in range(len(frames) - 1):
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Calculate absolute difference
                diff = cv2.absdiff(gray1, gray2)
                
                # Threshold to create binary motion mask
                _, motion_binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                motion_mask += motion_binary.astype(np.float32)
            
            # Normalize motion mask
            motion_mask = motion_mask / (len(frames) - 1)
            
            # Calculate static ratio (inverse of motion)
            motion_pixels = np.sum(motion_mask > 50)  # Pixels with significant motion
            total_pixels = motion_mask.size
            static_ratio = (total_pixels - motion_pixels) / total_pixels
            
            return max(0, min(static_ratio, 1.0))
            
        except Exception:
            return 0.5
    
    def _calculate_scene_change_frequency(self, frames: List[np.ndarray]) -> float:
        """Detect major scene changes (cuts, not gradual transitions)."""
        if len(frames) < 2:
            return 0.0
        
        try:
            scene_changes = 0
            threshold = 0.3  # Threshold for scene change detection
            
            for i in range(len(frames) - 1):
                # Calculate histogram difference for scene change detection
                hist1 = cv2.calcHist([frames[i]], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([frames[i + 1]], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                
                # Chi-squared distance for histogram comparison
                chi_squared = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                normalized_distance = min(chi_squared / 10000.0, 1.0)
                
                if normalized_distance > threshold:
                    scene_changes += 1
            
            return min(scene_changes / len(frames), 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_fade_transition_presence(self, frames: List[np.ndarray]) -> float:
        """Detect fade in/out effects."""
        if len(frames) < 3:
            return 0.0
        
        try:
            fade_indicators = []
            
            for i in range(1, len(frames) - 1):
                # Calculate brightness progression
                brightness_prev = np.mean(cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY))
                brightness_curr = np.mean(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY))
                brightness_next = np.mean(cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY))
                
                # Check for monotonic brightness change (fade pattern)
                if ((brightness_prev < brightness_curr < brightness_next) or 
                    (brightness_prev > brightness_curr > brightness_next)):
                    fade_indicators.append(1)
                else:
                    fade_indicators.append(0)
            
            return np.mean(fade_indicators) if fade_indicators else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_cut_sharpness(self, frames: List[np.ndarray]) -> float:
        """Measure sharpness of cuts vs smooth transitions."""
        if len(frames) < 2:
            return 0.0
        
        try:
            cut_sharpness_scores = []
            
            for i in range(len(frames) - 1):
                # Calculate histogram difference between consecutive frames
                hist1 = cv2.calcHist([frames[i]], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([frames[i + 1]], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
                
                # Sharp cuts have high histogram differences
                bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                cut_sharpness_scores.append(min(bhattacharyya, 1.0))
            
            return np.mean(cut_sharpness_scores)
            
        except Exception:
            return 0.0
    
    def _calculate_temporal_entropy(self, frames: List[np.ndarray]) -> float:
        """Calculate information entropy across temporal dimension."""
        if len(frames) < 2:
            return 0.0
        
        try:
            # Convert frames to grayscale and flatten
            temporal_data = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                temporal_data.extend(gray.flatten())
            
            # Calculate histogram and entropy
            hist, _ = np.histogram(temporal_data, bins=256, range=(0, 255))
            hist = hist / np.sum(hist)  # Normalize
            
            # Calculate Shannon entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
            return min(entropy / 8.0, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def _calculate_loop_detection_confidence(self, frames: List[np.ndarray]) -> float:
        """Detect cyclical/repeating patterns in the GIF."""
        if len(frames) < 4:
            return 0.0
        
        try:
            # Compare first and last frames for loop detection
            first_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
            last_frame = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2GRAY)
            
            # Calculate structural similarity
            ssim_score = cv2.matchTemplate(first_frame, last_frame, cv2.TM_CCOEFF_NORMED)[0, 0]
            
            # Look for periodic patterns in middle frames
            mid_similarities = []
            for i in range(1, len(frames) - 1):
                mid_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                # Check similarity to corresponding frame if this were a perfect loop
                loop_index = i % len(frames)
                if loop_index < len(frames):
                    loop_frame = cv2.cvtColor(frames[loop_index], cv2.COLOR_RGB2GRAY)
                    similarity = cv2.matchTemplate(mid_frame, loop_frame, cv2.TM_CCOEFF_NORMED)[0, 0]
                    mid_similarities.append(max(0, similarity))
            
            # Combine first-last similarity with periodic pattern strength
            loop_confidence = (0.6 * max(0, ssim_score) + 
                              0.4 * np.mean(mid_similarities) if mid_similarities else 0)
            return min(loop_confidence, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_motion_complexity(self, frames: List[np.ndarray]) -> float:
        """Calculate complexity of motion vectors across the sequence."""
        if len(frames) < 3:
            return 0.0
        
        try:
            motion_directions = []
            motion_magnitudes = []
            
            for i in range(len(frames) - 1):
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Calculate gradients for motion direction analysis
                grad_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
                grad_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
                grad_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
                grad_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
                
                # Motion direction change
                direction_change = np.mean(np.abs(grad_x2 - grad_x1) + np.abs(grad_y2 - grad_y1))
                motion_directions.append(direction_change)
                
                # Motion magnitude
                diff = cv2.absdiff(gray1, gray2)
                motion_magnitudes.append(np.mean(diff))
            
            # Complexity = variance in both direction and magnitude
            direction_complexity = np.var(motion_directions) if motion_directions else 0
            magnitude_complexity = np.var(motion_magnitudes) if motion_magnitudes else 0
            
            # Normalize and combine
            total_complexity = (direction_complexity / 1000.0 + magnitude_complexity / 100.0) / 2
            return min(total_complexity, 1.0)
            
        except Exception:
            return 0.0 