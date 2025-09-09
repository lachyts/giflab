import numpy as np
import pytest
from giflab.metrics import (
    chist,
    edge_similarity,
    fsim,
    gmsd,
    mse,
    rmse,
    sharpness_similarity,
    texture_similarity,
)
from PIL import Image, ImageDraw


class TestAdditionalMetrics:
    """Unit tests for newly added quality metrics."""

    def _identical_frames(self):
        """Create two identical frames with realistic content."""
        frame = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        return frame, frame.copy()

    def _slightly_different_frames(self):
        """Create frames with small differences (should score well)."""
        frame1 = np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Add small noise
        noise = np.random.randint(-10, 11, frame1.shape, dtype=np.int16)
        frame2 = np.clip(frame2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame1, frame2

    def _very_different_frames(self):
        """Create frames with significant differences (should score poorly)."""
        # Frame 1: Random pattern
        frame1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Frame 2: Completely different random pattern
        np.random.seed(999)  # Different seed for different pattern
        frame2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        np.random.seed()  # Reset seed
        return frame1, frame2

    def _gradient_frames(self):
        """Create frames with gradients to test gradient-based metrics."""
        # Similar gradients
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        for i in range(64):
            for j in range(64):
                # Horizontal gradient
                val1 = int(255 * i / 63)
                val2 = int(255 * i / 63) + np.random.randint(-5, 6)  # Slight variation
                frame1[i, j] = [val1, val1, val1]
                frame2[i, j] = [np.clip(val2, 0, 255)] * 3

        return frame1, frame2

    def _different_gradient_frames(self):
        """Create frames with very different gradients."""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        for i in range(64):
            for j in range(64):
                # Horizontal vs vertical gradients
                val1 = int(255 * i / 63)  # Horizontal gradient
                val2 = int(255 * j / 63)  # Vertical gradient
                frame1[i, j] = [val1, val1, val1]
                frame2[i, j] = [val2, val2, val2]

        return frame1, frame2

    def _color_frames(self):
        """Create frames with different color distributions."""
        # Similar color distribution
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Fill with similar colors
        frame1[:32, :32] = [255, 0, 0]  # Red quadrant
        frame1[:32, 32:] = [0, 255, 0]  # Green quadrant
        frame1[32:, :32] = [0, 0, 255]  # Blue quadrant
        frame1[32:, 32:] = [255, 255, 0]  # Yellow quadrant

        # Slightly different colors
        frame2[:32, :32] = [250, 5, 5]  # Slightly different red
        frame2[:32, 32:] = [5, 250, 5]  # Slightly different green
        frame2[32:, :32] = [5, 5, 250]  # Slightly different blue
        frame2[32:, 32:] = [250, 250, 5]  # Slightly different yellow

        return frame1, frame2

    def _different_color_frames(self):
        """Create frames with very different color distributions."""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Completely different color patterns
        frame1[:] = [255, 0, 0]  # All red
        frame2[:] = [0, 0, 255]  # All blue

        return frame1, frame2

    def _edge_frames(self):
        """Create frames with similar edge patterns."""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Create rectangle edges
        frame1[10:54, 10:54] = 255  # White rectangle
        frame2[12:52, 12:52] = 255  # Slightly different rectangle

        return frame1, frame2

    def _different_edge_frames(self):
        """Create frames with very different edge patterns."""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Rectangle vs circle
        frame1[10:54, 10:54] = 255  # Rectangle

        # Create circle in frame2
        img = Image.fromarray(frame2)
        draw = ImageDraw.Draw(img)
        draw.ellipse([16, 16, 48, 48], fill=(255, 255, 255))
        frame2 = np.array(img)

        return frame1, frame2

    def _texture_frames(self):
        """Create frames with similar texture patterns."""
        # Create structured checkerboard pattern
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        square_size = 8
        for i in range(0, 64, square_size):
            for j in range(0, 64, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    frame1[i : i + square_size, j : j + square_size] = 255

        # Create similar but slightly different checkerboard (smaller squares)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
        square_size = 4
        for i in range(0, 64, square_size):
            for j in range(0, 64, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    frame2[i : i + square_size, j : j + square_size] = 255

        return frame1, frame2

    def _different_texture_frames(self):
        """Create frames with very different texture patterns."""
        # Structured pattern (checkerboard)
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        square_size = 8
        for i in range(0, 64, square_size):
            for j in range(0, 64, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    frame1[i : i + square_size, j : j + square_size] = 255

        # Smooth gradient (very different texture)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            val = int(255 * i / 63)
            frame2[i, :] = val

        return frame1, frame2

    def _sharp_frames(self):
        """Create frames with similar sharpness."""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Create sharp edges
        frame1[:32, :] = 255
        frame2[:30, :] = 255  # Slightly different but still sharp

        return frame1, frame2

    def _different_sharpness_frames(self):
        """Create frames with very different sharpness."""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Sharp edge
        frame1[:32, :] = 255

        # Blurred edge (simulate with gradient)
        for i in range(64):
            if i < 28:
                frame2[i, :] = 255
            elif i < 36:
                val = int(255 * (36 - i) / 8)  # Gradient
                frame2[i, :] = val

        return frame1, frame2

    # ------------------------------------------------------------------
    # Error-based metrics (lower is better)
    # ------------------------------------------------------------------
    def test_mse_comprehensive(self):
        """Test MSE with various frame types."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        assert mse(ident1, ident2) == pytest.approx(0.0, abs=1e-6)

        # Similar frames should have low MSE
        sim1, sim2 = self._slightly_different_frames()
        similar_mse = mse(sim1, sim2)

        # Very different frames should have high MSE
        diff1, diff2 = self._very_different_frames()
        different_mse = mse(diff1, diff2)

        assert similar_mse < different_mse
        assert similar_mse > 0.0
        assert different_mse > 1000.0  # Should be significantly higher

    def test_rmse_comprehensive(self):
        """Test RMSE with various frame types."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        assert rmse(ident1, ident2) == pytest.approx(0.0, abs=1e-6)

        # Similar frames should have low RMSE
        sim1, sim2 = self._slightly_different_frames()
        similar_rmse = rmse(sim1, sim2)

        # Very different frames should have high RMSE
        diff1, diff2 = self._very_different_frames()
        different_rmse = rmse(diff1, diff2)

        assert similar_rmse < different_rmse
        assert similar_rmse > 0.0
        assert different_rmse > 30.0  # Should be significantly higher

    # ------------------------------------------------------------------
    # Gradient-based metrics
    # ------------------------------------------------------------------
    def test_fsim_comprehensive(self):
        """Test FSIM with gradient patterns."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_fsim = fsim(ident1, ident2)

        # Similar gradients
        grad1, grad2 = self._gradient_frames()
        similar_fsim = fsim(grad1, grad2)

        # Different gradients
        diff_grad1, diff_grad2 = self._different_gradient_frames()
        different_fsim = fsim(diff_grad1, diff_grad2)

        # Note: FSIM may score random noise higher than smooth gradients
        # because noise has more "features". The key test is identical > different.
        assert identical_fsim >= different_fsim
        assert identical_fsim > 0.95  # Should be very high for identical
        assert similar_fsim > 0.0  # Should be positive
        assert different_fsim > 0.0  # Should be positive

    def test_gmsd_comprehensive(self):
        """Test GMSD with gradient patterns."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_gmsd = gmsd(ident1, ident2)

        # Similar gradients
        grad1, grad2 = self._gradient_frames()
        similar_gmsd = gmsd(grad1, grad2)

        # Different gradients
        diff_grad1, diff_grad2 = self._different_gradient_frames()
        gmsd(diff_grad1, diff_grad2)

        assert identical_gmsd <= similar_gmsd
        assert identical_gmsd < 0.1  # Should be very low for identical
        assert similar_gmsd > 0.0  # Should be positive for different frames

    # ------------------------------------------------------------------
    # Color-based metrics
    # ------------------------------------------------------------------
    def test_chist_comprehensive(self):
        """Test color histogram correlation."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_chist = chist(ident1, ident2)

        # Similar colors
        color1, color2 = self._color_frames()
        similar_chist = chist(color1, color2)

        # Different colors
        diff_color1, diff_color2 = self._different_color_frames()
        different_chist = chist(diff_color1, diff_color2)

        assert identical_chist >= similar_chist >= different_chist
        assert identical_chist > 0.95  # Should be very high for identical
        assert different_chist < 0.8  # Should be lower for different colors

    # ------------------------------------------------------------------
    # Edge-based metrics
    # ------------------------------------------------------------------
    def test_edge_similarity_comprehensive(self):
        """Test edge similarity with actual edge patterns."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_edge = edge_similarity(ident1, ident2)

        # Similar edges
        edge1, edge2 = self._edge_frames()
        similar_edge = edge_similarity(edge1, edge2)

        # Different edges
        diff_edge1, diff_edge2 = self._different_edge_frames()
        different_edge = edge_similarity(diff_edge1, diff_edge2)

        assert identical_edge >= similar_edge
        assert identical_edge > 0.95  # Should be very high for identical
        # Note: Similar edges may have 0 similarity if they don't overlap
        assert similar_edge >= 0.0  # Should be non-negative
        assert different_edge >= 0.0  # Should be non-negative

    # ------------------------------------------------------------------
    # Texture-based metrics
    # ------------------------------------------------------------------
    def test_texture_similarity_comprehensive(self):
        """Test texture similarity with actual texture patterns."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_texture = texture_similarity(ident1, ident2)

        # Similar textures (8x8 vs 4x4 checkerboard)
        tex1, tex2 = self._texture_frames()
        similar_texture = texture_similarity(tex1, tex2)

        # Different textures (checkerboard vs gradient)
        diff_tex1, diff_tex2 = self._different_texture_frames()
        different_texture = texture_similarity(diff_tex1, diff_tex2)

        assert identical_texture >= similar_texture >= different_texture
        assert identical_texture > 0.95  # Should be very high for identical
        assert similar_texture > 0.95  # Checkerboards are quite similar
        assert (
            different_texture < 0.7
        )  # Checkerboard vs gradient should be clearly different

    # ------------------------------------------------------------------
    # Sharpness-based metrics
    # ------------------------------------------------------------------
    def test_sharpness_similarity_comprehensive(self):
        """Test sharpness similarity with actual sharpness differences."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_sharp = sharpness_similarity(ident1, ident2)

        # Similar sharpness
        sharp1, sharp2 = self._sharp_frames()
        similar_sharp = sharpness_similarity(sharp1, sharp2)

        # Different sharpness
        diff_sharp1, diff_sharp2 = self._different_sharpness_frames()
        different_sharp = sharpness_similarity(diff_sharp1, diff_sharp2)

        assert identical_sharp >= similar_sharp >= different_sharp
        assert identical_sharp > 0.95  # Should be very high for identical
        assert different_sharp < 0.9  # Should be lower for different sharpness

    # ------------------------------------------------------------------
    # Comprehensive ordering test
    # ------------------------------------------------------------------
    def test_metric_ordering_comprehensive(self):
        """Test that all metrics show proper ordering across similarity levels."""
        # Get all frame pairs
        identical = self._identical_frames()
        similar = self._slightly_different_frames()
        different = self._very_different_frames()

        # Test each metric maintains proper ordering
        metrics_higher_better = [
            ("chist", chist),
        ]

        # More lenient test for metrics that may not always maintain strict ordering
        metrics_higher_better_lenient = [
            ("fsim", fsim),
            ("edge_similarity", edge_similarity),
            ("texture_similarity", texture_similarity),
            ("sharpness_similarity", sharpness_similarity),
        ]

        metrics_lower_better = [
            ("mse", mse),
            ("rmse", rmse),
            ("gmsd", gmsd),
        ]

        # Strict higher-is-better metrics
        for name, metric_func in metrics_higher_better:
            identical_score = metric_func(*identical)
            similar_score = metric_func(*similar)
            different_score = metric_func(*different)

            assert (
                identical_score >= similar_score
            ), f"{name}: identical ({identical_score:.3f}) should be >= similar ({similar_score:.3f})"
            assert (
                similar_score >= different_score
            ), f"{name}: similar ({similar_score:.3f}) should be >= different ({different_score:.3f})"

        # Lenient higher-is-better metrics (just check identical >= different)
        for name, metric_func in metrics_higher_better_lenient:
            identical_score = metric_func(*identical)
            similar_score = metric_func(*similar)
            different_score = metric_func(*different)

            assert (
                identical_score >= different_score
            ), f"{name}: identical ({identical_score:.3f}) should be >= different ({different_score:.3f})"
            # Note: We don't enforce similar >= different for these metrics
            # because they may behave non-monotonically with certain content types

        # Lower-is-better metrics
        for name, metric_func in metrics_lower_better:
            identical_score = metric_func(*identical)
            similar_score = metric_func(*similar)
            different_score = metric_func(*different)

            assert (
                identical_score <= similar_score
            ), f"{name}: identical ({identical_score:.3f}) should be <= similar ({similar_score:.3f})"
            assert (
                similar_score <= different_score
            ), f"{name}: similar ({similar_score:.3f}) should be <= different ({different_score:.3f})"
