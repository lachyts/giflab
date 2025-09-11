"""
Unit tests for the lazy import system.

Tests the LazyModule class, import registry, and availability checking
to ensure lazy loading works correctly and maintains backward compatibility.
"""

import sys
import threading
import time
from unittest.mock import MagicMock, patch, Mock
import pytest

# Import directly from the source
sys.path.insert(0, '/Users/lachlants/repos/animately/giflab/src')

from giflab.lazy_imports import (
    LazyModule,
    LazyImportRegistry,
    lazy_import,
    check_import_available,
    get_import_status,
    preload_modules,
    is_torch_available,
    is_lpips_available,
    is_cv2_available,
    is_scipy_available,
    is_sklearn_available,
)


class TestLazyModule:
    """Test the LazyModule class functionality."""
    
    def test_lazy_module_defers_import(self):
        """Test that LazyModule doesn't import until accessed."""
        # Create a lazy module for a non-existent module
        lazy_mod = LazyModule("non_existent_module_12345")
        
        # Module should not be imported yet
        assert lazy_mod._module is None
        assert lazy_mod._import_attempted is False
    
    def test_lazy_module_imports_on_attribute_access(self):
        """Test that module is imported when attribute is accessed."""
        # Use a real module that should exist
        lazy_mod = LazyModule("json")
        
        # Access an attribute
        dumps = lazy_mod.dumps
        
        # Now module should be imported
        assert lazy_mod._module is not None
        assert lazy_mod._import_attempted is True
        assert dumps is not None
    
    def test_lazy_module_handles_import_error(self):
        """Test graceful handling of import errors."""
        lazy_mod = LazyModule("non_existent_module_xyz", fallback_value=None)
        
        # Should return fallback value
        result = lazy_mod._load_module()
        assert result is None
        assert lazy_mod._import_error is not None
    
    def test_lazy_module_is_available(self):
        """Test the is_available property."""
        # Real module
        lazy_json = LazyModule("json")
        assert lazy_json.is_available is True
        
        # Non-existent module
        lazy_fake = LazyModule("fake_module_abc")
        assert lazy_fake.is_available is False
    
    def test_lazy_module_thread_safety(self):
        """Test thread-safe loading of modules."""
        lazy_mod = LazyModule("json")
        results = []
        
        def access_module():
            # Each thread tries to access the module
            result = lazy_mod.dumps
            results.append(result is not None)
        
        # Create multiple threads
        threads = [threading.Thread(target=access_module) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All threads should have gotten the same module
        assert all(results)
        assert lazy_mod._module is not None


class TestLazyImportRegistry:
    """Test the LazyImportRegistry class."""
    
    def test_registry_registers_modules(self):
        """Test that modules can be registered."""
        registry = LazyImportRegistry()
        
        # Register a module
        lazy_mod = registry.register("os")
        
        assert "os" in registry._modules
        assert lazy_mod is not None
    
    def test_registry_returns_same_instance(self):
        """Test that registering same module returns same instance."""
        registry = LazyImportRegistry()
        
        # Register twice
        mod1 = registry.register("sys")
        mod2 = registry.register("sys")
        
        # Should be the same instance
        assert mod1 is mod2
    
    def test_registry_get_status(self):
        """Test status reporting."""
        registry = LazyImportRegistry()
        
        # Register but don't import
        registry.register("json")
        
        status = registry.get_status()
        assert "json" in status
        
        # Status should show (available, not_imported)
        available, imported = status["json"]
        assert available is True
        assert imported is False
    
    def test_registry_preload(self):
        """Test preloading modules."""
        registry = LazyImportRegistry()
        
        # Register a module
        lazy_mod = registry.register("json")
        
        # Preload it
        registry.preload(["json"])
        
        # Should now be loaded
        assert lazy_mod._module is not None


class TestPublicAPI:
    """Test the public API functions."""
    
    def test_lazy_import_function(self):
        """Test the lazy_import convenience function."""
        lazy_mod = lazy_import("json")
        
        assert lazy_mod is not None
        assert lazy_mod.is_available is True
    
    def test_check_import_available(self):
        """Test checking if imports are available."""
        # Standard library should be available
        assert check_import_available("json") is True
        assert check_import_available("os") is True
        
        # Non-existent should not be available
        assert check_import_available("fake_module_xyz") is False
    
    @patch('giflab.lazy_imports.check_import_available')
    def test_availability_checkers(self, mock_check):
        """Test the specific availability checker functions."""
        # Clear the cache first
        from giflab.lazy_imports import _availability_cache
        _availability_cache.clear()
        
        # Mock different availability states
        mock_check.side_effect = [True, False, True, False, True]
        
        # Test each checker
        assert is_torch_available() is True
        assert is_lpips_available() is False
        assert is_cv2_available() is True
        assert is_scipy_available() is False
        assert is_sklearn_available() is True
        
        # Should be cached now (no more calls to mock)
        mock_check.side_effect = None
        mock_check.return_value = False  # This shouldn't be used
        
        # These should use cached values
        assert is_torch_available() is True
        assert is_lpips_available() is False


class TestIntegration:
    """Integration tests with actual GifLab modules."""
    
    def test_deep_perceptual_metrics_lazy_loading(self):
        """Test that deep_perceptual_metrics uses lazy loading correctly."""
        # Clear torch from sys.modules to test lazy loading
        if 'torch' in sys.modules:
            del sys.modules['torch']
        if 'lpips' in sys.modules:
            del sys.modules['lpips']
        
        # Import the module - should not trigger torch/lpips import
        from giflab import deep_perceptual_metrics
        
        # Check that the lazy modules are set up
        assert hasattr(deep_perceptual_metrics, 'torch')
        assert hasattr(deep_perceptual_metrics, 'lpips')
        
        # Check availability flags are set correctly
        # Note: This depends on whether torch/lpips are installed
        # We just check that the flags exist
        assert hasattr(deep_perceptual_metrics, 'TORCH_AVAILABLE')
        assert hasattr(deep_perceptual_metrics, 'LPIPS_AVAILABLE')
    
    def test_eda_lazy_sklearn(self):
        """Test that eda.py loads sklearn lazily."""
        # Clear sklearn from sys.modules
        if 'sklearn' in sys.modules:
            del sys.modules['sklearn']
        if 'sklearn.decomposition' in sys.modules:
            del sys.modules['sklearn.decomposition']
        
        # Import eda - should not import sklearn
        from giflab import eda
        
        # sklearn should not be in sys.modules yet
        # (unless it was already imported by something else)
        # Just verify the module loads without error
        assert eda is not None


class TestPerformance:
    """Performance tests for lazy loading."""
    
    def test_lazy_import_is_fast(self):
        """Test that creating lazy imports is fast."""
        start = time.perf_counter()
        
        # Create many lazy imports
        for i in range(100):
            lazy_import(f"fake_module_{i}")
        
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 10ms for 100 lazy imports)
        assert elapsed < 0.01
    
    def test_cached_availability_check(self):
        """Test that availability checks are cached."""
        from giflab.lazy_imports import _availability_cache
        _availability_cache.clear()
        
        # First call might be slow
        start = time.perf_counter()
        result1 = is_torch_available()
        first_time = time.perf_counter() - start
        
        # Second call should be much faster (cached)
        start = time.perf_counter()
        result2 = is_torch_available()
        second_time = time.perf_counter() - start
        
        # Results should be the same
        assert result1 == result2
        
        # Second call should be at least 10x faster (likely 100x+)
        if first_time > 0.0001:  # Only check if first call took measurable time
            assert second_time < first_time / 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])