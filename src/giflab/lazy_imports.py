"""
Lazy Import System for GifLab

This module provides a lazy loading mechanism for heavy dependencies
to reduce import time and memory usage. Dependencies are only loaded
when actually used.

Key Features:
- Thread-safe lazy loading
- Import availability checking
- Graceful fallback for missing dependencies
- Zero overhead after first import
- Maintains backward compatibility
"""

import importlib
import logging
import sys
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LazyModule:
    """
    A proxy for a module that defers import until first attribute access.
    
    This class acts as a transparent proxy that loads the actual module
    only when an attribute is accessed, reducing startup time for heavy
    dependencies that may not be used in every execution path.
    """
    
    def __init__(self, module_name: str, fallback_value: Any = None):
        """
        Initialize a lazy module proxy.
        
        Args:
            module_name: Full name of the module to lazy load
            fallback_value: Value to return if module import fails
        """
        self._module_name = module_name
        self._module: Optional[Any] = None
        self._fallback_value = fallback_value
        self._lock = threading.Lock()
        self._import_attempted = False
        self._import_error: Optional[Exception] = None
    
    def _load_module(self) -> Optional[Any]:
        """Thread-safe module loading."""
        if self._module is not None:
            return self._module
            
        with self._lock:
            # Double-check pattern for thread safety
            if self._module is not None:
                return self._module
                
            if self._import_attempted:
                # Already tried and failed
                if self._import_error:
                    return self._fallback_value
                return self._module
            
            self._import_attempted = True
            
            try:
                logger.debug(f"Lazy loading module: {self._module_name}")
                self._module = importlib.import_module(self._module_name)
                logger.debug(f"Successfully loaded: {self._module_name}")
                return self._module
            except ImportError as e:
                logger.debug(f"Failed to import {self._module_name}: {e}")
                self._import_error = e
                return self._fallback_value
    
    def __getattr__(self, name: str) -> Any:
        """Load module on first attribute access."""
        module = self._load_module()
        if module is None:
            if self._fallback_value is not None:
                return getattr(self._fallback_value, name, None)
            raise ImportError(
                f"Module '{self._module_name}' is not available and no fallback provided"
            )
        return getattr(module, name)
    
    def __bool__(self) -> bool:
        """Check if module is available."""
        module = self._load_module()
        return module is not None
    
    @property
    def is_available(self) -> bool:
        """Check if module can be imported."""
        module = self._load_module()
        return module is not None


class LazyImportRegistry:
    """
    Registry for managing lazy imports across the application.
    
    Provides centralized management of lazy loaded modules with
    import status tracking and diagnostic capabilities.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._modules: Dict[str, LazyModule] = {}
        self._lock = threading.Lock()
    
    def register(self, module_name: str, fallback_value: Any = None) -> LazyModule:
        """
        Register a module for lazy loading.
        
        Args:
            module_name: Full name of the module
            fallback_value: Optional fallback if import fails
            
        Returns:
            LazyModule proxy for the module
        """
        with self._lock:
            if module_name not in self._modules:
                self._modules[module_name] = LazyModule(module_name, fallback_value)
            return self._modules[module_name]
    
    def get_status(self) -> Dict[str, Tuple[bool, bool]]:
        """
        Get import status for all registered modules.
        
        Returns:
            Dict mapping module names to (registered, imported) tuples
        """
        status = {}
        for name, module in self._modules.items():
            imported = module._module is not None
            available = module.is_available
            status[name] = (available, imported)
        return status
    
    def preload(self, module_names: Optional[list] = None) -> None:
        """
        Preload specified modules or all registered modules.
        
        Args:
            module_names: List of module names to preload, or None for all
        """
        if module_names is None:
            module_names = list(self._modules.keys())
        
        for name in module_names:
            if name in self._modules:
                # Trigger import by checking availability
                _ = self._modules[name].is_available


# Global registry instance
_registry = LazyImportRegistry()


def lazy_import(module_name: str, fallback_value: Any = None) -> LazyModule:
    """
    Create a lazy import proxy for a module.
    
    This is the main entry point for creating lazy imports. The module
    will not be imported until an attribute is accessed.
    
    Args:
        module_name: Full name of the module to import
        fallback_value: Optional fallback if import fails
        
    Returns:
        LazyModule proxy that will load the module on first use
        
    Example:
        >>> torch = lazy_import('torch')
        >>> # torch is not imported yet
        >>> tensor = torch.tensor([1, 2, 3])  # Now torch is imported
    """
    return _registry.register(module_name, fallback_value)


def check_import_available(module_name: str) -> bool:
    """
    Check if a module can be imported without actually importing it.
    
    Uses importlib.util to check module availability without side effects.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        True if module can be imported, False otherwise
    """
    import importlib.util
    
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def get_import_status() -> Dict[str, Tuple[bool, bool]]:
    """
    Get the current status of all lazy imports.
    
    Returns:
        Dict mapping module names to (available, imported) status tuples
    """
    return _registry.get_status()


def preload_modules(module_names: Optional[list] = None) -> None:
    """
    Preload lazy modules to avoid delays during execution.
    
    Useful for warming up imports in background threads or during
    application initialization when you know certain modules will be needed.
    
    Args:
        module_names: List of modules to preload, or None for all registered
    """
    _registry.preload(module_names)


# Convenience lazy imports for heavy dependencies
def get_torch() -> Any:
    """Get torch module with lazy loading."""
    return lazy_import('torch')


def get_lpips() -> Any:
    """Get lpips module with lazy loading."""
    return lazy_import('lpips')


def get_cv2() -> Any:
    """Get cv2 module with lazy loading."""
    return lazy_import('cv2')


def get_scipy() -> Any:
    """Get scipy module with lazy loading."""
    return lazy_import('scipy')


def get_sklearn() -> Any:
    """Get sklearn module with lazy loading."""
    return lazy_import('sklearn')


# Thread-safe availability flags
_availability_cache: Dict[str, bool] = {}
_availability_lock = threading.Lock()


def is_torch_available() -> bool:
    """Check if torch is available for import."""
    with _availability_lock:
        if 'torch' not in _availability_cache:
            _availability_cache['torch'] = check_import_available('torch')
        return _availability_cache['torch']


def is_lpips_available() -> bool:
    """Check if lpips is available for import."""
    with _availability_lock:
        if 'lpips' not in _availability_cache:
            _availability_cache['lpips'] = check_import_available('lpips')
        return _availability_cache['lpips']


def is_cv2_available() -> bool:
    """Check if cv2 is available for import."""
    with _availability_lock:
        if 'cv2' not in _availability_cache:
            _availability_cache['cv2'] = check_import_available('cv2')
        return _availability_cache['cv2']


def is_scipy_available() -> bool:
    """Check if scipy is available for import."""
    with _availability_lock:
        if 'scipy' not in _availability_cache:
            _availability_cache['scipy'] = check_import_available('scipy')
        return _availability_cache['scipy']


def is_sklearn_available() -> bool:
    """Check if sklearn is available for import."""
    with _availability_lock:
        if 'sklearn' not in _availability_cache:
            _availability_cache['sklearn'] = check_import_available('sklearn')
        return _availability_cache['sklearn']