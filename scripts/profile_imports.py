#!/usr/bin/env python
"""Profile import times for GifLab modules to identify optimization opportunities."""

import time
import sys
import importlib
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def time_import(module_name: str) -> Tuple[float, bool]:
    """Time the import of a module."""
    try:
        start = time.perf_counter()
        importlib.import_module(module_name)
        end = time.perf_counter()
        return (end - start) * 1000, True  # Return in milliseconds
    except ImportError as e:
        print(f"  ⚠️ Failed to import {module_name}: {e}")
        return 0.0, False


def profile_imports() -> Dict[str, float]:
    """Profile import times for all GifLab modules."""
    print("=" * 60)
    print("GifLab Import Time Profiling")
    print("=" * 60)
    
    # Save initial state of imported modules
    initial_modules = set(sys.modules.keys())
    
    modules_to_profile = [
        # Core modules
        ("giflab", "Core package"),
        ("giflab.config", "Configuration"),
        
        # Metrics modules
        ("giflab.metrics", "Basic metrics"),
        ("giflab.enhanced_metrics", "Enhanced metrics"),
        ("giflab.ssimulacra2_metrics", "SSIMulacra2 metrics"),
        ("giflab.deep_perceptual_metrics", "LPIPS metrics"),
        ("giflab.temporal_artifacts", "Temporal analysis"),
        
        # Validation modules
        ("giflab.optimization_validation", "Optimization validation"),
        ("giflab.text_ui_validation", "Text UI validation"),
        
        # Caching modules
        ("giflab.caching", "Caching system"),
        ("giflab.caching.frame_cache", "Frame cache"),
        ("giflab.caching.validation_cache", "Validation cache"),
        ("giflab.caching.resized_frame_cache", "Resized frame cache"),
        
        # Sampling modules
        ("giflab.sampling", "Sampling system"),
        ("giflab.sampling.frame_sampler", "Frame sampler"),
        ("giflab.sampling.strategies", "Sampling strategies"),
        
        # CLI modules
        ("giflab.cli", "CLI system"),
    ]
    
    results = {}
    total_time = 0.0
    
    # First, measure cold start (Python interpreter startup)
    print("\n📊 Module Import Times:")
    print("-" * 60)
    
    for module_name, description in modules_to_profile:
        import_time, success = time_import(module_name)
        if success:
            results[module_name] = import_time
            total_time += import_time
            status = "✅"
            print(f"{status} {module_name:<40} {import_time:8.2f} ms  # {description}")
        else:
            status = "❌"
            print(f"{status} {module_name:<40} {'N/A':>8}     # {description}")
    
    print("-" * 60)
    print(f"Total import time: {total_time:.2f} ms")
    
    # Now profile heavy dependencies
    print("\n📦 Heavy Dependencies:")
    print("-" * 60)
    
    heavy_deps = [
        ("numpy", "Numerical computing"),
        ("cv2", "OpenCV image processing"),
        ("PIL", "Pillow image library"),
        ("torch", "PyTorch deep learning"),
        ("lpips", "LPIPS perceptual metric"),
        ("scipy", "Scientific computing"),
        ("sklearn", "Scikit-learn ML"),
        ("click", "CLI framework"),
    ]
    
    dep_results = {}
    dep_total = 0.0
    
    for module_name, description in heavy_deps:
        # Skip if already imported (like torch which was imported by LPIPS)
        if module_name in sys.modules:
            print(f"⏭️  {module_name:<20} {'cached':>8}     # {description} (already imported)")
            continue
        
        import_time, success = time_import(module_name)
        if success:
            dep_results[module_name] = import_time
            dep_total += import_time
            print(f"✅ {module_name:<20} {import_time:8.2f} ms  # {description}")
        else:
            print(f"⚠️ {module_name:<20} {'N/A':>8}     # {description} (not installed)")
    
    print("-" * 60)
    print(f"Total dependency import time: {dep_total:.2f} ms")
    
    # Identify bottlenecks
    print("\n🔥 Top 5 Slowest Imports:")
    print("-" * 60)
    
    all_results = {**results, **dep_results}
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (module_name, import_time) in enumerate(sorted_results[:5], 1):
        percentage = (import_time / (total_time + dep_total)) * 100 if (total_time + dep_total) > 0 else 0
        print(f"{i}. {module_name:<30} {import_time:8.2f} ms ({percentage:5.1f}%)")
    
    # Provide optimization recommendations
    print("\n💡 Optimization Recommendations:")
    print("-" * 60)
    
    recommendations = []
    
    # Check for torch/LPIPS
    if "torch" in dep_results and dep_results["torch"] > 100:
        recommendations.append(f"• torch takes {dep_results['torch']:.0f}ms - consider lazy loading for LPIPS")
    
    if "lpips" in dep_results and dep_results["lpips"] > 50:
        recommendations.append(f"• lpips takes {dep_results['lpips']:.0f}ms - defer until LPIPS metric is used")
    
    # Check for cv2
    if "cv2" in dep_results and dep_results["cv2"] > 100:
        recommendations.append(f"• cv2 takes {dep_results['cv2']:.0f}ms - lazy load for image operations")
    
    # Check for scipy/sklearn
    if "scipy" in dep_results and dep_results["scipy"] > 50:
        recommendations.append(f"• scipy takes {dep_results['scipy']:.0f}ms - defer for scientific operations")
    
    if "sklearn" in dep_results and dep_results["sklearn"] > 50:
        recommendations.append(f"• sklearn takes {dep_results['sklearn']:.0f}ms - lazy load for ML features")
    
    # Check GifLab modules
    slow_modules = [m for m, t in results.items() if t > 50]
    if slow_modules:
        recommendations.append(f"• {len(slow_modules)} GifLab modules take >50ms to import")
        recommendations.append("  Consider breaking up large modules or deferring imports")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✨ No major import bottlenecks detected!")
    
    print("\n" + "=" * 60)
    print(f"Grand Total Import Time: {(total_time + dep_total):.2f} ms")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    profile_imports()