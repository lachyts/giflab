#!/usr/bin/env python
"""
Benchmark script to measure import time improvements from lazy loading.

Compares import times before and after lazy loading implementation.
"""

import subprocess
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def measure_import_time(module_name: str, setup_code: str = "") -> float:
    """Measure the time to import a module."""
    code = f"""
import sys
import time
sys.path.insert(0, '{project_root / "src"}')
{setup_code}
start = time.perf_counter()
import {module_name}
end = time.perf_counter()
print(end - start)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ["poetry", "run", "python", temp_file],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            print(f"Error importing {module_name}: {result.stderr}")
            return -1
    finally:
        Path(temp_file).unlink()


def benchmark_imports():
    """Run import time benchmarks."""
    print("=" * 70)
    print("GifLab Import Time Benchmark - Lazy Loading Improvements")
    print("=" * 70)
    
    # Modules to benchmark
    modules_to_test = [
        ("giflab", "Core package"),
        ("giflab.deep_perceptual_metrics", "LPIPS metrics (with lazy torch/lpips)"),
        ("giflab.metrics", "Basic metrics"),
        ("giflab.eda", "EDA utilities (with lazy sklearn)"),
        ("giflab.cli", "CLI system"),
        ("giflab.caching", "Caching system"),
        ("giflab.sampling", "Sampling system"),
    ]
    
    print("\n📊 Import Time Measurements (average of 3 runs):")
    print("-" * 70)
    print(f"{'Module':<45} {'Time (ms)':<12} {'Description'}")
    print("-" * 70)
    
    total_time = 0
    results = {}
    
    for module_name, description in modules_to_test:
        # Run multiple times and take average
        times = []
        for _ in range(3):
            t = measure_import_time(module_name)
            if t > 0:
                times.append(t)
        
        if times:
            avg_time = sum(times) / len(times) * 1000  # Convert to ms
            results[module_name] = avg_time
            total_time += avg_time
            print(f"{module_name:<45} {avg_time:>8.2f} ms  {description}")
        else:
            print(f"{module_name:<45} {'ERROR':>8}     {description}")
    
    print("-" * 70)
    print(f"{'Total import time:':<45} {total_time:>8.2f} ms")
    
    # Compare with baseline if available
    print("\n📈 Comparison with Baseline (before lazy loading):")
    print("-" * 70)
    
    # Baseline measurements from initial profiling
    baseline = {
        "giflab": 1585.22,
        "giflab.deep_perceptual_metrics": 1325.96,
        "giflab.metrics": 0.01,
        "giflab.cli": 9.75,
        "giflab.caching": 0.00,
        "giflab.sampling": 3.03,
    }
    
    improvements = []
    for module, current_time in results.items():
        if module in baseline:
            baseline_time = baseline[module]
            if baseline_time > 0:
                improvement = (baseline_time - current_time) / baseline_time * 100
                speedup = baseline_time / current_time if current_time > 0 else 0
                improvements.append((module, baseline_time, current_time, improvement, speedup))
    
    if improvements:
        print(f"{'Module':<35} {'Before':<10} {'After':<10} {'Improvement':<12} {'Speedup'}")
        print("-" * 70)
        for module, before, after, improvement, speedup in improvements:
            # Truncate module name if too long
            display_name = module if len(module) <= 34 else module[:31] + "..."
            print(f"{display_name:<35} {before:>6.1f} ms  {after:>6.1f} ms  {improvement:>6.1f}%      {speedup:>5.1f}x")
    
    # Calculate overall improvement
    total_baseline = sum(baseline.values())
    total_current = sum(results.get(m, 0) for m in baseline.keys())
    
    if total_baseline > 0 and total_current > 0:
        total_improvement = (total_baseline - total_current) / total_baseline * 100
        total_speedup = total_baseline / total_current
        
        print("-" * 70)
        print(f"{'TOTAL':<35} {total_baseline:>6.1f} ms  {total_current:>6.1f} ms  {total_improvement:>6.1f}%      {total_speedup:>5.1f}x")
    
    # Test that lazy loading is working
    print("\n✅ Lazy Loading Verification:")
    print("-" * 70)
    
    # Check if torch is imported when just importing giflab
    code = """
import sys
sys.path.insert(0, '{root}')
import giflab
print('torch' in sys.modules)
""".format(root=project_root / "src")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ["poetry", "run", "python", temp_file],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        torch_imported = result.stdout.strip() == "True"
        
        if torch_imported:
            print("⚠️  WARNING: torch is being imported at startup (lazy loading may not be working)")
        else:
            print("✅ SUCCESS: torch is NOT imported at startup (lazy loading is working)")
    finally:
        Path(temp_file).unlink()
    
    # Check sklearn
    code = """
import sys
sys.path.insert(0, '{root}')
import giflab.eda
print('sklearn' in sys.modules)
""".format(root=project_root / "src")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ["poetry", "run", "python", temp_file],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        sklearn_imported = result.stdout.strip() == "True"
        
        if sklearn_imported:
            print("⚠️  WARNING: sklearn is being imported when importing eda")
        else:
            print("✅ SUCCESS: sklearn is NOT imported when importing eda (lazy loading is working)")
    finally:
        Path(temp_file).unlink()
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_imports()