"""Cache management commands for GifLab frame caching system."""

import json
from pathlib import Path

import click

from ..caching import get_frame_cache, get_validation_cache
from .utils import handle_generic_error


@click.group("cache")
def cache() -> None:
    """Manage the GifLab frame cache for performance optimization."""
    pass


@cache.command("status")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output cache statistics in JSON format"
)
def cache_status(output_json: bool) -> None:
    """Display frame cache statistics and status.
    
    Shows cache hit rates, memory usage, disk usage, and other metrics
    that help understand cache performance.
    
    Examples:
    
        # View cache status
        giflab cache status
        
        # Get machine-readable JSON output
        giflab cache status --json
    """
    try:
        frame_cache = get_frame_cache()
        stats = frame_cache.get_stats()
        
        if output_json:
            output = {
                "enabled": frame_cache.enabled,
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions,
                "memory_bytes": stats.memory_bytes,
                "memory_mb": round(stats.memory_bytes / (1024 * 1024), 2),
                "disk_entries": stats.disk_entries,
                "total_accesses": stats.total_accesses,
                "memory_limit_mb": frame_cache.memory_limit_bytes / (1024 * 1024),
                "disk_limit_mb": frame_cache.disk_limit_bytes / (1024 * 1024),
                "ttl_hours": frame_cache.ttl_seconds / 3600,
                "disk_path": str(frame_cache.disk_path)
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("📊 Frame Cache Status")
            click.echo("=" * 50)
            
            if not frame_cache.enabled:
                click.echo("❌ Cache is DISABLED")
                click.echo("   Enable it by setting FRAME_CACHE['enabled'] = True in config")
                return
            
            click.echo("✅ Cache is ENABLED")
            click.echo()
            
            # Performance metrics
            click.echo("Performance Metrics:")
            click.echo(f"  Total Accesses: {stats.total_accesses:,}")
            click.echo(f"  Cache Hits:     {stats.hits:,}")
            click.echo(f"  Cache Misses:   {stats.misses:,}")
            click.echo(f"  Hit Rate:       {stats.hit_rate:.1%}")
            click.echo(f"  Evictions:      {stats.evictions:,}")
            click.echo()
            
            # Memory usage
            memory_mb = stats.memory_bytes / (1024 * 1024)
            memory_limit_mb = frame_cache.memory_limit_bytes / (1024 * 1024)
            memory_percent = (stats.memory_bytes / frame_cache.memory_limit_bytes * 100) if frame_cache.memory_limit_bytes > 0 else 0
            
            click.echo("Memory Cache:")
            click.echo(f"  Usage:          {memory_mb:.1f} MB / {memory_limit_mb:.0f} MB ({memory_percent:.1f}%)")
            click.echo()
            
            # Disk cache
            click.echo("Disk Cache:")
            click.echo(f"  Entries:        {stats.disk_entries:,}")
            click.echo(f"  Path:           {frame_cache.disk_path}")
            click.echo(f"  Limit:          {frame_cache.disk_limit_bytes / (1024 * 1024):.0f} MB")
            click.echo(f"  TTL:            {frame_cache.ttl_seconds / 3600:.0f} hours")
            
            if stats.total_accesses > 0:
                click.echo()
                if stats.hit_rate > 0.8:
                    click.echo("💚 Excellent cache performance!")
                elif stats.hit_rate > 0.5:
                    click.echo("🟡 Good cache performance")
                else:
                    click.echo("🔴 Low cache hit rate - consider warming cache")
    
    except Exception as e:
        handle_generic_error("Cache status", e)


@cache.command("clear")
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
@click.option(
    "--memory-only",
    is_flag=True,
    help="Clear only the in-memory cache, keep disk cache"
)
def cache_clear(confirm: bool, memory_only: bool) -> None:
    """Clear the frame cache.
    
    Removes all cached frames from memory and optionally from disk.
    This can help free up space or resolve cache-related issues.
    
    Examples:
    
        # Clear all caches with confirmation
        giflab cache clear
        
        # Clear without confirmation
        giflab cache clear --confirm
        
        # Clear only memory cache
        giflab cache clear --memory-only --confirm
    """
    try:
        if not confirm:
            if memory_only:
                message = "Clear in-memory frame cache?"
            else:
                message = "Clear all frame cache (memory and disk)?"
            
            if not click.confirm(message):
                click.echo("Cancelled.")
                return
        
        frame_cache = get_frame_cache()
        
        if memory_only:
            # Just clear memory cache
            with frame_cache._lock:
                old_size = frame_cache._memory_bytes
                frame_cache._memory_cache.clear()
                frame_cache._memory_bytes = 0
                frame_cache._stats.memory_bytes = 0
            
            click.echo(f"✅ Cleared in-memory cache (freed {old_size / (1024 * 1024):.1f} MB)")
        else:
            # Clear everything
            stats = frame_cache.get_stats()
            old_memory_mb = stats.memory_bytes / (1024 * 1024)
            old_disk_entries = stats.disk_entries
            
            frame_cache.clear()
            
            click.echo(f"✅ Cleared frame cache:")
            click.echo(f"   Memory: freed {old_memory_mb:.1f} MB")
            click.echo(f"   Disk: removed {old_disk_entries:,} entries")
    
    except Exception as e:
        handle_generic_error("Cache clear", e)


@cache.command("warm")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recursively warm cache for all GIFs in directory"
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum frames to cache per GIF"
)
@click.option(
    "--pattern",
    default="*.gif",
    help="File pattern to match (default: *.gif)"
)
def cache_warm(path: Path, recursive: bool, max_frames: int | None, pattern: str) -> None:
    """Pre-load GIF files into the cache for better performance.
    
    This command extracts frames from specified GIF files and stores them
    in the cache, improving performance for subsequent operations.
    
    Examples:
    
        # Warm cache for a single file
        giflab cache warm animation.gif
        
        # Warm cache for all GIFs in a directory
        giflab cache warm data/gifs/ --recursive
        
        # Warm cache with frame limit
        giflab cache warm data/ --recursive --max-frames 30
        
        # Warm cache for specific pattern
        giflab cache warm data/ --recursive --pattern "*test*.gif"
    """
    try:
        frame_cache = get_frame_cache()
        
        if not frame_cache.enabled:
            click.echo("❌ Cache is disabled. Enable it in config to use warming.")
            return
        
        # Collect files to warm
        gif_files = []
        
        if path.is_file():
            if path.suffix.lower() == ".gif":
                gif_files.append(path)
            else:
                click.echo(f"⚠️  {path} is not a GIF file")
                return
        elif path.is_dir():
            if recursive:
                gif_files = list(path.rglob(pattern))
            else:
                gif_files = list(path.glob(pattern))
        
        if not gif_files:
            click.echo(f"No GIF files found matching pattern: {pattern}")
            return
        
        click.echo(f"🔥 Warming cache for {len(gif_files)} GIF files...")
        
        # Get initial stats
        initial_stats = frame_cache.get_stats()
        
        # Warm the cache
        with click.progressbar(gif_files, label="Processing") as bar:
            frame_cache.warm_cache(list(bar), max_frames=max_frames)
        
        # Get final stats
        final_stats = frame_cache.get_stats()
        
        # Report results
        new_entries = final_stats.disk_entries - initial_stats.disk_entries
        memory_increase_mb = (final_stats.memory_bytes - initial_stats.memory_bytes) / (1024 * 1024)
        
        click.echo()
        click.echo(f"✅ Cache warming complete:")
        click.echo(f"   Files processed: {len(gif_files)}")
        click.echo(f"   New cache entries: {new_entries}")
        click.echo(f"   Memory usage increase: {memory_increase_mb:.1f} MB")
        
        if final_stats.hit_rate > initial_stats.hit_rate:
            improvement = (final_stats.hit_rate - initial_stats.hit_rate) * 100
            click.echo(f"   Hit rate improved: +{improvement:.1f}%")
    
    except Exception as e:
        handle_generic_error("Cache warm", e)


@cache.command("invalidate")
@click.argument("gif_path", type=click.Path(exists=True, path_type=Path))
def cache_invalidate(gif_path: Path) -> None:
    """Invalidate cache entry for a specific GIF file.
    
    Removes the cached frames for a specific file. Useful when a file
    has been modified and you want to force re-extraction.
    
    Examples:
    
        # Invalidate cache for a specific file
        giflab cache invalidate animation.gif
    """
    try:
        if not gif_path.suffix.lower() == ".gif":
            click.echo(f"⚠️  {gif_path} is not a GIF file")
            return
        
        frame_cache = get_frame_cache()
        
        if not frame_cache.enabled:
            click.echo("❌ Cache is disabled")
            return
        
        frame_cache.invalidate(gif_path)
        click.echo(f"✅ Invalidated cache for: {gif_path.name}")
    
    except Exception as e:
        handle_generic_error("Cache invalidate", e)


@cache.command("config")
def cache_config() -> None:
    """Display current cache configuration settings.
    
    Shows the configuration values being used by the frame cache,
    including memory limits, disk limits, TTL, and paths.
    
    Examples:
    
        # View cache configuration
        giflab cache config
    """
    try:
        from ..config import FRAME_CACHE
        
        frame_cache = get_frame_cache()
        
        click.echo("⚙️  Frame Cache Configuration")
        click.echo("=" * 50)
        
        # Config from module
        click.echo("Configuration (from config.py):")
        for key, value in FRAME_CACHE.items():
            if key == "disk_path" and value is None:
                value = "~/.giflab_cache/frame_cache.db (default)"
            click.echo(f"  {key:20s}: {value}")
        
        click.echo()
        
        # Runtime values
        click.echo("Runtime Values:")
        click.echo(f"  {'Enabled':20s}: {frame_cache.enabled}")
        click.echo(f"  {'Memory Limit':20s}: {frame_cache.memory_limit_bytes / (1024 * 1024):.0f} MB")
        click.echo(f"  {'Disk Limit':20s}: {frame_cache.disk_limit_bytes / (1024 * 1024):.0f} MB")
        click.echo(f"  {'TTL':20s}: {frame_cache.ttl_seconds / 3600:.0f} hours")
        click.echo(f"  {'Disk Path':20s}: {frame_cache.disk_path}")
        
        # Check if path exists and get size
        if frame_cache.disk_path.exists():
            size_mb = frame_cache.disk_path.stat().st_size / (1024 * 1024)
            click.echo(f"  {'Disk File Size':20s}: {size_mb:.1f} MB")
        
        click.echo()
        click.echo("💡 To modify configuration, edit FRAME_CACHE in src/giflab/config.py")
    
    except Exception as e:
        handle_generic_error("Cache config", e)


@cache.command("resize-status")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output resize cache statistics in JSON format"
)
def cache_resize_status(output_json: bool) -> None:
    """Display resized frame cache statistics and status.
    
    Shows cache hit rates, memory usage, buffer pool statistics, and other metrics
    specific to the resized frame cache for performance optimization.
    
    Examples:
    
        # View resize cache status
        giflab cache resize-status
        
        # Get machine-readable JSON output
        giflab cache resize-status --json
    """
    try:
        from ..caching.resized_frame_cache import get_resize_cache
        from ..config import FRAME_CACHE
        
        if not FRAME_CACHE.get("resize_cache_enabled", True):
            click.echo("❌ Resize cache is DISABLED")
            click.echo("   Enable it by setting FRAME_CACHE['resize_cache_enabled'] = True in config")
            return
        
        resize_cache = get_resize_cache()
        stats = resize_cache.get_stats()
        
        if output_json:
            output = {
                "enabled": FRAME_CACHE.get("resize_cache_enabled", True),
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": stats["hit_rate"],
                "evictions": stats["evictions"],
                "ttl_evictions": stats["ttl_evictions"],
                "entries": stats["entries"],
                "memory_mb": stats["memory_mb"],
                "memory_limit_mb": stats["memory_limit_mb"],
                "avg_hit_count": stats["avg_hit_count"],
            }
            if "buffer_pool" in stats:
                output["buffer_pool"] = stats["buffer_pool"]
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("📊 Resized Frame Cache Status")
            click.echo("=" * 50)
            
            click.echo("✅ Resize cache is ENABLED")
            click.echo()
            
            # Performance metrics
            click.echo("Performance Metrics:")
            click.echo(f"  Total Requests: {stats['hits'] + stats['misses']:,}")
            click.echo(f"  Cache Hits:     {stats['hits']:,}")
            click.echo(f"  Cache Misses:   {stats['misses']:,}")
            click.echo(f"  Hit Rate:       {stats['hit_rate']:.1%}")
            click.echo(f"  Evictions:      {stats['evictions']:,}")
            click.echo(f"  TTL Evictions:  {stats['ttl_evictions']:,}")
            click.echo()
            
            # Memory usage
            memory_percent = (stats['memory_mb'] / stats['memory_limit_mb'] * 100) if stats['memory_limit_mb'] > 0 else 0
            
            click.echo("Memory Usage:")
            click.echo(f"  Entries:        {stats['entries']:,}")
            click.echo(f"  Usage:          {stats['memory_mb']:.1f} MB / {stats['memory_limit_mb']:.0f} MB ({memory_percent:.1f}%)")
            click.echo(f"  Avg Hit Count:  {stats['avg_hit_count']:.1f}")
            
            # Buffer pool stats if available
            if "buffer_pool" in stats:
                pool_stats = stats["buffer_pool"]
                click.echo()
                click.echo("Buffer Pool Statistics:")
                click.echo(f"  Allocations:    {pool_stats['allocations']:,}")
                click.echo(f"  Reuses:         {pool_stats['reuses']:,}")
                click.echo(f"  Reuse Rate:     {pool_stats['reuse_rate']:.1%}")
                click.echo(f"  Pool Memory:    {pool_stats['total_memory_mb']:.1f} MB")
            
            if stats['hit_rate'] > 0.8:
                click.echo()
                click.echo("💚 Excellent resize cache performance!")
            elif stats['hit_rate'] > 0.5:
                click.echo()
                click.echo("🟡 Good resize cache performance")
            elif stats['hits'] + stats['misses'] > 0:
                click.echo()
                click.echo("🔴 Low resize cache hit rate")
    
    except Exception as e:
        handle_generic_error("Resize cache status", e)


@cache.command("resize-clear")
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
def cache_resize_clear(confirm: bool) -> None:
    """Clear the resized frame cache.
    
    Removes all cached resized frames from memory and clears buffer pools.
    This can help free up memory or resolve resize cache-related issues.
    
    Examples:
    
        # Clear resize cache with confirmation
        giflab cache resize-clear
        
        # Clear without confirmation
        giflab cache resize-clear --confirm
    """
    try:
        from ..caching.resized_frame_cache import get_resize_cache
        from ..config import FRAME_CACHE
        
        if not FRAME_CACHE.get("resize_cache_enabled", True):
            click.echo("❌ Resize cache is disabled")
            return
        
        if not confirm:
            if not click.confirm("Clear resized frame cache?"):
                click.echo("Cancelled.")
                return
        
        resize_cache = get_resize_cache()
        stats = resize_cache.get_stats()
        old_memory_mb = stats["memory_mb"]
        old_entries = stats["entries"]
        
        resize_cache.clear()
        
        click.echo(f"✅ Cleared resized frame cache:")
        click.echo(f"   Entries removed: {old_entries:,}")
        click.echo(f"   Memory freed: {old_memory_mb:.1f} MB")
    
    except Exception as e:
        handle_generic_error("Resize cache clear", e)


@cache.command("resize-top")
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top entries to show (default: 10)"
)
def cache_resize_top(top_n: int) -> None:
    """Show the most frequently used resize cache entries.
    
    Displays the top cached resize operations by hit count to help understand
    which frame sizes are being used most frequently.
    
    Examples:
    
        # Show top 10 most used resize operations
        giflab cache resize-top
        
        # Show top 20 entries
        giflab cache resize-top --top-n 20
    """
    try:
        from ..caching.resized_frame_cache import get_resize_cache
        from ..config import FRAME_CACHE
        
        if not FRAME_CACHE.get("resize_cache_enabled", True):
            click.echo("❌ Resize cache is disabled")
            return
        
        resize_cache = get_resize_cache()
        top_entries = resize_cache.get_most_used(top_n)
        
        if not top_entries:
            click.echo("No entries in resize cache")
            return
        
        click.echo(f"🏆 Top {len(top_entries)} Most Used Resize Cache Entries")
        click.echo("=" * 60)
        click.echo(f"{'Rank':<6} {'Size':<15} {'Interpolation':<15} {'Hits':<8} {'Memory'}")
        click.echo("-" * 60)
        
        for i, entry in enumerate(top_entries, 1):
            size_str = f"{entry['size'][0]}x{entry['size'][1]}"
            click.echo(
                f"{i:<6} {size_str:<15} {entry['interpolation']:<15} "
                f"{entry['hit_count']:<8} {entry['memory_kb']:.1f} KB"
            )
        
        click.echo()
        stats = resize_cache.get_stats()
        click.echo(f"Total cache hit rate: {stats['hit_rate']:.1%}")
    
    except Exception as e:
        handle_generic_error("Resize cache top", e)


@cache.command("validation-status")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output validation cache statistics in JSON format"
)
def cache_validation_status(output_json: bool) -> None:
    """Display validation cache statistics and status.
    
    Shows cache hit rates, memory usage, disk usage, and metric-specific
    statistics for the validation result cache.
    
    Examples:
    
        # View validation cache status
        giflab cache validation-status
        
        # Get machine-readable JSON output
        giflab cache validation-status --json
    """
    try:
        from ..config import VALIDATION_CACHE
        
        if not VALIDATION_CACHE.get("enabled", True):
            click.echo("❌ Validation cache is DISABLED")
            click.echo("   Enable it by setting VALIDATION_CACHE['enabled'] = True in config")
            return
        
        validation_cache = get_validation_cache()
        stats = validation_cache.get_stats()
        metric_stats = validation_cache.get_metric_stats()
        
        if output_json:
            output = {
                "enabled": validation_cache.enabled,
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions,
                "memory_entries": stats.memory_entries,
                "memory_bytes": stats.memory_bytes,
                "memory_mb": round(stats.memory_bytes / (1024 * 1024), 2),
                "disk_entries": stats.disk_entries,
                "disk_bytes": stats.disk_bytes,
                "disk_mb": round(stats.disk_bytes / (1024 * 1024), 2),
                "memory_limit_mb": validation_cache.memory_limit_bytes / (1024 * 1024),
                "disk_limit_mb": validation_cache.disk_limit_bytes / (1024 * 1024),
                "ttl_hours": validation_cache.ttl_seconds / 3600,
                "disk_path": str(validation_cache.disk_path),
                "metrics_breakdown": metric_stats,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("📊 Validation Cache Status")
            click.echo("=" * 50)
            
            click.echo("✅ Validation cache is ENABLED")
            click.echo()
            
            # Performance metrics
            total_accesses = stats.hits + stats.misses
            click.echo("Performance Metrics:")
            click.echo(f"  Total Accesses: {total_accesses:,}")
            click.echo(f"  Cache Hits:     {stats.hits:,}")
            click.echo(f"  Cache Misses:   {stats.misses:,}")
            click.echo(f"  Hit Rate:       {stats.hit_rate:.1%}")
            click.echo(f"  Evictions:      {stats.evictions:,}")
            click.echo()
            
            # Memory cache
            memory_mb = stats.memory_bytes / (1024 * 1024)
            memory_limit_mb = validation_cache.memory_limit_bytes / (1024 * 1024)
            memory_percent = (stats.memory_bytes / validation_cache.memory_limit_bytes * 100) if validation_cache.memory_limit_bytes > 0 else 0
            
            click.echo("Memory Cache:")
            click.echo(f"  Entries:        {stats.memory_entries:,}")
            click.echo(f"  Usage:          {memory_mb:.1f} MB / {memory_limit_mb:.0f} MB ({memory_percent:.1f}%)")
            click.echo()
            
            # Disk cache
            disk_mb = stats.disk_bytes / (1024 * 1024)
            disk_limit_mb = validation_cache.disk_limit_bytes / (1024 * 1024)
            disk_percent = (stats.disk_bytes / validation_cache.disk_limit_bytes * 100) if validation_cache.disk_limit_bytes > 0 else 0
            
            click.echo("Disk Cache:")
            click.echo(f"  Entries:        {stats.disk_entries:,}")
            click.echo(f"  Usage:          {disk_mb:.1f} MB / {disk_limit_mb:.0f} MB ({disk_percent:.1f}%)")
            click.echo(f"  Path:           {validation_cache.disk_path}")
            click.echo(f"  TTL:            {validation_cache.ttl_seconds / 3600:.0f} hours")
            click.echo()
            
            # Metrics breakdown
            if metric_stats:
                click.echo("Metrics Breakdown:")
                for metric_type, count in metric_stats.items():
                    click.echo(f"  {metric_type:15s}: {count:,} entries")
                click.echo()
            
            # Performance assessment
            if total_accesses > 0:
                if stats.hit_rate > 0.8:
                    click.echo("💚 Excellent validation cache performance!")
                elif stats.hit_rate > 0.5:
                    click.echo("🟡 Good validation cache performance")
                else:
                    click.echo("🔴 Low cache hit rate - consider pre-warming cache")
    
    except Exception as e:
        handle_generic_error("Validation cache status", e)


@cache.command("validation-clear")
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
@click.option(
    "--memory-only",
    is_flag=True,
    help="Clear only the in-memory cache, keep disk cache"
)
@click.option(
    "--metric",
    type=str,
    help="Clear only entries for a specific metric type (e.g., ssim, lpips)"
)
def cache_validation_clear(confirm: bool, memory_only: bool, metric: str | None) -> None:
    """Clear the validation cache.
    
    Removes cached validation results from memory and optionally from disk.
    Can target specific metric types or clear everything.
    
    Examples:
    
        # Clear all validation cache with confirmation
        giflab cache validation-clear
        
        # Clear without confirmation
        giflab cache validation-clear --confirm
        
        # Clear only memory cache
        giflab cache validation-clear --memory-only --confirm
        
        # Clear only LPIPS entries
        giflab cache validation-clear --metric lpips --confirm
    """
    try:
        from ..config import VALIDATION_CACHE
        
        if not VALIDATION_CACHE.get("enabled", True):
            click.echo("❌ Validation cache is disabled")
            return
        
        validation_cache = get_validation_cache()
        
        # Determine action description
        if metric:
            action = f"Clear all {metric} entries from validation cache?"
        elif memory_only:
            action = "Clear in-memory validation cache?"
        else:
            action = "Clear all validation cache (memory and disk)?"
        
        if not confirm:
            if not click.confirm(action):
                click.echo("Cancelled.")
                return
        
        # Get initial stats
        initial_stats = validation_cache.get_stats()
        
        if metric:
            # Clear specific metric type
            validation_cache.invalidate_by_metric(metric)
            click.echo(f"✅ Cleared all {metric} entries from validation cache")
        elif memory_only:
            # Clear only memory cache
            with validation_cache._lock:
                old_size = validation_cache._memory_bytes
                old_entries = len(validation_cache._memory_cache)
                validation_cache._memory_cache.clear()
                validation_cache._memory_bytes = 0
            
            click.echo(f"✅ Cleared in-memory validation cache:")
            click.echo(f"   Entries removed: {old_entries:,}")
            click.echo(f"   Memory freed: {old_size / (1024 * 1024):.1f} MB")
        else:
            # Clear everything
            old_memory_mb = initial_stats.memory_bytes / (1024 * 1024)
            old_disk_entries = initial_stats.disk_entries
            
            validation_cache.clear()
            
            click.echo(f"✅ Cleared validation cache:")
            click.echo(f"   Memory: freed {old_memory_mb:.1f} MB")
            click.echo(f"   Disk: removed {old_disk_entries:,} entries")
    
    except Exception as e:
        handle_generic_error("Validation cache clear", e)


@cache.command("validation-config")
def cache_validation_config() -> None:
    """Display current validation cache configuration settings.
    
    Shows the configuration values being used by the validation cache,
    including memory limits, disk limits, TTL, and metric-specific settings.
    
    Examples:
    
        # View validation cache configuration
        giflab cache validation-config
    """
    try:
        from ..config import VALIDATION_CACHE
        
        validation_cache = get_validation_cache()
        
        click.echo("⚙️  Validation Cache Configuration")
        click.echo("=" * 50)
        
        # Config from module
        click.echo("Configuration (from config.py):")
        for key, value in VALIDATION_CACHE.items():
            if key == "disk_path" and value is None:
                value = "~/.giflab_cache/validation_cache.db (default)"
            click.echo(f"  {key:25s}: {value}")
        
        click.echo()
        
        # Runtime values
        click.echo("Runtime Values:")
        click.echo(f"  {'Enabled':25s}: {validation_cache.enabled}")
        click.echo(f"  {'Memory Limit':25s}: {validation_cache.memory_limit_bytes / (1024 * 1024):.0f} MB")
        click.echo(f"  {'Disk Limit':25s}: {validation_cache.disk_limit_bytes / (1024 * 1024):.0f} MB")
        click.echo(f"  {'TTL':25s}: {validation_cache.ttl_seconds / 3600:.0f} hours")
        click.echo(f"  {'Disk Path':25s}: {validation_cache.disk_path}")
        
        # Check if path exists and get size
        if validation_cache.disk_path.exists():
            size_mb = validation_cache.disk_path.stat().st_size / (1024 * 1024)
            click.echo(f"  {'Disk File Size':25s}: {size_mb:.1f} MB")
        
        click.echo()
        
        # Metric-specific cache settings
        click.echo("Metric-Specific Cache Settings:")
        metric_settings = {
            "SSIM": VALIDATION_CACHE.get("cache_ssim", True),
            "MS-SSIM": VALIDATION_CACHE.get("cache_ms_ssim", True),
            "LPIPS": VALIDATION_CACHE.get("cache_lpips", True),
            "Gradient/Color": VALIDATION_CACHE.get("cache_gradient_color", True),
            "SSIMulacra2": VALIDATION_CACHE.get("cache_ssimulacra2", True),
        }
        
        for metric, enabled in metric_settings.items():
            status = "✅ Enabled" if enabled else "❌ Disabled"
            click.echo(f"  {metric:15s}: {status}")
        
        click.echo()
        click.echo("💡 To modify configuration, edit VALIDATION_CACHE in src/giflab/config.py")
    
    except Exception as e:
        handle_generic_error("Validation cache config", e)