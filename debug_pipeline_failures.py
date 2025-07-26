#!/usr/bin/env python3
"""
Pipeline Failure Diagnostic Script

This script helps systematically debug pipeline failures, particularly
the gifski frame size mismatch issues observed in elimination testing.

Usage:
    python debug_pipeline_failures.py --test-individual-tools
    python debug_pipeline_failures.py --test-problematic-pipeline
    python debug_pipeline_failures.py --test-alternatives
    python debug_pipeline_failures.py --full-analysis
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import click


class PipelineFailureDiagnostic:
    """Systematic pipeline failure analysis and debugging."""
    
    def __init__(self, test_gif: str = "test_elimination/few_colors.gif"):
        self.test_gif = Path(test_gif)
        self.debug_dir = Path("debug_pipeline_failures")
        self.debug_dir.mkdir(exist_ok=True)
        
    def test_individual_tools(self) -> Dict[str, Any]:
        """Test individual tools to isolate where failures occur."""
        results = {}
        
        click.echo("🔧 Testing individual tools...")
        
        # Test FFmpeg
        click.echo("  Testing FFmpeg...")
        try:
            cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", 
                   "-show_entries", "frame=width,height", "-of", "csv=p=0", str(self.test_gif)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                frame_dims = result.stdout.strip().split('\n')
                results['ffmpeg'] = {
                    'status': 'success',
                    'frame_dimensions': frame_dims[:10],  # First 10 frames
                    'unique_dimensions': len(set(frame_dims))
                }
                click.echo(f"    ✅ FFmpeg: {len(frame_dims)} frames, {len(set(frame_dims))} unique dimensions")
            else:
                results['ffmpeg'] = {'status': 'failed', 'error': result.stderr}
                click.echo(f"    ❌ FFmpeg failed: {result.stderr}")
        except Exception as e:
            results['ffmpeg'] = {'status': 'error', 'error': str(e)}
            click.echo(f"    ❌ FFmpeg error: {e}")
        
        # Test Gifski availability
        click.echo("  Testing Gifski...")
        try:
            result = subprocess.run(["gifski", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                results['gifski'] = {'status': 'available', 'version': result.stdout.strip()}
                click.echo(f"    ✅ Gifski available: {result.stdout.strip()}")
            else:
                results['gifski'] = {'status': 'unavailable', 'error': result.stderr}
                click.echo(f"    ❌ Gifski unavailable: {result.stderr}")
        except Exception as e:
            results['gifski'] = {'status': 'error', 'error': str(e)}
            click.echo(f"    ❌ Gifski error: {e}")
        
        # Test Animately
        click.echo("  Testing Animately...")
        try:
            animately_path = Path("bin/darwin/arm64/animately")  # Adjust for your platform
            if animately_path.exists():
                result = subprocess.run([str(animately_path), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                results['animately'] = {
                    'status': 'available', 
                    'path': str(animately_path),
                    'version_check': result.returncode == 0
                }
                click.echo(f"    ✅ Animately found at: {animately_path}")
            else:
                results['animately'] = {'status': 'not_found', 'searched_path': str(animately_path)}
                click.echo(f"    ❌ Animately not found at: {animately_path}")
        except Exception as e:
            results['animately'] = {'status': 'error', 'error': str(e)}
            click.echo(f"    ❌ Animately error: {e}")
        
        return results
    
    def test_problematic_pipeline(self) -> Dict[str, Any]:
        """Test the specific problematic pipeline combination."""
        click.echo("\n🚨 Testing problematic pipeline: animately-frame + ffmpeg-color + gifski-lossy")
        
        try:
            # This would need to be adapted to use the actual GifLab pipeline execution
            from giflab.dynamic_pipeline import generate_all_pipelines
            from giflab.pipeline_elimination import PipelineEliminator
            
            pipelines = generate_all_pipelines()
            problematic = [p for p in pipelines if (
                'animately-frame' in p.identifier() and 
                'ffmpeg-color' in p.identifier() and 
                'gifski-lossy' in p.identifier()
            )]
            
            click.echo(f"  Found {len(problematic)} problematic pipeline combinations")
            
            if problematic:
                # Test first one as example
                test_pipeline = problematic[0]
                click.echo(f"  Testing: {test_pipeline.identifier()}")
                
                # This would execute the pipeline and capture the failure
                return {
                    'pipeline_id': test_pipeline.identifier(),
                    'status': 'test_needed',
                    'note': 'Manual execution required'
                }
        
        except ImportError as e:
            click.echo(f"  ❌ Could not import GifLab modules: {e}")
            return {'status': 'import_error', 'error': str(e)}
        except Exception as e:
            click.echo(f"  ❌ Error testing problematic pipeline: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_alternative_pipelines(self) -> Dict[str, Any]:
        """Test working alternative pipeline combinations."""
        click.echo("\n✅ Testing alternative pipeline combinations...")
        
        alternatives = [
            "animately-frame + ffmpeg-color + gifsicle-lossy",
            "animately-frame + ffmpeg-color + imagemagick-lossy", 
            "animately-frame + animately-color + gifski-lossy",
            "gifsicle-frame + ffmpeg-color + gifski-lossy"
        ]
        
        results = {}
        for alt in alternatives:
            click.echo(f"  Testing: {alt}")
            # This would need actual pipeline execution
            results[alt] = {'status': 'test_needed', 'note': 'Manual execution required'}
        
        return results
    
    def analyze_frame_dimensions(self) -> Dict[str, Any]:
        """Analyze frame dimension consistency issues."""
        click.echo("\n📐 Analyzing frame dimensions...")
        
        try:
            # Get detailed frame information
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", 
                   "-show_frames", str(self.test_gif)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                frames = data.get('frames', [])
                
                dimensions = []
                for i, frame in enumerate(frames):
                    if frame.get('media_type') == 'video':
                        width = frame.get('width')
                        height = frame.get('height')
                        if width and height:
                            dimensions.append((width, height))
                
                unique_dims = set(dimensions)
                analysis = {
                    'total_frames': len(dimensions),
                    'unique_dimensions': len(unique_dims),
                    'dimensions': list(unique_dims),
                    'consistent': len(unique_dims) == 1,
                    'frame_details': dimensions[:10]  # First 10 frames
                }
                
                if analysis['consistent']:
                    click.echo(f"  ✅ Consistent dimensions: {list(unique_dims)[0]}")
                else:
                    click.echo(f"  ❌ Inconsistent dimensions found: {list(unique_dims)}")
                    click.echo(f"     Total variations: {len(unique_dims)}")
                
                return analysis
                
        except Exception as e:
            click.echo(f"  ❌ Error analyzing dimensions: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive diagnostic report."""
        from datetime import datetime
        
        report_lines = [
            "# Pipeline Failure Diagnostic Report",
            f"Test GIF: {self.test_gif}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            ""
        ]
        
        # Tool availability summary
        if 'individual_tools' in results:
            report_lines.extend([
                "### Tool Availability",
                ""
            ])
            tools = results['individual_tools']
            for tool, data in tools.items():
                status = data.get('status', 'unknown')
                report_lines.append(f"- **{tool}**: {status}")
                if status == 'error':
                    report_lines.append(f"  - Error: {data.get('error', 'Unknown')}")
            report_lines.append("")
        
        # Frame dimension analysis
        if 'frame_analysis' in results:
            analysis = results['frame_analysis']
            report_lines.extend([
                "### Frame Dimension Analysis",
                "",
                f"- Total frames: {analysis.get('total_frames', 'Unknown')}",
                f"- Unique dimensions: {analysis.get('unique_dimensions', 'Unknown')}",
                f"- Consistent: {'✅' if analysis.get('consistent') else '❌'}",
                ""
            ])
            if not analysis.get('consistent'):
                dims = analysis.get('dimensions', [])
                report_lines.append("**Dimension variations found:**")
                for dim in dims:
                    report_lines.append(f"- {dim[0]}×{dim[1]}")
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if 'frame_analysis' in results and not results['frame_analysis'].get('consistent'):
            report_lines.extend([
                "**CRITICAL**: Frame dimension inconsistencies detected!",
                "- This likely explains the gifski failures",
                "- gifski requires all frames to have identical dimensions",
                "- Solution: Add frame normalization step or avoid gifski for inconsistent content",
                ""
            ])
        
        report_lines.extend([
            "## Next Steps",
            "",
            "1. Fix frame dimension inconsistencies",
            "2. Test alternative pipeline combinations", 
            "3. Implement pipeline validation",
            "4. Re-run elimination testing",
            ""
        ])
        
        return "\n".join(report_lines)


@click.command()
@click.option('--test-gif', default="test_elimination/few_colors.gif", 
              help="Test GIF to use for diagnostics")
@click.option('--test-individual-tools', is_flag=True, 
              help="Test individual tool availability and functionality")
@click.option('--test-problematic-pipeline', is_flag=True,
              help="Test the specific problematic pipeline combination")
@click.option('--test-alternatives', is_flag=True,
              help="Test alternative working pipeline combinations")
@click.option('--full-analysis', is_flag=True,
              help="Run complete diagnostic analysis")
def main(test_gif: str, test_individual_tools: bool, test_problematic_pipeline: bool,
         test_alternatives: bool, full_analysis: bool):
    """Pipeline failure diagnostic tool."""
    
    diagnostic = PipelineFailureDiagnostic(test_gif)
    results = {}
    
    if full_analysis or test_individual_tools:
        results['individual_tools'] = diagnostic.test_individual_tools()
    
    if full_analysis or test_problematic_pipeline:
        results['problematic_pipeline'] = diagnostic.test_problematic_pipeline()
    
    if full_analysis or test_alternatives:
        results['alternatives'] = diagnostic.test_alternative_pipelines()
    
    if full_analysis:
        results['frame_analysis'] = diagnostic.analyze_frame_dimensions()
    
    # Save results
    results_file = diagnostic.debug_dir / "diagnostic_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    if full_analysis:
        report = diagnostic.generate_report(results)
        report_file = diagnostic.debug_dir / "diagnostic_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        click.echo(f"\n📊 Full diagnostic report saved to: {report_file}")
    
    click.echo(f"\n💾 Raw results saved to: {results_file}")
    
    if not any([test_individual_tools, test_problematic_pipeline, test_alternatives, full_analysis]):
        click.echo("\n❓ No specific tests requested. Use --help to see options or --full-analysis for everything.")


if __name__ == "__main__":
    main() 