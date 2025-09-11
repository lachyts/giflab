#!/usr/bin/env python3
"""
Generate performance regression reports in various formats.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class PerformanceReportGenerator:
    """Generates performance reports from benchmark results."""
    
    def __init__(self, report_file: Path):
        """Initialize with a report file."""
        with open(report_file, 'r') as f:
            self.report_data = json.load(f)
    
    def generate_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = []
        
        # Summary
        lines.append("### Summary")
        lines.append("")
        lines.append(f"- **Commit:** `{self.report_data.get('git_commit', 'unknown')}`")
        lines.append(f"- **Timestamp:** {self.report_data.get('timestamp', 'unknown')}")
        lines.append(f"- **Scenarios Run:** {self.report_data.get('scenarios_run', 0)}")
        lines.append(f"- **Scenarios Passed:** {self.report_data.get('scenarios_passed', 0)}")
        lines.append("")
        
        # Status
        if self.report_data.get('has_regressions'):
            lines.append("### ⚠️ Performance Regressions Detected")
            lines.append("")
            
            for regression in self.report_data.get('regressions', []):
                scenario = regression['scenario']
                details = regression['details']
                
                lines.append(f"#### Scenario: `{scenario}`")
                lines.append("")
                
                # Critical regressions
                if details['regressions']['critical']:
                    lines.append("**🔴 Critical Regressions:**")
                    for reg in details['regressions']['critical']:
                        lines.append(f"- **{reg['metric']}**: {reg['change_percent']:.1f}% "
                                   f"(baseline: {reg['baseline']:.2f}, current: {reg['current']:.2f})")
                    lines.append("")
                
                # Warning regressions
                if details['regressions']['warning']:
                    lines.append("**🟡 Warning Regressions:**")
                    for reg in details['regressions']['warning']:
                        lines.append(f"- **{reg['metric']}**: {reg['change_percent']:.1f}% "
                                   f"(baseline: {reg['baseline']:.2f}, current: {reg['current']:.2f})")
                    lines.append("")
        else:
            lines.append("### ✅ All Performance Tests Passed")
            lines.append("")
            lines.append("No performance regressions detected.")
            lines.append("")
        
        # Detailed results table
        lines.append("### Detailed Results")
        lines.append("")
        lines.append("| Scenario | Status | Frame Cache Hit | Validation Cache Hit | Total Time (ms) |")
        lines.append("|----------|--------|-----------------|---------------------|-----------------|")
        
        # Parse results
        for result_json in self.report_data.get('results', []):
            result = json.loads(result_json)
            scenario = result['scenario_name']
            metrics = result['metrics']
            
            status = "✅ Pass" if result['success'] else "❌ Fail"
            frame_hit = metrics.get('frame_cache_hit_rate', 0) * 100
            val_hit = metrics.get('validation_cache_hit_rate', 0) * 100
            total_time = metrics.get('total_validation_time_ms', 0)
            
            lines.append(f"| {scenario} | {status} | {frame_hit:.1f}% | {val_hit:.1f}% | {total_time:.2f} |")
        
        lines.append("")
        
        # Recommendations
        if self.report_data.get('has_regressions'):
            lines.append("### Recommendations")
            lines.append("")
            lines.append("1. Review recent changes that might impact caching performance")
            lines.append("2. Check if new code bypasses optimization systems")
            lines.append("3. Verify that configuration changes haven't disabled caching")
            lines.append("4. Run `poetry run python -m giflab.benchmarks.regression_suite baseline` locally to compare")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_html(self) -> str:
        """Generate an HTML report."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Performance Regression Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; 
                    padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 6px; margin: 20px 0; }
        .summary-item { margin: 10px 0; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .warning { color: #ffc107; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #007bff; color: white; padding: 12px; text-align: left; }
        td { padding: 12px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .regression-critical { background: #ffebee; border-left: 4px solid #f44336; 
                              padding: 10px; margin: 10px 0; }
        .regression-warning { background: #fff3e0; border-left: 4px solid #ff9800; 
                             padding: 10px; margin: 10px 0; }
        .metric-change { font-family: monospace; font-size: 14px; }
        .recommendations { background: #e3f2fd; padding: 20px; border-radius: 6px; 
                          margin: 20px 0; }
        .footer { text-align: center; color: #999; margin-top: 40px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Performance Regression Report</h1>
"""
        
        # Summary section
        html += f"""
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-item"><strong>Commit:</strong> <code>{self.report_data.get('git_commit', 'unknown')}</code></div>
            <div class="summary-item"><strong>Timestamp:</strong> {self.report_data.get('timestamp', 'unknown')}</div>
            <div class="summary-item"><strong>Scenarios Run:</strong> {self.report_data.get('scenarios_run', 0)}</div>
            <div class="summary-item"><strong>Scenarios Passed:</strong> {self.report_data.get('scenarios_passed', 0)}</div>
        </div>
"""
        
        # Status section
        if self.report_data.get('has_regressions'):
            html += "<h2>⚠️ Performance Regressions Detected</h2>"
            
            for regression in self.report_data.get('regressions', []):
                scenario = regression['scenario']
                details = regression['details']
                
                html += f"<h3>Scenario: <code>{scenario}</code></h3>"
                
                # Critical regressions
                if details['regressions']['critical']:
                    for reg in details['regressions']['critical']:
                        html += f"""
                        <div class="regression-critical">
                            <strong>🔴 CRITICAL:</strong> {reg['metric']}<br>
                            <span class="metric-change">Change: {reg['change_percent']:.1f}%</span><br>
                            Baseline: {reg['baseline']:.2f} → Current: {reg['current']:.2f}
                        </div>
"""
                
                # Warning regressions
                if details['regressions']['warning']:
                    for reg in details['regressions']['warning']:
                        html += f"""
                        <div class="regression-warning">
                            <strong>🟡 WARNING:</strong> {reg['metric']}<br>
                            <span class="metric-change">Change: {reg['change_percent']:.1f}%</span><br>
                            Baseline: {reg['baseline']:.2f} → Current: {reg['current']:.2f}
                        </div>
"""
        else:
            html += """
            <h2 class="pass">✅ All Performance Tests Passed</h2>
            <p>No performance regressions detected.</p>
"""
        
        # Detailed results table
        html += """
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Status</th>
                    <th>Frame Cache Hit Rate</th>
                    <th>Validation Cache Hit Rate</th>
                    <th>Total Time (ms)</th>
                    <th>Memory Peak (MB)</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add table rows
        for result_json in self.report_data.get('results', []):
            result = json.loads(result_json)
            scenario = result['scenario_name']
            metrics = result['metrics']
            
            status_class = "pass" if result['success'] else "fail"
            status_text = "✅ Pass" if result['success'] else "❌ Fail"
            frame_hit = metrics.get('frame_cache_hit_rate', 0) * 100
            val_hit = metrics.get('validation_cache_hit_rate', 0) * 100
            total_time = metrics.get('total_validation_time_ms', 0)
            memory_peak = metrics.get('memory_usage_peak_mb', 0)
            
            html += f"""
                <tr>
                    <td>{scenario}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{frame_hit:.1f}%</td>
                    <td>{val_hit:.1f}%</td>
                    <td>{total_time:.2f}</td>
                    <td>{memory_peak:.1f}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        # Recommendations
        if self.report_data.get('has_regressions'):
            html += """
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ol>
                <li>Review recent changes that might impact caching performance</li>
                <li>Check if new code bypasses optimization systems</li>
                <li>Verify that configuration changes haven't disabled caching</li>
                <li>Run <code>poetry run python -m giflab.benchmarks.regression_suite baseline</code> locally to compare</li>
            </ol>
        </div>
"""
        
        # Footer
        html += f"""
        <div class="footer">
            Generated at {datetime.now().isoformat()} | GifLab Performance Regression Detection
        </div>
    </div>
</body>
</html>
"""
        
        return html


def main():
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--report-file", type=Path, required=True, help="Input report JSON file")
    parser.add_argument("--output-format", choices=["markdown", "html"], default="markdown")
    
    args = parser.parse_args()
    
    if not args.report_file.exists():
        print(f"Error: Report file {args.report_file} not found")
        return 1
    
    generator = PerformanceReportGenerator(args.report_file)
    
    if args.output_format == "markdown":
        print(generator.generate_markdown())
    else:
        print(generator.generate_html())
    
    return 0


if __name__ == "__main__":
    exit(main())