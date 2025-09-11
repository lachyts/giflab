#!/usr/bin/env python3
"""
Analyze performance trends over time from historical benchmark data.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import statistics
from collections import defaultdict


class PerformanceTrendAnalyzer:
    """Analyzes performance trends from historical benchmark data."""
    
    def __init__(self, history_dir: Path):
        """Initialize with a directory containing historical reports."""
        self.history_dir = history_dir
        self.historical_data = self._load_historical_data()
    
    def _load_historical_data(self) -> List[Dict]:
        """Load all historical performance reports."""
        reports = []
        
        # Find all JSON report files
        for report_file in self.history_dir.glob("**/ci_report_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    # Add file metadata
                    report['_file_path'] = str(report_file)
                    report['_file_date'] = datetime.fromtimestamp(
                        report_file.stat().st_mtime
                    )
                    reports.append(report)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load {report_file}: {e}")
        
        # Sort by timestamp
        reports.sort(key=lambda r: r.get('timestamp', ''))
        return reports
    
    def analyze_metric_trends(
        self, 
        metric_name: str,
        scenario_name: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Analyze trends for a specific metric.
        
        Returns:
            Dict mapping scenario names to list of (timestamp, value) tuples
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        trends = defaultdict(list)
        
        for report in self.historical_data:
            try:
                timestamp = datetime.fromisoformat(report['timestamp'])
                if timestamp < cutoff_date:
                    continue
                
                # Parse results
                for result_json in report.get('results', []):
                    result = json.loads(result_json)
                    
                    if scenario_name and result['scenario_name'] != scenario_name:
                        continue
                    
                    if metric_name in result['metrics']:
                        value = result['metrics'][metric_name]
                        trends[result['scenario_name']].append((timestamp, value))
            except (KeyError, json.JSONDecodeError):
                continue
        
        return dict(trends)
    
    def detect_gradual_degradation(
        self,
        metric_name: str,
        threshold_percent: float = 5.0,
        window_days: int = 7
    ) -> List[Dict]:
        """
        Detect gradual performance degradation over time.
        
        Returns:
            List of detected degradations with details
        """
        degradations = []
        trends = self.analyze_metric_trends(metric_name, days_back=30)
        
        for scenario, data_points in trends.items():
            if len(data_points) < 3:
                continue
            
            # Group by week
            weekly_averages = self._calculate_weekly_averages(data_points)
            
            if len(weekly_averages) < 2:
                continue
            
            # Compare recent week to previous weeks
            recent_avg = weekly_averages[-1][1]
            prev_avg = statistics.mean([avg for _, avg in weekly_averages[:-1]])
            
            if prev_avg > 0:
                change_percent = ((recent_avg - prev_avg) / prev_avg) * 100
                
                # For hit rates, negative change is degradation
                if "hit_rate" in metric_name:
                    change_percent = -change_percent
                
                if change_percent > threshold_percent:
                    degradations.append({
                        "scenario": scenario,
                        "metric": metric_name,
                        "change_percent": change_percent,
                        "recent_average": recent_avg,
                        "previous_average": prev_avg,
                        "data_points": len(data_points)
                    })
        
        return degradations
    
    def _calculate_weekly_averages(
        self, 
        data_points: List[Tuple[datetime, float]]
    ) -> List[Tuple[datetime, float]]:
        """Calculate weekly averages from data points."""
        if not data_points:
            return []
        
        # Group by week
        weekly_data = defaultdict(list)
        for timestamp, value in data_points:
            week_start = timestamp - timedelta(days=timestamp.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            weekly_data[week_start].append(value)
        
        # Calculate averages
        weekly_averages = []
        for week_start, values in sorted(weekly_data.items()):
            avg = statistics.mean(values)
            weekly_averages.append((week_start, avg))
        
        return weekly_averages
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for all metrics."""
        summary = {
            "total_reports": len(self.historical_data),
            "date_range": {
                "start": None,
                "end": None
            },
            "scenarios": {},
            "overall_trends": {}
        }
        
        if self.historical_data:
            summary["date_range"]["start"] = self.historical_data[0].get('timestamp')
            summary["date_range"]["end"] = self.historical_data[-1].get('timestamp')
        
        # Analyze each metric
        key_metrics = [
            "frame_cache_hit_rate",
            "validation_cache_hit_rate",
            "total_validation_time_ms",
            "memory_usage_peak_mb"
        ]
        
        for metric in key_metrics:
            trends = self.analyze_metric_trends(metric, days_back=30)
            
            for scenario, data_points in trends.items():
                if scenario not in summary["scenarios"]:
                    summary["scenarios"][scenario] = {}
                
                if data_points:
                    values = [v for _, v in data_points]
                    summary["scenarios"][scenario][metric] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
        
        # Detect degradations
        summary["degradations"] = []
        for metric in key_metrics:
            degradations = self.detect_gradual_degradation(metric)
            summary["degradations"].extend(degradations)
        
        return summary
    
    def generate_markdown_report(self) -> str:
        """Generate a Markdown trend report."""
        summary = self.generate_summary_statistics()
        lines = []
        
        lines.append("# Performance Trend Analysis")
        lines.append("")
        lines.append(f"**Analysis Period:** {summary['date_range']['start']} to {summary['date_range']['end']}")
        lines.append(f"**Total Reports Analyzed:** {summary['total_reports']}")
        lines.append("")
        
        # Degradation warnings
        if summary['degradations']:
            lines.append("## ⚠️ Performance Degradations Detected")
            lines.append("")
            for deg in summary['degradations']:
                lines.append(f"- **{deg['scenario']}** - {deg['metric']}: "
                           f"{deg['change_percent']:.1f}% degradation")
                lines.append(f"  - Previous average: {deg['previous_average']:.2f}")
                lines.append(f"  - Recent average: {deg['recent_average']:.2f}")
            lines.append("")
        else:
            lines.append("## ✅ No Significant Performance Degradations")
            lines.append("")
        
        # Scenario statistics
        lines.append("## Scenario Performance Statistics (Last 30 Days)")
        lines.append("")
        
        for scenario, metrics in summary['scenarios'].items():
            lines.append(f"### {scenario}")
            lines.append("")
            lines.append("| Metric | Mean | Median | Std Dev | Min | Max | Samples |")
            lines.append("|--------|------|--------|---------|-----|-----|---------|")
            
            for metric_name, stats in metrics.items():
                # Format based on metric type
                if "rate" in metric_name:
                    mean = f"{stats['mean']*100:.1f}%"
                    median = f"{stats['median']*100:.1f}%"
                    min_val = f"{stats['min']*100:.1f}%"
                    max_val = f"{stats['max']*100:.1f}%"
                    stdev = f"{stats['stdev']*100:.1f}%"
                elif "time" in metric_name or "ms" in metric_name:
                    mean = f"{stats['mean']:.1f}ms"
                    median = f"{stats['median']:.1f}ms"
                    min_val = f"{stats['min']:.1f}ms"
                    max_val = f"{stats['max']:.1f}ms"
                    stdev = f"{stats['stdev']:.1f}ms"
                elif "mb" in metric_name.lower():
                    mean = f"{stats['mean']:.1f}MB"
                    median = f"{stats['median']:.1f}MB"
                    min_val = f"{stats['min']:.1f}MB"
                    max_val = f"{stats['max']:.1f}MB"
                    stdev = f"{stats['stdev']:.1f}MB"
                else:
                    mean = f"{stats['mean']:.2f}"
                    median = f"{stats['median']:.2f}"
                    min_val = f"{stats['min']:.2f}"
                    max_val = f"{stats['max']:.2f}"
                    stdev = f"{stats['stdev']:.2f}"
                
                lines.append(f"| {metric_name} | {mean} | {median} | {stdev} | "
                           f"{min_val} | {max_val} | {stats['count']} |")
            lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        if summary['degradations']:
            lines.append("1. **Investigate Recent Changes**: Review commits from the past week")
            lines.append("2. **Check Configuration**: Verify no cache settings were modified")
            lines.append("3. **Profile Hot Paths**: Run detailed profiling on degraded scenarios")
            lines.append("4. **Update Baselines**: Consider if degradation is acceptable and update baselines")
        else:
            lines.append("1. **Continue Monitoring**: Performance remains stable")
            lines.append("2. **Consider Optimization**: Look for opportunities to improve further")
            lines.append("3. **Document Success**: Share performance improvements with team")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_html_report(self) -> str:
        """Generate an HTML trend report with charts."""
        summary = self.generate_summary_statistics()
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Performance Trend Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 40px; 
            background: #f5f5f5; 
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        h1 { 
            color: #333; 
            border-bottom: 2px solid #007bff; 
            padding-bottom: 10px; 
        }
        h2 { 
            color: #555; 
            margin-top: 30px; 
        }
        .summary-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .degradation-warning {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 10px 0;
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .metric-good { color: #28a745; }
        .metric-warning { color: #ffc107; }
        .metric-bad { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Performance Trend Analysis</h1>
"""
        
        # Summary
        html += f"""
        <div class="summary-box">
            <h2>Analysis Summary</h2>
            <p><strong>Period:</strong> {summary['date_range']['start']} to {summary['date_range']['end']}</p>
            <p><strong>Reports Analyzed:</strong> {summary['total_reports']}</p>
            <p><strong>Scenarios Tracked:</strong> {len(summary['scenarios'])}</p>
        </div>
"""
        
        # Degradation warnings
        if summary['degradations']:
            html += "<h2>⚠️ Performance Degradations Detected</h2>"
            for deg in summary['degradations']:
                html += f"""
                <div class="degradation-warning">
                    <strong>{deg['scenario']}</strong> - {deg['metric']}<br>
                    Degradation: <span class="metric-bad">{deg['change_percent']:.1f}%</span><br>
                    Previous avg: {deg['previous_average']:.2f} → Recent avg: {deg['recent_average']:.2f}
                </div>
"""
        
        # Performance charts placeholder
        html += """
        <h2>Performance Trends</h2>
        <div class="chart-container">
            <canvas id="trendChart"></canvas>
        </div>
"""
        
        # Statistics table
        html += "<h2>Detailed Statistics (Last 30 Days)</h2>"
        
        for scenario, metrics in summary['scenarios'].items():
            html += f"<h3>{scenario}</h3>"
            html += """
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Samples</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for metric_name, stats in metrics.items():
                # Format values
                if "rate" in metric_name:
                    mean = f"{stats['mean']*100:.1f}%"
                    median = f"{stats['median']*100:.1f}%"
                    values = [f"{stats['min']*100:.1f}%", f"{stats['max']*100:.1f}%"]
                    stdev = f"{stats['stdev']*100:.1f}%"
                else:
                    mean = f"{stats['mean']:.2f}"
                    median = f"{stats['median']:.2f}"
                    values = [f"{stats['min']:.2f}", f"{stats['max']:.2f}"]
                    stdev = f"{stats['stdev']:.2f}"
                
                html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{mean}</td>
                        <td>{median}</td>
                        <td>{stdev}</td>
                        <td>{values[0]}</td>
                        <td>{values[1]}</td>
                        <td>{stats['count']}</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
"""
        
        # Add chart script (placeholder - would need actual data)
        html += """
        <script>
            // Placeholder for trend chart
            const ctx = document.getElementById('trendChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                    datasets: [{
                        label: 'Cache Hit Rate',
                        data: [85, 87, 86, 88],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        </script>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html


def main():
    parser = argparse.ArgumentParser(description="Analyze performance trends")
    parser.add_argument("--history-dir", type=Path, required=True, 
                       help="Directory containing historical reports")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output file path")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown",
                       help="Output format")
    
    args = parser.parse_args()
    
    if not args.history_dir.exists():
        print(f"Error: History directory {args.history_dir} not found")
        return 1
    
    analyzer = PerformanceTrendAnalyzer(args.history_dir)
    
    if args.format == "markdown":
        report = analyzer.generate_markdown_report()
    else:
        report = analyzer.generate_html_report()
    
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report generated: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())