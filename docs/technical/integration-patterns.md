# Integration Patterns: Targeted Experiment Presets

This document provides comprehensive integration patterns for incorporating the targeted experiment presets system into various workflows, tools, and development environments.

## Overview

The targeted presets system is designed for seamless integration with existing workflows while providing new capabilities for efficient experiment design. This document covers integration patterns for common scenarios and advanced use cases.

## CLI Integration Patterns

### 1. Command Line Automation

#### Basic Automation Scripts

**Bash Integration**:
```bash
#!/bin/bash
# automated_experiment_suite.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="results_${TIMESTAMP}"

# Define experiment suite
declare -A EXPERIMENTS=(
    ["frame_comparison"]="--preset frame-focus"
    ["color_comparison"]="--preset color-optimization"
    ["lossy_analysis"]="--preset lossy-quality-sweep"
    ["tool_baseline"]="--preset tool-comparison-baseline"
)

# Execute experiments with error handling
for experiment_name in "${!EXPERIMENTS[@]}"; do
    echo "Starting experiment: $experiment_name"
    output_dir="${BASE_OUTPUT_DIR}/${experiment_name}"
    
    if poetry run python -m giflab experiment \
        ${EXPERIMENTS[$experiment_name]} \
        --output-dir "$output_dir" \
        --use-cache \
        --quality-threshold 0.05; then
        echo "✓ Completed: $experiment_name"
    else
        echo "✗ Failed: $experiment_name"
        exit 1
    fi
done

echo "All experiments completed successfully in $BASE_OUTPUT_DIR"
```

**PowerShell Integration**:
```powershell
# automated_experiments.ps1

param(
    [string]$OutputBase = "results_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$QualityThreshold = "0.05"
)

$Experiments = @{
    "frame_study" = "--preset frame-focus"
    "color_study" = "--preset color-optimization"
    "quick_validation" = "--preset quick-test"
}

foreach ($experiment in $Experiments.GetEnumerator()) {
    $experimentName = $experiment.Key
    $experimentArgs = $experiment.Value
    $outputDir = Join-Path $OutputBase $experimentName
    
    Write-Host "Running experiment: $experimentName" -ForegroundColor Green
    
    try {
        & poetry run python -m giflab experiment $experimentArgs.Split(' ') `
            --output-dir $outputDir `
            --quality-threshold $QualityThreshold `
            --use-cache
        Write-Host "✓ Completed: $experimentName" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed: $experimentName - $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
```

#### Advanced CLI Integration

**Parameter Matrix Testing**:
```bash
#!/bin/bash
# parameter_matrix_testing.sh

# Define parameter combinations to test
FRAME_ALGORITHMS=("animately-frame" "ffmpeg-frame" "gifsicle-frame")
COLOR_COUNTS=(64 32 16)
QUALITY_THRESHOLDS=(0.02 0.05 0.1)

for frame_alg in "${FRAME_ALGORITHMS[@]}"; do
    for color_count in "${COLOR_COUNTS[@]}"; do
        for quality in "${QUALITY_THRESHOLDS[@]}"; do
            experiment_name="matrix_${frame_alg}_${color_count}_${quality//./_}"
            output_dir="matrix_results/$experiment_name"
            
            echo "Testing: $frame_alg with $color_count colors at $quality quality"
            
            poetry run python -m giflab experiment \
                --variable-slot "frame=$frame_alg" \
                --lock-slot color=ffmpeg-color \
                --lock-slot lossy=none-lossy \
                --slot-params "color=colors:$color_count" \
                --quality-threshold "$quality" \
                --output-dir "$output_dir" \
                --use-cache
        done
    done
done
```

### 2. Environment-Based Configuration

**Environment Variable Integration**:
```bash
# Configuration via environment variables
export GIFLAB_DEFAULT_PRESET="frame-focus"
export GIFLAB_OUTPUT_BASE="/shared/results"
export GIFLAB_QUALITY_THRESHOLD="0.05"
export GIFLAB_USE_GPU="true"
export GIFLAB_USE_CACHE="true"

# Wrapper script using environment configuration
#!/bin/bash
# run_experiment_with_env.sh

PRESET=${GIFLAB_DEFAULT_PRESET:-"quick-test"}
OUTPUT_DIR=${GIFLAB_OUTPUT_BASE:-"./results"}/$(date +%Y%m%d_%H%M%S)
QUALITY=${GIFLAB_QUALITY_THRESHOLD:-"0.05"}

ARGS="--preset $PRESET --output-dir $OUTPUT_DIR --quality-threshold $QUALITY"

if [ "$GIFLAB_USE_GPU" = "true" ]; then
    ARGS="$ARGS --use-gpu"
fi

if [ "$GIFLAB_USE_CACHE" = "true" ]; then
    ARGS="$ARGS --use-cache"
fi

echo "Running experiment with: $ARGS"
poetry run python -m giflab experiment $ARGS
```

## Python API Integration Patterns

### 1. Programmatic Integration

#### Basic API Usage

**Simple Preset Execution**:
```python
from giflab.experimental.runner import ExperimentalRunner
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_preset_study(preset_id: str, output_dir: str, **kwargs) -> dict:
    """Execute preset study with error handling and result summary."""
    try:
        # Initialize runner
        runner = ExperimentalRunner(
            output_dir=Path(output_dir),
            use_cache=kwargs.get('use_cache', True)
        )
        
        # Execute experiment
        result = runner.run_targeted_experiment(
            preset_id=preset_id,
            quality_threshold=kwargs.get('quality_threshold', 0.05),
            use_targeted_gifs=kwargs.get('use_targeted_gifs', False)
        )
        
        # Return summary
        return {
            'preset_id': preset_id,
            'success': True,
            'jobs_run': result.total_jobs_run,
            'output_dir': output_dir,
            'eliminated_pipelines': len(result.eliminated_pipelines),
            'retained_pipelines': len(result.retained_pipelines)
        }
        
    except Exception as e:
        logging.error(f"Experiment failed for {preset_id}: {e}")
        return {
            'preset_id': preset_id,
            'success': False,
            'error': str(e)
        }

# Example usage
results = []
presets = ['frame-focus', 'color-optimization', 'quick-test']

for preset in presets:
    result = run_preset_study(
        preset_id=preset,
        output_dir=f'results/{preset}',
        quality_threshold=0.1,
        use_cache=True
    )
    results.append(result)
    print(f"Preset {preset}: {'✓' if result['success'] else '✗'}")
```

#### Advanced API Integration

**Custom Workflow Integration**:
```python
from giflab.experimental.targeted_presets import ExperimentPreset, SlotConfiguration, PRESET_REGISTRY
from giflab.experimental.targeted_generator import TargetedPipelineGenerator
from giflab.experimental.runner import ExperimentalRunner
import json
from pathlib import Path
from typing import List, Dict, Any

class ExperimentWorkflow:
    """Advanced workflow for experiment management."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.generator = TargetedPipelineGenerator()
        self.results_history = []
    
    def create_custom_preset(self, name: str, config: Dict[str, Any]) -> str:
        """Create and register custom preset from configuration."""
        preset = ExperimentPreset(
            name=name,
            description=config.get('description', f'Custom preset: {name}'),
            frame_slot=SlotConfiguration(**config['frame_slot']),
            color_slot=SlotConfiguration(**config['color_slot']),
            lossy_slot=SlotConfiguration(**config['lossy_slot']),
            max_combinations=config.get('max_combinations'),
            tags=config.get('tags', [])
        )
        
        preset_id = f"custom-{name.lower().replace(' ', '-')}"
        PRESET_REGISTRY.register(preset_id, preset)
        return preset_id
    
    def validate_preset_efficiency(self, preset_id: str) -> Dict[str, Any]:
        """Validate preset efficiency before execution."""
        preset = PRESET_REGISTRY.get(preset_id)
        validation = self.generator.validate_preset_feasibility(preset)
        
        return {
            'valid': validation['valid'],
            'errors': validation.get('errors', []),
            'estimated_pipelines': validation.get('estimated_pipelines', 0),
            'efficiency_gain': validation.get('efficiency_gain', 0.0),
            'recommendation': self._get_efficiency_recommendation(validation)
        }
    
    def _get_efficiency_recommendation(self, validation: Dict[str, Any]) -> str:
        """Provide efficiency recommendations based on validation."""
        pipeline_count = validation.get('estimated_pipelines', 0)
        efficiency = validation.get('efficiency_gain', 0.0)
        
        if pipeline_count > 200:
            return "Consider using max_combinations to limit pipeline count"
        elif efficiency < 0.8:
            return "Low efficiency gain - consider more focused preset"
        elif pipeline_count < 5:
            return "Very focused preset - good for quick validation"
        else:
            return "Well-balanced preset configuration"
    
    def execute_experiment_batch(self, batch_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute batch of experiments with comprehensive reporting."""
        results = []
        
        for experiment in batch_config['experiments']:
            preset_id = experiment['preset_id']
            
            # Validate efficiency first
            validation = self.validate_preset_efficiency(preset_id)
            if not validation['valid']:
                results.append({
                    'preset_id': preset_id,
                    'status': 'validation_failed',
                    'errors': validation['errors']
                })
                continue
            
            # Execute experiment
            runner = ExperimentalRunner(
                output_dir=self.base_output_dir / preset_id,
                use_cache=batch_config.get('use_cache', True)
            )
            
            try:
                result = runner.run_targeted_experiment(
                    preset_id=preset_id,
                    **experiment.get('parameters', {})
                )
                
                experiment_result = {
                    'preset_id': preset_id,
                    'status': 'completed',
                    'validation': validation,
                    'execution': {
                        'jobs_run': result.total_jobs_run,
                        'eliminated_pipelines': len(result.eliminated_pipelines),
                        'retained_pipelines': len(result.retained_pipelines)
                    }
                }
                
            except Exception as e:
                experiment_result = {
                    'preset_id': preset_id,
                    'status': 'execution_failed',
                    'error': str(e)
                }
            
            results.append(experiment_result)
            self.results_history.append(experiment_result)
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive experiment report."""
        report_lines = [
            "# Experiment Batch Report\n",
            f"Executed {len(results)} experiments\n"
        ]
        
        successful = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] != 'completed']
        
        report_lines.append(f"- Successful: {len(successful)}")
        report_lines.append(f"- Failed: {len(failed)}\n")
        
        if successful:
            report_lines.append("## Successful Experiments\n")
            for result in successful:
                validation = result['validation']
                execution = result['execution']
                report_lines.append(
                    f"**{result['preset_id']}**: "
                    f"{execution['jobs_run']} jobs, "
                    f"{validation['estimated_pipelines']} pipelines, "
                    f"{validation['efficiency_gain']:.1%} efficiency gain\n"
                )
        
        if failed:
            report_lines.append("## Failed Experiments\n")
            for result in failed:
                report_lines.append(f"**{result['preset_id']}**: {result.get('error', 'Unknown error')}\n")
        
        return '\n'.join(report_lines)

# Example usage
workflow = ExperimentWorkflow('batch_results')

# Create custom preset
custom_config = {
    'description': 'Custom frame algorithm comparison',
    'frame_slot': {'type': 'variable', 'scope': ['animately-frame', 'ffmpeg-frame']},
    'color_slot': {'type': 'locked', 'implementation': 'ffmpeg-color', 'parameters': {'colors': 32}},
    'lossy_slot': {'type': 'locked', 'implementation': 'none-lossy'},
    'max_combinations': 50
}

custom_preset_id = workflow.create_custom_preset('Frame Algorithm Study', custom_config)

# Execute batch
batch_config = {
    'experiments': [
        {'preset_id': 'frame-focus'},
        {'preset_id': 'color-optimization'}, 
        {'preset_id': custom_preset_id},
        {'preset_id': 'quick-test', 'parameters': {'quality_threshold': 0.1}}
    ],
    'use_cache': True
}

results = workflow.execute_experiment_batch(batch_config)
report = workflow.generate_report(results)
print(report)
```

### 2. Framework Integration

#### Flask Web Application Integration

**Web API for Preset Management**:
```python
from flask import Flask, jsonify, request
from giflab.experimental.targeted_presets import PRESET_REGISTRY
from giflab.experimental.runner import ExperimentalRunner
from pathlib import Path
import threading
import uuid

app = Flask(__name__)

# In-memory job tracking
active_jobs = {}

@app.route('/api/presets', methods=['GET'])
def list_presets():
    """List all available presets."""
    presets = PRESET_REGISTRY.list_presets()
    return jsonify({
        'presets': [
            {
                'id': preset_id,
                'name': PRESET_REGISTRY.get(preset_id).name,
                'description': description
            }
            for preset_id, description in presets.items()
        ]
    })

@app.route('/api/presets/<preset_id>/validate', methods=['GET'])
def validate_preset(preset_id: str):
    """Validate preset efficiency and feasibility."""
    try:
        from giflab.experimental.targeted_generator import TargetedPipelineGenerator
        
        preset = PRESET_REGISTRY.get(preset_id)
        generator = TargetedPipelineGenerator()
        validation = generator.validate_preset_feasibility(preset)
        
        return jsonify({
            'preset_id': preset_id,
            'valid': validation['valid'],
            'errors': validation.get('errors', []),
            'estimated_pipelines': validation.get('estimated_pipelines', 0),
            'efficiency_gain': validation.get('efficiency_gain', 0.0)
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/experiments', methods=['POST'])
def start_experiment():
    """Start experiment execution asynchronously."""
    data = request.get_json()
    preset_id = data.get('preset_id')
    quality_threshold = data.get('quality_threshold', 0.05)
    use_cache = data.get('use_cache', True)
    
    if not preset_id:
        return jsonify({'error': 'preset_id required'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    def run_experiment():
        """Execute experiment in background thread."""
        try:
            runner = ExperimentalRunner(
                output_dir=Path(f'web_experiments/{job_id}'),
                use_cache=use_cache
            )
            
            active_jobs[job_id]['status'] = 'running'
            result = runner.run_targeted_experiment(
                preset_id=preset_id,
                quality_threshold=quality_threshold
            )
            
            active_jobs[job_id].update({
                'status': 'completed',
                'result': {
                    'jobs_run': result.total_jobs_run,
                    'eliminated_pipelines': len(result.eliminated_pipelines),
                    'retained_pipelines': len(result.retained_pipelines)
                }
            })
            
        except Exception as e:
            active_jobs[job_id].update({
                'status': 'failed',
                'error': str(e)
            })
    
    # Initialize job tracking
    active_jobs[job_id] = {
        'status': 'queued',
        'preset_id': preset_id,
        'quality_threshold': quality_threshold
    }
    
    # Start background execution
    thread = threading.Thread(target=run_experiment)
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'preset_id': preset_id
    })

@app.route('/api/experiments/<job_id>', methods=['GET'])
def get_experiment_status(job_id: str):
    """Get experiment execution status."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(active_jobs[job_id])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

#### FastAPI Integration with Async Support

**Async API with Background Tasks**:
```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from giflab.experimental.targeted_presets import PRESET_REGISTRY
from giflab.experimental.runner import ExperimentalRunner
from pathlib import Path
import asyncio
import uuid

app = FastAPI(title="GifLab Targeted Experiments API")

# Job tracking
job_status: Dict[str, Dict[str, Any]] = {}

class ExperimentRequest(BaseModel):
    preset_id: str
    quality_threshold: Optional[float] = 0.05
    use_cache: Optional[bool] = True
    use_targeted_gifs: Optional[bool] = False

class PresetInfo(BaseModel):
    id: str
    name: str
    description: str
    estimated_pipelines: int
    efficiency_gain: float

@app.get("/presets", response_model=List[PresetInfo])
async def list_presets():
    """List all available presets with efficiency information."""
    from giflab.experimental.targeted_generator import TargetedPipelineGenerator
    
    presets = []
    generator = TargetedPipelineGenerator()
    
    for preset_id, description in PRESET_REGISTRY.list_presets().items():
        preset = PRESET_REGISTRY.get(preset_id)
        validation = generator.validate_preset_feasibility(preset)
        
        presets.append(PresetInfo(
            id=preset_id,
            name=preset.name,
            description=description,
            estimated_pipelines=validation.get('estimated_pipelines', 0),
            efficiency_gain=validation.get('efficiency_gain', 0.0)
        ))
    
    return presets

def execute_experiment(job_id: str, request: ExperimentRequest):
    """Execute experiment in background."""
    try:
        job_status[job_id]['status'] = 'running'
        
        runner = ExperimentalRunner(
            output_dir=Path(f'api_experiments/{job_id}'),
            use_cache=request.use_cache
        )
        
        result = runner.run_targeted_experiment(
            preset_id=request.preset_id,
            quality_threshold=request.quality_threshold,
            use_targeted_gifs=request.use_targeted_gifs
        )
        
        job_status[job_id].update({
            'status': 'completed',
            'result': {
                'jobs_run': result.total_jobs_run,
                'eliminated_pipelines': len(result.eliminated_pipelines),
                'retained_pipelines': len(result.retained_pipelines),
                'output_dir': f'api_experiments/{job_id}'
            }
        })
        
    except Exception as e:
        job_status[job_id].update({
            'status': 'failed',
            'error': str(e)
        })

@app.post("/experiments")
async def start_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """Start experiment execution."""
    job_id = str(uuid.uuid4())
    
    # Validate preset exists
    try:
        PRESET_REGISTRY.get(request.preset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Preset not found: {request.preset_id}")
    
    # Initialize job tracking
    job_status[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'preset_id': request.preset_id,
        'quality_threshold': request.quality_threshold
    }
    
    # Add to background tasks
    background_tasks.add_task(execute_experiment, job_id, request)
    
    return {
        'job_id': job_id,
        'status': 'queued',
        'preset_id': request.preset_id
    }

@app.get("/experiments/{job_id}")
async def get_experiment_status(job_id: str):
    """Get experiment status and results."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

# Example startup event
@app.on_event("startup")
async def startup_event():
    print(f"Available presets: {list(PRESET_REGISTRY.list_presets().keys())}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## CI/CD Integration Patterns

### 1. GitHub Actions Integration

**Automated Experiment Testing**:
```yaml
# .github/workflows/experiment-validation.yml
name: Experiment Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate-presets:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        virtualenvs-create: true
        virtualenvs-in-project: true
    
    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}
    
    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --no-interaction --no-root
    
    - name: Install project
      run: poetry install --no-interaction
    
    - name: Validate preset system
      run: |
        poetry run python -c "
        from giflab.experimental.targeted_presets import PRESET_REGISTRY
        from giflab.experimental.targeted_generator import TargetedPipelineGenerator
        
        print(f'Available presets: {len(PRESET_REGISTRY.list_presets())}')
        generator = TargetedPipelineGenerator()
        
        failed_presets = []
        for preset_id in PRESET_REGISTRY.list_presets().keys():
            preset = PRESET_REGISTRY.get(preset_id)
            validation = generator.validate_preset_feasibility(preset)
            if not validation['valid']:
                failed_presets.append(f'{preset_id}: {validation[\"errors\"]}')
        
        if failed_presets:
            print('Failed preset validations:')
            for failure in failed_presets:
                print(f'  {failure}')
            exit(1)
        else:
            print('All presets validated successfully')
        "
    
    - name: Run quick validation experiments
      run: |
        poetry run python -m giflab experiment --preset quick-test --output-dir ci_test_1
        poetry run python -m giflab experiment --preset frame-focus --output-dir ci_test_2 --use-targeted-gifs
    
    - name: Archive experiment results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: validation-results
        path: |
          ci_test_1/
          ci_test_2/

  benchmark-performance:
    runs-on: ubuntu-latest
    needs: validate-presets
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install
    
    - name: Benchmark preset performance
      run: |
        poetry run python -c "
        import time
        from giflab.experimental.runner import ExperimentalRunner
        from giflab.dynamic_pipeline import generate_all_pipelines
        from pathlib import Path
        
        # Benchmark traditional vs targeted
        print('Benchmarking traditional vs targeted approaches...')
        
        # Traditional approach
        start = time.time()
        all_pipelines = generate_all_pipelines()
        runner = ExperimentalRunner(use_cache=False)
        sampled = runner.select_pipelines_intelligently(all_pipelines, 'quick')
        traditional_time = time.time() - start
        
        # Targeted approach
        start = time.time()
        targeted = runner.generate_targeted_pipelines('frame-focus')
        targeted_time = time.time() - start
        
        # Results
        print(f'Traditional: {len(sampled)} pipelines in {traditional_time:.3f}s')
        print(f'Targeted: {len(targeted)} pipelines in {targeted_time:.3f}s')
        print(f'Efficiency improvement: {traditional_time/targeted_time:.1f}x faster')
        print(f'Pipeline reduction: {1 - len(targeted)/len(all_pipelines):.1%}')
        
        # Validate performance meets expectations
        assert traditional_time > targeted_time, 'Targeted should be faster'
        assert len(targeted) < len(all_pipelines) * 0.1, 'Should use <10% of pipelines'
        print('✓ Performance benchmarks passed')
        "
```

### 2. Jenkins Pipeline Integration

**Jenkins Declarative Pipeline**:
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        POETRY_HOME = "${WORKSPACE}/.poetry"
        PATH = "${POETRY_HOME}/bin:${PATH}"
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    // Install Poetry if not present
                    sh '''
                        if [ ! -f "$POETRY_HOME/bin/poetry" ]; then
                            curl -sSL https://install.python-poetry.org | python3 -
                        fi
                        poetry --version
                    '''
                }
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh '''
                    poetry install --no-dev
                '''
            }
        }
        
        stage('Validate Presets') {
            steps {
                script {
                    def presetValidation = sh(
                        script: '''
                            poetry run python -c "
                            from giflab.experimental.targeted_presets import PRESET_REGISTRY
                            from giflab.experimental.targeted_generator import TargetedPipelineGenerator
                            
                            generator = TargetedPipelineGenerator()
                            failed = []
                            
                            for preset_id in PRESET_REGISTRY.list_presets().keys():
                                preset = PRESET_REGISTRY.get(preset_id)
                                validation = generator.validate_preset_feasibility(preset)
                                if not validation['valid']:
                                    failed.append(preset_id)
                            
                            if failed:
                                print(f'Failed presets: {failed}')
                                exit(1)
                            print('All presets valid')
                            "
                        ''',
                        returnStatus: true
                    )
                    
                    if (presetValidation != 0) {
                        error "Preset validation failed"
                    }
                }
            }
        }
        
        stage('Run Experiment Suite') {
            parallel {
                stage('Quick Tests') {
                    steps {
                        sh '''
                            mkdir -p jenkins_results/quick_tests
                            poetry run python -m giflab experiment \
                                --preset quick-test \
                                --output-dir jenkins_results/quick_tests \
                                --use-cache \
                                --use-targeted-gifs
                        '''
                    }
                }
                
                stage('Frame Analysis') {
                    steps {
                        sh '''
                            mkdir -p jenkins_results/frame_analysis
                            poetry run python -m giflab experiment \
                                --preset frame-focus \
                                --output-dir jenkins_results/frame_analysis \
                                --use-cache \
                                --use-targeted-gifs
                        '''
                    }
                }
                
                stage('Color Analysis') {
                    steps {
                        sh '''
                            mkdir -p jenkins_results/color_analysis
                            poetry run python -m giflab experiment \
                                --preset color-optimization \
                                --output-dir jenkins_results/color_analysis \
                                --use-cache \
                                --use-targeted-gifs
                        '''
                    }
                }
            }
        }
        
        stage('Performance Report') {
            steps {
                script {
                    sh '''
                        poetry run python -c "
                        import json
                        from pathlib import Path
                        from giflab.experimental.targeted_presets import PRESET_REGISTRY
                        from giflab.experimental.targeted_generator import TargetedPipelineGenerator
                        
                        generator = TargetedPipelineGenerator()
                        report = {'presets': []}
                        
                        for preset_id in ['quick-test', 'frame-focus', 'color-optimization']:
                            preset = PRESET_REGISTRY.get(preset_id)
                            validation = generator.validate_preset_feasibility(preset)
                            
                            report['presets'].append({
                                'id': preset_id,
                                'name': preset.name,
                                'estimated_pipelines': validation['estimated_pipelines'],
                                'efficiency_gain': validation['efficiency_gain']
                            })
                        
                        with open('jenkins_results/performance_report.json', 'w') as f:
                            json.dump(report, f, indent=2)
                        
                        print('Performance report generated')
                        "
                    '''
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'jenkins_results/**/*', allowEmptyArchive: true
            
            script {
                if (fileExists('jenkins_results/performance_report.json')) {
                    def report = readJSON file: 'jenkins_results/performance_report.json'
                    
                    echo "Performance Report:"
                    report.presets.each { preset ->
                        echo "  ${preset.name}: ${preset.estimated_pipelines} pipelines (${(preset.efficiency_gain * 100).round(1)}% efficiency)"
                    }
                }
            }
        }
        
        failure {
            emailext (
                subject: "GifLab Experiment Pipeline Failed: ${env.BUILD_NUMBER}",
                body: "The GifLab experiment validation pipeline has failed. Check the logs for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## Docker Integration Patterns

### 1. Containerized Experiments

**Docker Container for Preset Execution**:
```dockerfile
# Dockerfile.experiments
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    gifsicle \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY src/ ./src/
COPY tests/ ./tests/

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Create output directory
RUN mkdir -p /app/results

# Set environment variables
ENV GIFLAB_OUTPUT_DIR=/app/results
ENV GIFLAB_USE_CACHE=true

# Entry point script
COPY docker/experiment_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--preset", "quick-test"]
```

**Docker Entry Point Script**:
```bash
#!/bin/bash
# docker/experiment_entrypoint.sh

set -e

# Default values
OUTPUT_DIR=${GIFLAB_OUTPUT_DIR:-"/app/results"}
PRESET=${GIFLAB_DEFAULT_PRESET:-"quick-test"}
QUALITY_THRESHOLD=${GIFLAB_QUALITY_THRESHOLD:-"0.05"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset|-p)
            PRESET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quality-threshold)
            QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        --list-presets)
            echo "Available presets:"
            python -m giflab experiment --list-presets
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Running experiment with preset: $PRESET"
echo "Output directory: $OUTPUT_DIR"
echo "Quality threshold: $QUALITY_THRESHOLD"

# Execute experiment
python -m giflab experiment \
    --preset "$PRESET" \
    --output-dir "$OUTPUT_DIR" \
    --quality-threshold "$QUALITY_THRESHOLD" \
    --use-cache

echo "Experiment completed successfully"
ls -la "$OUTPUT_DIR"
```

**Docker Compose for Multiple Experiments**:
```yaml
# docker-compose.experiments.yml
version: '3.8'

services:
  frame-study:
    build:
      context: .
      dockerfile: Dockerfile.experiments
    command: ["--preset", "frame-focus", "--quality-threshold", "0.05"]
    volumes:
      - ./results/frame-study:/app/results
      - ./test_data:/app/test_data:ro
    environment:
      - GIFLAB_USE_CACHE=true
    
  color-study:
    build:
      context: .
      dockerfile: Dockerfile.experiments
    command: ["--preset", "color-optimization", "--quality-threshold", "0.05"]
    volumes:
      - ./results/color-study:/app/results
      - ./test_data:/app/test_data:ro
    environment:
      - GIFLAB_USE_CACHE=true
    depends_on:
      - frame-study
  
  baseline-study:
    build:
      context: .
      dockerfile: Dockerfile.experiments
    command: ["--preset", "tool-comparison-baseline", "--quality-threshold", "0.1"]
    volumes:
      - ./results/baseline-study:/app/results
      - ./test_data:/app/test_data:ro
    environment:
      - GIFLAB_USE_CACHE=true
    depends_on:
      - color-study
    
  results-aggregator:
    image: python:3.9-slim
    command: |
      bash -c "
        echo 'Aggregating experiment results...'
        find /results -name '*.json' -exec cat {} \;
        echo 'Results aggregation completed'
      "
    volumes:
      - ./results:/results:ro
    depends_on:
      - frame-study
      - color-study
      - baseline-study
```

### 2. Kubernetes Integration

**Kubernetes Job for Preset Execution**:
```yaml
# k8s/experiment-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: giflab-experiment
  labels:
    app: giflab
    experiment-type: targeted-preset
spec:
  template:
    metadata:
      labels:
        app: giflab
        experiment-type: targeted-preset
    spec:
      restartPolicy: Never
      containers:
      - name: experiment
        image: giflab/experiments:latest
        command: ["python", "-m", "giflab", "experiment"]
        args: 
        - "--preset"
        - "$(PRESET_ID)"
        - "--output-dir"
        - "/shared/results"
        - "--quality-threshold"
        - "$(QUALITY_THRESHOLD)"
        - "--use-cache"
        env:
        - name: PRESET_ID
          valueFrom:
            configMapKeyRef:
              name: experiment-config
              key: preset_id
        - name: QUALITY_THRESHOLD
          valueFrom:
            configMapKeyRef:
              name: experiment-config
              key: quality_threshold
        volumeMounts:
        - name: results-storage
          mountPath: /shared/results
        - name: cache-storage
          mountPath: /root/.cache
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: results-storage
        persistentVolumeClaim:
          claimName: experiment-results-pvc
      - name: cache-storage
        persistentVolumeClaim:
          claimName: experiment-cache-pvc

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: experiment-config
data:
  preset_id: "frame-focus"
  quality_threshold: "0.05"

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: experiment-results-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: experiment-cache-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
```

## Database Integration Patterns

### 1. Experiment Result Storage

**SQLite Integration for Result Tracking**:
```python
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from giflab.experimental.runner import ExperimentalRunner
from giflab.experimental.targeted_presets import PRESET_REGISTRY

class ExperimentDatabase:
    """Database integration for experiment result tracking."""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preset_id TEXT NOT NULL,
                    preset_name TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status TEXT NOT NULL,
                    quality_threshold REAL,
                    jobs_run INTEGER,
                    pipelines_generated INTEGER,
                    eliminated_pipelines INTEGER,
                    retained_pipelines INTEGER,
                    output_dir TEXT,
                    error_message TEXT,
                    configuration TEXT,
                    results TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_preset_id ON experiments(preset_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_started_at ON experiments(started_at)
            ''')
    
    def start_experiment(self, preset_id: str, quality_threshold: float = 0.05, 
                        output_dir: str = None) -> int:
        """Record experiment start and return experiment ID."""
        preset = PRESET_REGISTRY.get(preset_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO experiments (
                    preset_id, preset_name, started_at, status, 
                    quality_threshold, output_dir, configuration
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                preset_id,
                preset.name,
                datetime.now(),
                'running',
                quality_threshold,
                output_dir,
                json.dumps({
                    'frame_slot': preset.frame_slot.__dict__,
                    'color_slot': preset.color_slot.__dict__,
                    'lossy_slot': preset.lossy_slot.__dict__,
                    'max_combinations': preset.max_combinations,
                    'custom_sampling': preset.custom_sampling
                })
            ))
            return cursor.lastrowid
    
    def complete_experiment(self, experiment_id: int, result, pipelines_generated: int):
        """Record successful experiment completion."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE experiments SET
                    completed_at = ?,
                    status = 'completed',
                    jobs_run = ?,
                    pipelines_generated = ?,
                    eliminated_pipelines = ?,
                    retained_pipelines = ?,
                    results = ?
                WHERE id = ?
            ''', (
                datetime.now(),
                result.total_jobs_run,
                pipelines_generated,
                len(result.eliminated_pipelines),
                len(result.retained_pipelines),
                json.dumps({
                    'eliminated_pipelines': list(result.eliminated_pipelines),
                    'retained_pipelines': list(result.retained_pipelines)
                }),
                experiment_id
            ))
    
    def fail_experiment(self, experiment_id: int, error_message: str):
        """Record experiment failure."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE experiments SET
                    completed_at = ?,
                    status = 'failed',
                    error_message = ?
                WHERE id = ?
            ''', (datetime.now(), error_message, experiment_id))
    
    def get_experiment_history(self, preset_id: str = None, limit: int = 100) -> list:
        """Get experiment history with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            if preset_id:
                cursor = conn.execute('''
                    SELECT * FROM experiments 
                    WHERE preset_id = ? 
                    ORDER BY started_at DESC 
                    LIMIT ?
                ''', (preset_id, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM experiments 
                    ORDER BY started_at DESC 
                    LIMIT ?
                ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_preset_statistics(self) -> dict:
        """Get statistics across all experiments."""
        with sqlite3.connect(self.db_path) as conn:
            # Success rate by preset
            cursor = conn.execute('''
                SELECT 
                    preset_id,
                    preset_name,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                    AVG(CASE WHEN status = 'completed' THEN jobs_run ELSE NULL END) as avg_jobs,
                    AVG(pipelines_generated) as avg_pipelines
                FROM experiments 
                GROUP BY preset_id, preset_name
                ORDER BY total_runs DESC
            ''')
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

# Usage example
def run_tracked_experiment(preset_id: str, output_dir: str, quality_threshold: float = 0.05):
    """Run experiment with database tracking."""
    db = ExperimentDatabase()
    experiment_id = db.start_experiment(preset_id, quality_threshold, output_dir)
    
    try:
        runner = ExperimentalRunner(
            output_dir=Path(output_dir),
            use_cache=True
        )
        
        pipelines = runner.generate_targeted_pipelines(preset_id)
        result = runner.run_experimental_analysis(test_pipelines=pipelines)
        
        db.complete_experiment(experiment_id, result, len(pipelines))
        print(f"✓ Experiment {experiment_id} completed successfully")
        
        return result
        
    except Exception as e:
        db.fail_experiment(experiment_id, str(e))
        print(f"✗ Experiment {experiment_id} failed: {e}")
        raise

# Generate statistics report
def generate_statistics_report():
    """Generate comprehensive statistics report."""
    db = ExperimentDatabase()
    stats = db.get_preset_statistics()
    
    print("Experiment Statistics Report")
    print("=" * 50)
    
    for stat in stats:
        success_rate = (stat['successful_runs'] / stat['total_runs']) * 100 if stat['total_runs'] > 0 else 0
        print(f"""
Preset: {stat['preset_name']} ({stat['preset_id']})
  Total Runs: {stat['total_runs']}
  Success Rate: {success_rate:.1f}%
  Average Jobs: {stat['avg_jobs']:.1f}
  Average Pipelines: {stat['avg_pipelines']:.1f}
        """)

if __name__ == "__main__":
    # Example usage
    run_tracked_experiment("frame-focus", "tracked_results")
    generate_statistics_report()
```

This comprehensive integration patterns guide provides robust foundations for incorporating the targeted experiment presets system into diverse development and deployment environments.