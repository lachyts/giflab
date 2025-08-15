"""Integration helpers for wrapper validation system.

This module provides utilities to integrate validation into existing wrapper
apply() methods and pipeline execution without disrupting their current functionality.
"""

import logging
from pathlib import Path
from typing import Any, Optional

# Import will be done locally to avoid circular imports
from .core import WrapperOutputValidator
from .pipeline_validation import PipelineStageValidator
from .types import ValidationResult

logger = logging.getLogger(__name__)


def add_validation_to_result(
    input_path: Path,
    output_path: Path,
    wrapper_params: dict[str, Any],
    wrapper_result: dict[str, Any],
    wrapper_type: str,
    config: Any = None
) -> dict[str, Any]:
    """Add validation results to wrapper result metadata.
    
    This function integrates validation into existing wrapper results without
    breaking existing functionality. If validation fails or errors occur,
    the original wrapper result is preserved and validation info is added.
    
    Args:
        input_path: Original input file
        output_path: Wrapper output file
        wrapper_params: Parameters passed to wrapper
        wrapper_result: Result dict from wrapper execution
        wrapper_type: Type of wrapper operation ("color_reduction", "frame_reduction", "lossy_compression")
        config: Validation configuration (uses default if None)
        
    Returns:
        Enhanced wrapper result dict with validation information added
    """
    if config is None:
        # Import here to avoid circular imports
        from ..config import DEFAULT_VALIDATION_CONFIG
        validation_config = DEFAULT_VALIDATION_CONFIG
    else:
        validation_config = config
    
    # Skip validation if disabled
    if not validation_config.ENABLE_WRAPPER_VALIDATION:
        return wrapper_result
    
    # Create enhanced result starting with original result
    enhanced_result = wrapper_result.copy()
    
    try:
        # Run validation
        validator = WrapperOutputValidator(validation_config)
        validations = validator.validate_wrapper_output(
            input_path=input_path,
            output_path=output_path,
            wrapper_params=wrapper_params,
            wrapper_metadata=wrapper_result,
            wrapper_type=wrapper_type
        )
        
        # Add validation results to metadata
        enhanced_result['validations'] = [
            {
                'is_valid': v.is_valid,
                'validation_type': v.validation_type,
                'expected': v.expected,
                'actual': v.actual,
                'error_message': v.error_message,
                'details': v.details
            }
            for v in validations
        ]
        enhanced_result['validation_passed'] = all(v.is_valid for v in validations)
        enhanced_result['validation_count'] = len(validations)
        
        # Log validation failures if configured
        if validation_config.LOG_VALIDATION_FAILURES:
            failed_validations = [v for v in validations if not v.is_valid]
            if failed_validations:
                logger.warning(
                    f"Validation failures for {wrapper_type} on {output_path.name}: "
                    f"{[v.error_message for v in failed_validations]}"
                )
        
        # Fail the entire operation if configured (usually False)
        if validation_config.FAIL_ON_VALIDATION_ERROR and not enhanced_result['validation_passed']:
            raise RuntimeError(f"Wrapper validation failed: {[v.error_message for v in validations if not v.is_valid]}")
            
    except Exception as e:
        # Never let validation break the pipeline
        logger.error(f"Validation integration error for {wrapper_type}: {e}")
        enhanced_result['validation_error'] = str(e)
        enhanced_result['validation_passed'] = None
        enhanced_result['validations'] = []
        enhanced_result['validation_count'] = 0
    
    return enhanced_result


def get_wrapper_type_from_class(wrapper_instance: Any) -> str:
    """Determine wrapper type from wrapper class instance.
    
    Args:
        wrapper_instance: Instance of a wrapper class
        
    Returns:
        String indicating wrapper type for validation
    """
    # Check for VARIABLE attribute first (most reliable)
    if hasattr(wrapper_instance, 'VARIABLE'):
        return str(wrapper_instance.VARIABLE)
    
    # Fallback to class name analysis
    class_name = wrapper_instance.__class__.__name__.lower()
    
    if 'color' in class_name:
        return 'color_reduction'
    elif 'frame' in class_name:
        return 'frame_reduction'
    elif 'lossy' in class_name or 'compressor' in class_name:
        return 'lossy_compression'
    else:
        return 'unknown'


def validate_wrapper_apply_result(
    wrapper_instance: Any,
    input_path: Path,
    output_path: Path,
    params: dict[str, Any],
    result: dict[str, Any]
) -> dict[str, Any]:
    """Convenience function to validate wrapper apply() result.
    
    This function can be easily integrated into existing wrapper apply() methods
    by calling it just before returning the result.
    
    Args:
        wrapper_instance: The wrapper instance (for type detection)
        input_path: Input file path
        output_path: Output file path
        params: Parameters dict passed to apply()
        result: Result dict from wrapper execution
        
    Returns:
        Enhanced result dict with validation info
    """
    wrapper_type = get_wrapper_type_from_class(wrapper_instance)
    
    return add_validation_to_result(
        input_path=input_path,
        output_path=output_path,
        wrapper_params=params or {},
        wrapper_result=result,
        wrapper_type=wrapper_type
    )


def validate_pipeline_execution_result(
    input_path: Path,
    pipeline: Any,  # Pipeline type from dynamic_pipeline
    pipeline_params: dict[str, Any],
    pipeline_result: dict[str, Any],
    stage_outputs: dict[str, Path] | None = None,
    final_output_path: Path | None = None,
    validation_config: Any = None
) -> dict[str, Any]:
    """Validate pipeline execution result and add validation info.
    
    This function validates an entire pipeline execution and adds validation
    information to the pipeline result without breaking existing functionality.
    
    Args:
        input_path: Original input file
        pipeline: Pipeline that was executed
        pipeline_params: Parameters used for pipeline
        pipeline_result: Original result from pipeline execution
        stage_outputs: Optional dict of stage outputs
        final_output_path: Optional final output file path
        validation_config: Optional validation configuration
        
    Returns:
        Enhanced pipeline result with validation information
    """
    if validation_config is None:
        # Import here to avoid circular imports
        from ..config import DEFAULT_VALIDATION_CONFIG
        validation_config = DEFAULT_VALIDATION_CONFIG
    
    # Skip validation if disabled
    if not validation_config.ENABLE_WRAPPER_VALIDATION:
        return pipeline_result
    
    # Create enhanced result starting with original result
    enhanced_result = pipeline_result.copy()
    
    try:
        # Initialize validators
        wrapper_validator = WrapperOutputValidator(validation_config)
        pipeline_validator = PipelineStageValidator(wrapper_validator)
        
        # Extract stage information from pipeline result if not provided
        if stage_outputs is None:
            stage_outputs = _extract_stage_outputs_from_result(pipeline_result)
        
        if final_output_path is None:
            final_output_path = _extract_final_output_from_result(pipeline_result)
        
        if not final_output_path or not final_output_path.exists():
            logger.warning("Cannot validate pipeline - final output path not found")
            enhanced_result.update({
                "pipeline_validations": [],
                "pipeline_validation_passed": None,
                "pipeline_validation_error": "Final output path not available"
            })
            return enhanced_result
        
        # Extract stage metadata from pipeline result
        stage_metadata = _extract_stage_metadata_from_result(pipeline_result, pipeline)
        
        # Run comprehensive pipeline validation
        validations = pipeline_validator.validate_pipeline_execution(
            input_path=input_path,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            stage_outputs=stage_outputs,
            stage_metadata=stage_metadata,
            final_output_path=final_output_path
        )
        
        # Process validation results
        validation_passed = all(v.is_valid for v in validations)
        validation_summary = _create_pipeline_validation_summary(validations, pipeline)
        
        # Add validation information to pipeline result
        enhanced_result.update({
            "pipeline_validations": [_validation_result_to_dict(v) for v in validations],
            "pipeline_validation_passed": validation_passed,
            "pipeline_validation_summary": validation_summary
        })
        
        # Log validation failures if configured
        if validation_config.LOG_VALIDATION_FAILURES:
            failed_validations = [v for v in validations if not v.is_valid]
            if failed_validations:
                logger.warning(
                    f"Pipeline validation failures for {pipeline.identifier()}: "
                    f"{[v.error_message for v in failed_validations]}"
                )
        
    except Exception as e:
        logger.warning(f"Pipeline validation failed for {pipeline.identifier()}: {e}")
        enhanced_result.update({
            "pipeline_validations": [],
            "pipeline_validation_passed": None,
            "pipeline_validation_error": str(e)
        })
    
    return enhanced_result


def create_validation_report(
    validations: list[ValidationResult],
    context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a comprehensive validation report from validation results.
    
    Args:
        validations: List of validation results
        context: Optional context information
        
    Returns:
        Comprehensive validation report
    """
    if not validations:
        return {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "overall_status": "no_validations",
            "validation_types": {},
            "context": context or {}
        }
    
    # Group validations by type
    by_type: dict[str, dict[str, Any]] = {}
    for validation in validations:
        v_type = validation.validation_type
        if v_type not in by_type:
            by_type[v_type] = {"passed": 0, "failed": 0, "results": []}
        
        if validation.is_valid:
            by_type[v_type]["passed"] += 1
        else:
            by_type[v_type]["failed"] += 1
        
        by_type[v_type]["results"].append(_validation_result_to_dict(validation))
    
    # Calculate summary statistics
    total_validations = len(validations)
    passed_validations = sum(1 for v in validations if v.is_valid)
    failed_validations = total_validations - passed_validations
    
    overall_status = "passed" if failed_validations == 0 else "failed"
    
    return {
        "total_validations": total_validations,
        "passed_validations": passed_validations,
        "failed_validations": failed_validations,
        "overall_status": overall_status,
        "success_rate": passed_validations / total_validations if total_validations > 0 else 0.0,
        "validation_types": by_type,
        "failed_validation_messages": [v.error_message for v in validations if not v.is_valid and v.error_message],
        "context": context or {}
    }


# Helper functions for pipeline validation integration

def _validation_result_to_dict(validation: ValidationResult) -> dict[str, Any]:
    """Convert ValidationResult to dictionary for JSON serialization."""
    return {
        "is_valid": validation.is_valid,
        "validation_type": validation.validation_type,
        "expected": validation.expected,
        "actual": validation.actual,
        "error_message": validation.error_message,
        "details": validation.details or {}
    }


def _create_pipeline_validation_summary(validations: list[ValidationResult], pipeline: Any) -> dict[str, Any]:
    """Create pipeline-specific validation summary."""
    if not validations:
        return {"total": 0, "passed": 0, "failed": 0, "pipeline_info": {}}
    
    total = len(validations)
    passed = sum(1 for v in validations if v.is_valid)
    failed = total - passed
    
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "types_validated": list({v.validation_type for v in validations}),
        "critical_failures": [v.validation_type for v in validations if not v.is_valid],
        "pipeline_info": {
            "pipeline_id": pipeline.identifier(),
            "pipeline_stages": len(pipeline.steps),
            "stage_types": [step.variable for step in pipeline.steps],
            "pipeline_tools": [step.tool_cls.__name__ for step in pipeline.steps]
        }
    }


def _extract_stage_outputs_from_result(pipeline_result: dict[str, Any]) -> dict[str, Path]:
    """Extract stage outputs from pipeline result."""
    stage_outputs = {}
    
    # Look for stage output information in the result
    if "stage_outputs" in pipeline_result:
        for stage_name, output_path in pipeline_result["stage_outputs"].items():
            if isinstance(output_path, str | Path):
                stage_outputs[stage_name] = Path(output_path)
    
    return stage_outputs


def _extract_final_output_from_result(pipeline_result: dict[str, Any]) -> Path | None:
    """Extract final output path from pipeline result."""
    if "output_path" in pipeline_result:
        return Path(pipeline_result["output_path"])
    elif "final_output" in pipeline_result:
        return Path(pipeline_result["final_output"])
    return None


def _extract_stage_metadata_from_result(pipeline_result: dict[str, Any], pipeline: Any) -> dict[str, dict[str, Any]]:
    """Extract stage metadata from pipeline result."""
    stage_metadata = {}
    
    # Try to extract metadata for each stage
    for step in pipeline.steps:
        stage_name = f"{step.variable}_{step.tool_cls.__name__}"
        
        # Look for stage-specific metadata in the result
        if "stage_metadata" in pipeline_result and stage_name in pipeline_result["stage_metadata"]:
            stage_metadata[stage_name] = pipeline_result["stage_metadata"][stage_name]
        else:
            # Use general pipeline metadata as fallback
            stage_metadata[stage_name] = {
                "stage_variable": step.variable,
                "tool_class": step.tool_cls.__name__,
                "pipeline_result": pipeline_result  # Include full context
            }
    
    return stage_metadata
