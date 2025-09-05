"""Pipeline stage validation for multi-stage compression pipelines.

This module provides validation for multi-stage pipelines, ensuring data integrity
and consistency between pipeline stages, particularly for complex pipelines that
involve intermediate PNG sequences or multiple compression stages.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

from ..meta import extract_gif_metadata
from .core import WrapperOutputValidator
from .timing_validation import TimingGridValidator, validate_frame_timing_for_operation
from .types import ValidationResult

logger = logging.getLogger(__name__)


class PipelineStageValidator:
    """Validates pipeline stages and inter-stage data consistency.

    This validator ensures that multi-stage pipelines maintain data integrity
    between stages and that intermediate files (PNG sequences, intermediate GIFs)
    are properly formatted and consistent.
    """

    def __init__(self, wrapper_validator: WrapperOutputValidator | None = None):
        """Initialize pipeline stage validator.

        Args:
            wrapper_validator: Optional wrapper validator instance to reuse configuration
        """
        self.wrapper_validator = wrapper_validator or WrapperOutputValidator()

        # Initialize timing validator for frame timing validation
        self.timing_validator = TimingGridValidator()

        # Pipeline-specific validation thresholds
        self.pipeline_thresholds = {
            # Allow slight variations between stages due to rounding/encoding
            "dimension_tolerance": 0,  # Dimensions must be exact
            "frame_count_tolerance": 1,  # Allow ±1 frame between stages
            "stage_fps_tolerance": 0.15,  # Allow 15% FPS variation between stages
            "png_sequence_frame_tolerance": 1,  # Allow ±1 frame in PNG sequences
            "timing_validation_enabled": True,  # Enable timing validation for frame operations
        }

    def validate_pipeline_execution(
        self,
        input_path: Path,
        pipeline: Any,  # Pipeline type - import avoided for circular import
        pipeline_params: dict[str, Any],
        stage_outputs: dict[str, Path],
        stage_metadata: dict[str, dict[str, Any]],
        final_output_path: Path,
    ) -> list[ValidationResult]:
        """Validate entire pipeline execution including all inter-stage consistency.

        Args:
            input_path: Original input file
            pipeline: Pipeline definition with stages
            pipeline_params: Parameters used for pipeline execution
            stage_outputs: Dict mapping stage names to output file paths
            stage_metadata: Dict mapping stage names to their execution metadata
            final_output_path: Final output file from complete pipeline

        Returns:
            List of ValidationResult objects for all pipeline validations
        """
        validations = []

        # Track progression through pipeline stages
        previous_stage_path = input_path
        extract_gif_metadata(input_path)

        try:
            for i, step in enumerate(pipeline.steps):
                stage_name = f"{step.variable}_{step.tool_cls.__name__}"

                if stage_name not in stage_outputs:
                    validations.append(
                        ValidationResult(
                            is_valid=False,
                            validation_type="pipeline_stage_execution",
                            expected=f"stage_output_{stage_name}",
                            actual="missing_stage_output",
                            error_message=f"Stage {stage_name} output file not found in stage_outputs",
                            details={"stage_index": i, "stage_variable": step.variable},
                        )
                    )
                    continue

                current_stage_path = stage_outputs[stage_name]
                current_stage_metadata = stage_metadata.get(stage_name, {})

                # Validate this individual stage
                stage_validation = self.validate_pipeline_stage(
                    previous_stage_path,
                    current_stage_path,
                    step,
                    pipeline_params,
                    current_stage_metadata,
                    stage_index=i,
                )
                validations.extend(stage_validation)

                # Update for next iteration
                previous_stage_path = current_stage_path
                try:
                    extract_gif_metadata(current_stage_path)
                except OSError as e:
                    logger.error(f"Cannot access stage output file {stage_name}: {e}")
                except Exception as e:
                    logger.error(
                        f"Could not extract metadata from stage {stage_name}: {e}"
                    )

            # Final validation: compare input to final output
            final_validation = self.validate_pipeline_integrity(
                input_path, final_output_path, pipeline, pipeline_params, stage_metadata
            )
            validations.extend(final_validation)

        except Exception as e:
            logger.error(f"Pipeline validation error: {e}")
            validations.append(
                ValidationResult(
                    is_valid=False,
                    validation_type="pipeline_validation_error",
                    expected="successful_pipeline_validation",
                    actual="validation_exception",
                    error_message=f"Pipeline validation failed: {str(e)}",
                    details={"exception": str(e), "pipeline_id": pipeline.identifier()},
                )
            )

        return validations

    def validate_pipeline_stage(
        self,
        input_path: Path,
        output_path: Path,
        pipeline_step: Any,  # PipelineStep type - import avoided for circular import
        pipeline_params: dict[str, Any],
        stage_metadata: dict[str, Any],
        stage_index: int,
    ) -> list[ValidationResult]:
        """Validate a single pipeline stage.

        Args:
            input_path: Input file for this stage
            output_path: Output file from this stage
            pipeline_step: Pipeline step definition
            pipeline_params: Parameters for the entire pipeline
            stage_metadata: Metadata from stage execution
            stage_index: Index of this stage in the pipeline

        Returns:
            List of ValidationResult objects for this stage
        """
        validations = []

        try:
            # Basic file existence and integrity
            if not output_path.exists():
                validations.append(
                    ValidationResult(
                        is_valid=False,
                        validation_type="pipeline_stage_output",
                        expected="output_file_exists",
                        actual="output_file_missing",
                        error_message=f"Stage {pipeline_step.variable} output file not found: {output_path}",
                        details={
                            "stage_index": stage_index,
                            "stage_variable": pipeline_step.variable,
                            "tool_class": pipeline_step.tool_cls.__name__,
                        },
                    )
                )
                return validations

            # Extract metadata from both files
            input_metadata = extract_gif_metadata(input_path)
            output_metadata = extract_gif_metadata(output_path)

            # Stage-specific validations
            if pipeline_step.variable == "frame_reduction":
                frame_validation = self.validate_frame_reduction_stage(
                    input_metadata, output_metadata, pipeline_params, stage_metadata
                )
                validations.extend(frame_validation)

                # Add timing validation for frame reduction operations
                if self.pipeline_thresholds.get("timing_validation_enabled", True):
                    timing_validation = self.timing_validator.validate_timing_integrity(
                        input_path, output_path
                    )
                    # Add operation context
                    if timing_validation.details:
                        timing_validation.details["pipeline_stage"] = "frame_reduction"
                        timing_validation.details["stage_index"] = stage_index
                    validations.append(timing_validation)

            elif pipeline_step.variable == "color_reduction":
                color_validation = self.validate_color_reduction_stage(
                    input_metadata, output_metadata, pipeline_params, stage_metadata
                )
                validations.extend(color_validation)

            elif pipeline_step.variable == "lossy_compression":
                lossy_validation = self.validate_lossy_compression_stage(
                    input_metadata, output_metadata, pipeline_params, stage_metadata
                )
                validations.extend(lossy_validation)

            # Inter-stage consistency validations
            consistency_validation = self.validate_inter_stage_consistency(
                input_path, output_path, input_metadata, output_metadata, pipeline_step
            )
            validations.extend(consistency_validation)

        except Exception as e:
            logger.error(f"Stage validation error for {pipeline_step.variable}: {e}")
            validations.append(
                ValidationResult(
                    is_valid=False,
                    validation_type="pipeline_stage_validation_error",
                    expected="successful_stage_validation",
                    actual="stage_validation_exception",
                    error_message=f"Stage validation failed: {str(e)}",
                    details={
                        "stage_variable": pipeline_step.variable,
                        "stage_index": stage_index,
                        "exception": str(e),
                    },
                )
            )

        return validations

    def validate_frame_reduction_stage(
        self,
        input_metadata: Any,
        output_metadata: Any,
        pipeline_params: dict[str, Any],
        stage_metadata: dict[str, Any],
    ) -> list[ValidationResult]:
        """Validate frame reduction stage in pipeline context."""
        validations = []

        input_frames = input_metadata.orig_frames
        output_frames = output_metadata.orig_frames

        # Check if frame reduction actually occurred when expected
        frame_ratio = pipeline_params.get("frame_ratio", 1.0)
        if frame_ratio < 1.0:
            expected_frames = max(1, int(input_frames * frame_ratio))
            actual_ratio = output_frames / input_frames if input_frames > 0 else 1.0

            frame_diff = abs(output_frames - expected_frames)
            is_valid = frame_diff <= self.pipeline_thresholds["frame_count_tolerance"]

            validations.append(
                ValidationResult(
                    is_valid=is_valid,
                    validation_type="pipeline_frame_reduction",
                    expected={"frames": expected_frames, "ratio": frame_ratio},
                    actual={"frames": output_frames, "ratio": actual_ratio},
                    error_message=None
                    if is_valid
                    else f"Frame reduction in pipeline: expected ~{expected_frames} frames, got {output_frames}",
                    details={
                        "input_frames": input_frames,
                        "frame_difference": frame_diff,
                        "tolerance": self.pipeline_thresholds["frame_count_tolerance"],
                        "timing_validation_recommended": True,  # Flag for comprehensive analysis
                    },
                )
            )

        return validations

    def validate_color_reduction_stage(
        self,
        input_metadata: Any,
        output_metadata: Any,
        pipeline_params: dict[str, Any],
        stage_metadata: dict[str, Any],
    ) -> list[ValidationResult]:
        """Validate color reduction stage in pipeline context."""
        validations = []

        # Color reduction validation is handled by the individual wrapper validator
        # Here we focus on pipeline-specific concerns like ensuring the stage executed

        target_colors = pipeline_params.get("colors")
        if target_colors:
            input_colors = input_metadata.orig_n_colors
            output_colors = output_metadata.orig_n_colors

            # In pipeline context, we expect some color reduction unless input already had fewer colors
            reduction_expected = target_colors < input_colors
            reduction_occurred = output_colors < input_colors

            if reduction_expected and not reduction_occurred:
                validations.append(
                    ValidationResult(
                        is_valid=False,
                        validation_type="pipeline_color_reduction",
                        expected={
                            "colors_reduced": True,
                            "target_colors": target_colors,
                        },
                        actual={
                            "colors_reduced": False,
                            "output_colors": output_colors,
                        },
                        error_message=f"Expected color reduction to {target_colors} colors, but colors remained at {output_colors}",
                        details={
                            "input_colors": input_colors,
                            "target_colors": target_colors,
                            "output_colors": output_colors,
                        },
                    )
                )
            else:
                validations.append(
                    ValidationResult(
                        is_valid=True,
                        validation_type="pipeline_color_reduction",
                        expected={"target_colors": target_colors},
                        actual={"output_colors": output_colors},
                        details={"reduction_occurred": reduction_occurred},
                    )
                )

        return validations

    def validate_lossy_compression_stage(
        self,
        input_metadata: Any,
        output_metadata: Any,
        pipeline_params: dict[str, Any],
        stage_metadata: dict[str, Any],
    ) -> list[ValidationResult]:
        """Validate lossy compression stage in pipeline context."""
        validations = []

        # For lossy compression, main concern is that file integrity is maintained
        # while achieving some compression (file size should typically decrease)

        lossy_level = pipeline_params.get("lossy", 0)
        if lossy_level > 0:
            # Basic validation that lossy compression stage executed
            validations.append(
                ValidationResult(
                    is_valid=True,
                    validation_type="pipeline_lossy_compression",
                    expected={"lossy_applied": True},
                    actual={"lossy_level": lossy_level},
                    details={
                        "lossy_level_requested": lossy_level,
                        "stage_metadata": stage_metadata,
                    },
                )
            )

        return validations

    def validate_inter_stage_consistency(
        self,
        input_path: Path,
        output_path: Path,
        input_metadata: Any,
        output_metadata: Any,
        pipeline_step: Any,  # PipelineStep type - import avoided for circular import
    ) -> list[ValidationResult]:
        """Validate consistency between pipeline stages."""
        validations = []

        # Dimension consistency - dimensions should not change unexpectedly
        input_width, input_height = (
            input_metadata.orig_width,
            input_metadata.orig_height,
        )
        output_width, output_height = (
            output_metadata.orig_width,
            output_metadata.orig_height,
        )

        dimensions_changed = (input_width != output_width) or (
            input_height != output_height
        )

        # Only certain operations should change dimensions
        dimension_change_allowed = pipeline_step.variable in [
            "frame_reduction"
        ]  # Could add more if needed

        if dimensions_changed and not dimension_change_allowed:
            validations.append(
                ValidationResult(
                    is_valid=False,
                    validation_type="pipeline_dimension_consistency",
                    expected={"width": input_width, "height": input_height},
                    actual={"width": output_width, "height": output_height},
                    error_message=f"Unexpected dimension change in {pipeline_step.variable}: {input_width}x{input_height} → {output_width}x{output_height}",
                    details={
                        "stage_variable": pipeline_step.variable,
                        "dimension_change_allowed": dimension_change_allowed,
                    },
                )
            )
        else:
            validations.append(
                ValidationResult(
                    is_valid=True,
                    validation_type="pipeline_dimension_consistency",
                    expected={"consistent_dimensions": True},
                    actual={"width": output_width, "height": output_height},
                    details={"dimensions_changed": dimensions_changed},
                )
            )

        # FPS consistency - FPS should be preserved unless frame reduction occurred
        input_fps = input_metadata.orig_fps
        output_fps = output_metadata.orig_fps

        if input_fps > 0 and output_fps > 0:
            fps_change_percent = abs(output_fps - input_fps) / input_fps
            fps_consistent = (
                fps_change_percent <= self.pipeline_thresholds["stage_fps_tolerance"]
            )

            # Frame reduction might legitimately change FPS, others should preserve it
            fps_change_expected = pipeline_step.variable == "frame_reduction"

            if not fps_consistent and not fps_change_expected:
                validations.append(
                    ValidationResult(
                        is_valid=False,
                        validation_type="pipeline_fps_consistency",
                        expected={"fps": input_fps, "fps_preserved": True},
                        actual={
                            "fps": output_fps,
                            "fps_change_percent": fps_change_percent,
                        },
                        error_message=f"Unexpected FPS change in {pipeline_step.variable}: {input_fps:.2f} → {output_fps:.2f} ({fps_change_percent:.1%})",
                        details={
                            "stage_variable": pipeline_step.variable,
                            "fps_tolerance": self.pipeline_thresholds[
                                "stage_fps_tolerance"
                            ],
                        },
                    )
                )
            else:
                validations.append(
                    ValidationResult(
                        is_valid=True,
                        validation_type="pipeline_fps_consistency",
                        expected={"fps_preserved": True},
                        actual={"fps": output_fps},
                        details={"fps_change_percent": fps_change_percent},
                    )
                )

        return validations

    def validate_pipeline_integrity(
        self,
        original_input: Path,
        final_output: Path,
        pipeline: Any,  # Pipeline type - import avoided for circular import
        pipeline_params: dict[str, Any],
        all_stage_metadata: dict[str, dict[str, Any]],
    ) -> list[ValidationResult]:
        """Validate overall pipeline integrity from input to final output."""
        validations = []

        try:
            # Compare original input to final output for overall pipeline effectiveness
            input_metadata = extract_gif_metadata(original_input)
            final_metadata = extract_gif_metadata(final_output)

            # Overall frame count validation
            expected_final_frames = input_metadata.orig_frames
            frame_ratio = pipeline_params.get("frame_ratio", 1.0)
            if frame_ratio < 1.0:
                expected_final_frames = max(
                    1, int(input_metadata.orig_frames * frame_ratio)
                )

            actual_final_frames = final_metadata.orig_frames
            frame_diff = abs(actual_final_frames - expected_final_frames)
            frames_valid = (
                frame_diff <= self.pipeline_thresholds["frame_count_tolerance"]
            )

            validations.append(
                ValidationResult(
                    is_valid=frames_valid,
                    validation_type="pipeline_overall_integrity",
                    expected={
                        "final_frames": expected_final_frames,
                        "pipeline_stages": len(pipeline.steps),
                    },
                    actual={
                        "final_frames": actual_final_frames,
                        "frame_difference": frame_diff,
                    },
                    error_message=None
                    if frames_valid
                    else f"Pipeline final frame count mismatch: expected {expected_final_frames}, got {actual_final_frames}",
                    details={
                        "original_frames": input_metadata.orig_frames,
                        "frame_ratio_applied": frame_ratio,
                        "pipeline_id": pipeline.identifier(),
                        "total_stages": len(pipeline.steps),
                    },
                )
            )

            # Overall color count validation
            target_colors = pipeline_params.get("colors")
            if target_colors:
                final_colors = final_metadata.orig_n_colors
                color_target_met = (
                    final_colors
                    <= target_colors
                    + self.wrapper_validator.config.COLOR_COUNT_TOLERANCE
                )

                validations.append(
                    ValidationResult(
                        is_valid=color_target_met,
                        validation_type="pipeline_color_integrity",
                        expected={"final_colors": target_colors},
                        actual={"final_colors": final_colors},
                        error_message=None
                        if color_target_met
                        else f"Pipeline color target not met: expected ≤{target_colors}, got {final_colors}",
                        details={
                            "original_colors": input_metadata.orig_n_colors,
                            "color_tolerance": self.wrapper_validator.config.COLOR_COUNT_TOLERANCE,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Pipeline integrity validation error: {e}")
            validations.append(
                ValidationResult(
                    is_valid=False,
                    validation_type="pipeline_integrity_validation_error",
                    expected="successful_integrity_validation",
                    actual="integrity_validation_exception",
                    error_message=f"Pipeline integrity validation failed: {str(e)}",
                    details={"exception": str(e), "pipeline_id": pipeline.identifier()},
                )
            )

        return validations

    def validate_png_sequence_integrity(
        self,
        png_sequence_dir: Path,
        reference_gif: Path,
        expected_frame_count: int | None = None,
    ) -> ValidationResult:
        """Validate PNG sequence integrity for gifski/animately advanced pipelines.

        Args:
            png_sequence_dir: Directory containing PNG sequence
            reference_gif: Reference GIF for comparison
            expected_frame_count: Expected number of PNG frames

        Returns:
            ValidationResult for PNG sequence integrity
        """
        try:
            if not png_sequence_dir.exists():
                return ValidationResult(
                    is_valid=False,
                    validation_type="png_sequence_integrity",
                    expected="png_sequence_directory",
                    actual="directory_missing",
                    error_message=f"PNG sequence directory not found: {png_sequence_dir}",
                )

            # Count PNG files in sequence
            png_files = sorted(png_sequence_dir.glob("*.png"))
            png_count = len(png_files)

            # Get reference frame count
            ref_metadata = extract_gif_metadata(reference_gif)
            ref_frame_count = ref_metadata.orig_frames

            # Use expected count if provided, otherwise use reference
            target_frame_count = expected_frame_count or ref_frame_count

            # Validate frame count
            frame_diff = abs(png_count - target_frame_count)
            frame_count_valid = (
                frame_diff <= self.pipeline_thresholds["png_sequence_frame_tolerance"]
            )

            if not frame_count_valid:
                return ValidationResult(
                    is_valid=False,
                    validation_type="png_sequence_integrity",
                    expected={"png_frames": target_frame_count},
                    actual={"png_frames": png_count},
                    error_message=f"PNG sequence frame count mismatch: expected {target_frame_count}, found {png_count}",
                    details={
                        "frame_difference": frame_diff,
                        "tolerance": self.pipeline_thresholds[
                            "png_sequence_frame_tolerance"
                        ],
                        "png_files_found": len(png_files),
                    },
                )

            # Validate PNG file integrity (sample first few frames)
            sample_frames = min(3, len(png_files))
            dimensions = None

            for i in range(sample_frames):
                png_file = png_files[i]
                try:
                    with Image.open(png_file) as img:
                        if img.format != "PNG":
                            return ValidationResult(
                                is_valid=False,
                                validation_type="png_sequence_integrity",
                                expected="valid_png_format",
                                actual=f"invalid_format_{img.format}",
                                error_message=f"PNG sequence contains non-PNG file: {png_file.name}",
                                details={
                                    "invalid_file": str(png_file),
                                    "detected_format": img.format,
                                },
                            )

                        # Check dimension consistency
                        current_dimensions = img.size
                        if dimensions is None:
                            dimensions = current_dimensions
                        elif current_dimensions != dimensions:
                            return ValidationResult(
                                is_valid=False,
                                validation_type="png_sequence_integrity",
                                expected={"consistent_dimensions": dimensions},
                                actual={"inconsistent_dimensions": current_dimensions},
                                error_message=f"PNG sequence has inconsistent dimensions: {png_file.name} is {current_dimensions}, expected {dimensions}",
                                details={
                                    "first_frame_dimensions": dimensions,
                                    "inconsistent_file": str(png_file),
                                },
                            )

                except Exception as img_error:
                    return ValidationResult(
                        is_valid=False,
                        validation_type="png_sequence_integrity",
                        expected="readable_png_files",
                        actual="corrupted_png_file",
                        error_message=f"Cannot read PNG file in sequence: {png_file.name} - {str(img_error)}",
                        details={
                            "corrupted_file": str(png_file),
                            "error": str(img_error),
                        },
                    )

            # All validations passed
            return ValidationResult(
                is_valid=True,
                validation_type="png_sequence_integrity",
                expected={"png_frames": target_frame_count, "valid_png_files": True},
                actual={"png_frames": png_count, "dimensions": dimensions},
                details={
                    "png_files_validated": sample_frames,
                    "total_png_files": len(png_files),
                    "sequence_directory": str(png_sequence_dir),
                },
            )

        except OSError as e:
            logger.error(f"PNG sequence directory access error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="png_sequence_access_error",
                expected="accessible_png_sequence_directory",
                actual="directory_access_failed",
                error_message=f"Cannot access PNG sequence directory: {str(e)}",
                details={
                    "exception": str(e),
                    "sequence_dir": str(png_sequence_dir),
                    "error_type": "directory_access",
                },
            )
        except Exception as e:
            logger.exception(f"Unexpected PNG sequence validation error: {e}")
            return ValidationResult(
                is_valid=False,
                validation_type="png_sequence_validation_error",
                expected="successful_png_validation",
                actual="validation_exception",
                error_message=f"PNG sequence validation failed with unexpected error: {str(e)}",
                details={
                    "exception": str(e),
                    "sequence_dir": str(png_sequence_dir),
                    "error_type": "unexpected",
                },
            )

    def validate_timing_integrity_for_stage(
        self, input_path: Path, output_path: Path, stage_name: str, stage_index: int = 0
    ) -> ValidationResult:
        """Validate timing integrity for a specific pipeline stage.

        Args:
            input_path: Path to input GIF
            output_path: Path to output GIF
            stage_name: Name of the pipeline stage
            stage_index: Index of the stage in pipeline

        Returns:
            ValidationResult for timing validation
        """
        if not self.pipeline_thresholds.get("timing_validation_enabled", True):
            return ValidationResult(
                is_valid=True,
                validation_type="timing_validation_disabled",
                expected="timing_validation_disabled",
                actual="timing_validation_disabled",
                details={"stage_name": stage_name, "stage_index": stage_index},
            )

        timing_result = self.timing_validator.validate_timing_integrity(
            input_path, output_path
        )

        # Add pipeline context to timing validation
        if timing_result.details:
            timing_result.details.update(
                {
                    "pipeline_stage": stage_name,
                    "stage_index": stage_index,
                    "validation_context": "pipeline_stage_validation",
                }
            )

        return timing_result
