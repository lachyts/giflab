# 🧪 Validation Testing Patterns & Best Practices

This guide provides comprehensive testing patterns, best practices, and examples for the wrapper validation system.

---

## 📋 Testing Philosophy

### Core Testing Principles

1. **Validation Never Breaks Pipelines** - Test that validation failures are informational, not blocking
2. **Test Both Success and Failure Paths** - Validate that validation works correctly in both cases
3. **Use Known Test Fixtures** - Create predictable test scenarios with known properties
4. **Mock External Dependencies** - Isolate validation logic from compression engine behavior
5. **Performance Testing** - Ensure validation overhead remains acceptable

---

## 🏗️ Test Fixture Patterns

### Pattern 1: Known Property Test Fixtures

Create GIFs with precisely known characteristics for predictable testing:

```python
# tests/fixtures/create_validation_fixtures.py
from PIL import Image
import numpy as np
from pathlib import Path

class ValidationTestFixtures:
    """Create test fixtures with precisely known properties."""
    
    @staticmethod
    def create_exact_frame_gif(output_path: Path, frame_count: int, 
                              width: int = 100, height: int = 100, fps: float = 10.0):
        """Create GIF with exact frame count for testing."""
        
        frames = []
        for i in range(frame_count):
            # Create unique frame (different colors)
            color = (i * 50) % 255
            frame = Image.new('RGB', (width, height), color=(color, color, color))
            frames.append(frame)
        
        # Save with precise timing
        duration_ms = int(1000 / fps)  # Convert FPS to duration
        frames[0].save(
            output_path, 
            save_all=True, 
            append_images=frames[1:],
            duration=duration_ms,
            loop=0
        )
        
        return {
            'expected_frames': frame_count,
            'expected_fps': fps,
            'expected_width': width,
            'expected_height': height
        }
    
    @staticmethod
    def create_exact_color_gif(output_path: Path, color_count: int,
                              frame_count: int = 4, width: int = 100, height: int = 100):
        """Create GIF with exact color count for testing."""
        
        # Generate palette with exact number of colors
        colors = []
        for i in range(color_count):
            # Distribute colors evenly across RGB space
            r = (i * 255) // color_count
            g = ((i * 2) * 255) // color_count % 255
            b = ((i * 3) * 255) // color_count % 255
            colors.append((r, g, b))
        
        frames = []
        for frame_idx in range(frame_count):
            frame = Image.new('RGB', (width, height))
            pixels = []
            
            # Fill frame with colors from palette
            for y in range(height):
                for x in range(width):
                    color_idx = (x + y + frame_idx) % color_count
                    pixels.append(colors[color_idx])
            
            frame.putdata(pixels)
            frames.append(frame)
        
        # Convert to palette mode to ensure exact color count
        frames[0].save(output_path, save_all=True, append_images=frames[1:])
        
        return {
            'expected_colors': color_count,
            'expected_frames': frame_count,
            'color_palette': colors
        }
```

### Pattern 2: Edge Case Fixtures

```python
class EdgeCaseFixtures:
    """Create edge case test fixtures."""
    
    @staticmethod
    def create_single_frame_gif(output_path: Path):
        """Single frame GIF for edge case testing."""
        frame = Image.new('RGB', (50, 50), color='red')
        frame.save(output_path)
        
        return {'expected_frames': 1}
    
    @staticmethod
    def create_minimal_gif(output_path: Path):
        """Minimal valid GIF for size testing."""
        # Create 1x1 pixel, 1 frame GIF
        frame = Image.new('RGB', (1, 1), color='white')
        frame.save(output_path, optimize=True)
        
        return {
            'expected_frames': 1,
            'expected_width': 1,
            'expected_height': 1
        }
    
    @staticmethod
    def create_high_fps_gif(output_path: Path, fps: float = 50.0):
        """High FPS GIF for timing edge case testing."""
        frames = []
        for i in range(10):
            color = i * 25
            frame = Image.new('RGB', (50, 50), color=(color, color, color))
            frames.append(frame)
        
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms
        )
        
        return {'expected_fps': fps, 'expected_frames': 10}
```

---

## 🧪 Unit Testing Patterns

### Pattern 1: Validation Result Testing

```python
# tests/test_validation_results.py
import pytest
from pathlib import Path
from giflab.wrapper_validation import WrapperOutputValidator, ValidationConfig

class TestValidationResults:
    """Test validation result accuracy and consistency."""
    
    @pytest.fixture
    def validator(self):
        return WrapperOutputValidator()
    
    @pytest.fixture
    def test_fixtures(self, tmp_path):
        """Create test fixtures for validation testing."""
        fixtures = {}
        
        # Create 10-frame test GIF
        fixtures['10_frames'] = tmp_path / "10_frames.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            fixtures['10_frames'], frame_count=10, fps=15.0
        )
        
        # Create 32-color test GIF
        fixtures['32_colors'] = tmp_path / "32_colors.gif"
        ValidationTestFixtures.create_exact_color_gif(
            fixtures['32_colors'], color_count=32
        )
        
        return fixtures
    
    def test_frame_validation_accuracy(self, validator, test_fixtures, tmp_path):
        """Test frame count validation accuracy."""
        
        input_gif = test_fixtures['10_frames']
        output_gif = tmp_path / "output.gif"
        
        # Copy input to output (no actual compression for this test)
        import shutil
        shutil.copy(input_gif, output_gif)
        
        # Test frame reduction validation
        result = validator.validate_frame_reduction(
            input_path=input_gif,
            output_path=output_gif,
            expected_ratio=1.0,  # No reduction expected
            wrapper_metadata={"engine": "test"}
        )
        
        # Validate result structure
        assert result.is_valid is True
        assert result.validation_type == "frame_count"
        assert result.expected["ratio"] == 1.0
        assert result.actual["frames"] == 10
        assert result.error_message is None
    
    def test_frame_validation_failure(self, validator, test_fixtures, tmp_path):
        """Test frame validation detects failures correctly."""
        
        input_gif = test_fixtures['10_frames']
        
        # Create output with different frame count
        output_gif = tmp_path / "reduced_output.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            output_gif, frame_count=5  # Half the frames
        )
        
        # Test with expectation of no reduction
        result = validator.validate_frame_reduction(
            input_path=input_gif,
            output_path=output_gif,
            expected_ratio=1.0,  # Expect no reduction
            wrapper_metadata={"engine": "test"}
        )
        
        # Should detect the discrepancy
        assert result.is_valid is False
        assert "differs from expected" in result.error_message
        assert result.actual["frames"] == 5
    
    def test_color_validation_accuracy(self, validator, test_fixtures, tmp_path):
        """Test color count validation accuracy."""
        
        input_gif = test_fixtures['32_colors']
        
        # Create output with fewer colors
        output_gif = tmp_path / "reduced_colors.gif"
        ValidationTestFixtures.create_exact_color_gif(
            output_gif, color_count=16  # Half the colors
        )
        
        result = validator.validate_color_reduction(
            input_path=input_gif,
            output_path=output_gif,
            expected_colors=16,
            wrapper_metadata={"engine": "test"}
        )
        
        # Should pass validation
        assert result.is_valid is True
        assert result.validation_type == "color_count"
        assert result.actual <= 16 + validator.config.COLOR_COUNT_TOLERANCE
```

### Pattern 2: Mock-Based Testing

```python
# tests/test_validation_mocks.py
from unittest.mock import Mock, patch
import pytest

class TestValidationMocking:
    """Test validation logic with mocked dependencies."""
    
    @patch('giflab.wrapper_validation.core.extract_gif_metadata')
    def test_validation_with_mocked_metadata(self, mock_extract_metadata):
        """Test validation logic with controlled metadata."""
        
        # Setup mock metadata
        mock_metadata = Mock()
        mock_metadata.orig_frames = 8
        mock_metadata.orig_n_colors = 64
        mock_metadata.orig_fps = 12.0
        mock_metadata.orig_width = 200
        mock_metadata.orig_height = 150
        
        mock_extract_metadata.return_value = mock_metadata
        
        validator = WrapperOutputValidator()
        
        # Test with mocked metadata
        result = validator.validate_frame_reduction(
            input_path=Path("mock_input.gif"),
            output_path=Path("mock_output.gif"),
            expected_ratio=0.5,
            wrapper_metadata={"engine": "mock"}
        )
        
        # Verify behavior with mocked data
        assert mock_extract_metadata.call_count == 2  # Called for input and output
        assert result.expected["ratio"] == 0.5
    
    def test_validation_error_handling_with_mocks(self):
        """Test validation error handling with mocked failures."""
        
        validator = WrapperOutputValidator()
        
        # Mock file that doesn't exist
        nonexistent_path = Path("/nonexistent/file.gif")
        
        result = validator.validate_file_integrity(
            output_path=nonexistent_path,
            wrapper_metadata={"engine": "test"}
        )
        
        # Should handle error gracefully
        assert result.is_valid is False
        assert result.validation_type == "file_integrity"
        assert "does not exist" in result.error_message
```

---

## 🔧 Integration Testing Patterns

### Pattern 1: End-to-End Validation Testing

```python
# tests/test_validation_integration.py
class TestValidationIntegration:
    """Integration tests for complete validation workflows."""
    
    def test_wrapper_validation_integration(self, tmp_path):
        """Test full wrapper integration with validation."""
        
        from giflab.tool_wrappers import GifsicleFrameReducer
        from giflab.wrapper_validation.integration import validate_wrapper_apply_result
        
        # Create test input
        input_gif = tmp_path / "input.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            input_gif, frame_count=20, fps=10.0
        )
        
        output_gif = tmp_path / "output.gif"
        
        # Mock wrapper for controlled testing
        class MockFrameReducer:
            def __init__(self):
                self.NAME = "MockFrameReducer"
            
            def apply(self, input_path, output_path, params):
                # Simulate frame reduction
                expected_frames = int(20 * params.get("ratio", 1.0))
                ValidationTestFixtures.create_exact_frame_gif(
                    output_path, frame_count=expected_frames
                )
                
                return {
                    "success": True,
                    "compression_time": 1.5,
                    "original_frames": 20,
                    "output_frames": expected_frames
                }
        
        wrapper = MockFrameReducer()
        
        # Test validation integration
        result = wrapper.apply(input_gif, output_gif, {"ratio": 0.5})
        validated_result = validate_wrapper_apply_result(
            wrapper, input_gif, output_gif, {"ratio": 0.5}, result
        )
        
        # Verify integration
        assert "validations" in validated_result
        assert "validation_passed" in validated_result
        assert validated_result["validation_passed"] is True
        
        # Check specific validations
        frame_validations = [
            v for v in validated_result["validations"]
            if v["validation_type"] == "frame_count"
        ]
        assert len(frame_validations) == 1
        assert frame_validations[0]["is_valid"] is True
```

### Pattern 2: Pipeline Validation Testing

```python
class TestPipelineValidation:
    """Test multi-stage pipeline validation."""
    
    def test_multi_stage_pipeline_validation(self, tmp_path):
        """Test validation across multiple pipeline stages."""
        
        from giflab.wrapper_validation.pipeline_validation import PipelineStageValidator
        
        # Create test pipeline
        input_gif = tmp_path / "pipeline_input.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            input_gif, frame_count=30, fps=15.0
        )
        ValidationTestFixtures.create_exact_color_gif(
            input_gif, color_count=128  # Also set colors
        )
        
        # Simulate pipeline stages
        stage1_output = tmp_path / "stage1.gif"  # Frame reduction
        stage2_output = tmp_path / "stage2.gif"  # Color reduction
        
        # Create stage outputs
        ValidationTestFixtures.create_exact_frame_gif(
            stage1_output, frame_count=15  # 50% frame reduction
        )
        ValidationTestFixtures.create_exact_color_gif(
            stage2_output, color_count=32  # Color reduction
        )
        
        # Mock pipeline and steps
        mock_pipeline = Mock()
        mock_pipeline.identifier.return_value = "test_pipeline"
        mock_pipeline.steps = [
            Mock(variable="frame_reduction", tool_cls=Mock(__name__="MockFrameReducer")),
            Mock(variable="color_reduction", tool_cls=Mock(__name__="MockColorReducer"))
        ]
        
        validator = PipelineStageValidator()
        
        # Test pipeline validation
        stage_outputs = {
            "frame_reduction_MockFrameReducer": stage1_output,
            "color_reduction_MockColorReducer": stage2_output
        }
        
        stage_metadata = {
            "frame_reduction_MockFrameReducer": {"tool": "mock"},
            "color_reduction_MockColorReducer": {"tool": "mock"}
        }
        
        with patch('giflab.wrapper_validation.pipeline_validation.extract_gif_metadata') as mock_extract:
            # Mock metadata for different stages
            def metadata_side_effect(path):
                metadata = Mock()
                if "input" in str(path):
                    metadata.orig_frames = 30
                    metadata.orig_n_colors = 128
                elif "stage1" in str(path):
                    metadata.orig_frames = 15
                    metadata.orig_n_colors = 128
                elif "stage2" in str(path):
                    metadata.orig_frames = 15
                    metadata.orig_n_colors = 32
                
                metadata.orig_width = 100
                metadata.orig_height = 100
                metadata.orig_fps = 15.0
                return metadata
            
            mock_extract.side_effect = metadata_side_effect
            
            validations = validator.validate_pipeline_execution(
                input_path=input_gif,
                pipeline=mock_pipeline,
                pipeline_params={"frame_ratio": 0.5, "colors": 32},
                stage_outputs=stage_outputs,
                stage_metadata=stage_metadata,
                final_output_path=stage2_output
            )
        
        # Verify pipeline validation
        assert len(validations) > 0
        
        # Check for specific validation types
        validation_types = [v.validation_type for v in validations]
        assert "pipeline_overall_integrity" in validation_types
```

---

## ⚡ Performance Testing Patterns

### Pattern 1: Validation Overhead Testing

```python
# tests/test_validation_performance.py
import time
import statistics

class TestValidationPerformance:
    """Test validation system performance impact."""
    
    def test_validation_overhead_measurement(self, tmp_path):
        """Measure validation overhead compared to baseline."""
        
        # Create test GIF
        input_gif = tmp_path / "perf_test.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            input_gif, frame_count=50, width=200, height=200
        )
        
        output_gif = tmp_path / "output.gif"
        
        # Measure baseline (no validation)
        baseline_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            
            # Simulate compression without validation
            import shutil
            shutil.copy(input_gif, output_gif)
            
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)
        
        baseline_avg = statistics.mean(baseline_times)
        
        # Measure with validation
        validation_times = []
        validator = WrapperOutputValidator()
        
        for _ in range(10):
            start_time = time.perf_counter()
            
            # Simulate compression
            shutil.copy(input_gif, output_gif)
            
            # Add validation
            result = validator.validate_wrapper_output(
                input_path=input_gif,
                output_path=output_gif,
                wrapper_params={"ratio": 1.0},
                wrapper_metadata={"engine": "test"},
                wrapper_type="frame_reduction"
            )
            
            end_time = time.perf_counter()
            validation_times.append(end_time - start_time)
        
        validation_avg = statistics.mean(validation_times)
        overhead_percent = ((validation_avg - baseline_avg) / baseline_avg) * 100
        
        print(f"Baseline: {baseline_avg*1000:.2f}ms")
        print(f"With validation: {validation_avg*1000:.2f}ms") 
        print(f"Overhead: {overhead_percent:.1f}%")
        
        # Assert reasonable overhead (<20%)
        assert overhead_percent < 20.0, f"Validation overhead too high: {overhead_percent:.1f}%"
    
    def test_validation_scalability(self, tmp_path):
        """Test validation performance with different file sizes."""
        
        validator = WrapperOutputValidator()
        performance_data = []
        
        # Test with different file sizes
        frame_counts = [5, 20, 50, 100]
        
        for frame_count in frame_counts:
            input_gif = tmp_path / f"test_{frame_count}frames.gif"
            output_gif = tmp_path / f"output_{frame_count}frames.gif"
            
            # Create test GIF
            ValidationTestFixtures.create_exact_frame_gif(
                input_gif, frame_count=frame_count, width=150, height=150
            )
            
            # Copy as output
            import shutil
            shutil.copy(input_gif, output_gif)
            
            # Measure validation time
            start_time = time.perf_counter()
            
            validations = validator.validate_wrapper_output(
                input_path=input_gif,
                output_path=output_gif,
                wrapper_params={"ratio": 1.0},
                wrapper_metadata={"engine": "test"},
                wrapper_type="frame_reduction"
            )
            
            end_time = time.perf_counter()
            validation_time_ms = (end_time - start_time) * 1000
            
            performance_data.append({
                'frame_count': frame_count,
                'validation_time_ms': validation_time_ms,
                'file_size_bytes': input_gif.stat().st_size
            })
            
            print(f"{frame_count} frames: {validation_time_ms:.2f}ms")
        
        # Verify performance scales reasonably
        # Should not have exponential growth
        for i in range(1, len(performance_data)):
            prev_data = performance_data[i-1]
            curr_data = performance_data[i]
            
            time_growth = curr_data['validation_time_ms'] / prev_data['validation_time_ms']
            frame_growth = curr_data['frame_count'] / prev_data['frame_count']
            
            # Time growth should not significantly exceed frame growth
            assert time_growth < frame_growth * 2, f"Performance degradation too high: {time_growth:.2f}x"
```

---

## 🚨 Error Handling Testing Patterns

### Pattern 1: Graceful Failure Testing

```python
class TestValidationErrorHandling:
    """Test validation system error handling and recovery."""
    
    def test_validation_with_corrupted_files(self, tmp_path):
        """Test validation handles corrupted files gracefully."""
        
        validator = WrapperOutputValidator()
        
        # Create corrupted file
        corrupted_file = tmp_path / "corrupted.gif"
        corrupted_file.write_bytes(b"This is not a GIF file")
        
        valid_input = tmp_path / "valid_input.gif" 
        ValidationTestFixtures.create_exact_frame_gif(valid_input, frame_count=5)
        
        # Test file integrity validation
        result = validator.validate_file_integrity(
            output_path=corrupted_file,
            wrapper_metadata={"engine": "test"}
        )
        
        # Should fail gracefully
        assert result.is_valid is False
        assert result.validation_type == "file_integrity"
        assert result.error_message is not None
        assert "Cannot read output file as valid GIF" in result.error_message
    
    def test_validation_with_missing_files(self, tmp_path):
        """Test validation handles missing files gracefully."""
        
        validator = WrapperOutputValidator()
        
        missing_file = tmp_path / "does_not_exist.gif"
        
        result = validator.validate_file_integrity(
            output_path=missing_file,
            wrapper_metadata={"engine": "test"}
        )
        
        # Should handle missing file gracefully
        assert result.is_valid is False
        assert "does not exist" in result.error_message
    
    def test_validation_exception_handling(self, tmp_path):
        """Test validation handles internal exceptions gracefully."""
        
        validator = WrapperOutputValidator()
        
        # Mock a method to raise an exception
        original_method = validator._count_unique_colors
        
        def failing_method(path):
            raise Exception("Simulated internal error")
        
        validator._count_unique_colors = failing_method
        
        try:
            input_gif = tmp_path / "test.gif"
            output_gif = tmp_path / "output.gif"
            
            ValidationTestFixtures.create_exact_frame_gif(input_gif, frame_count=5)
            ValidationTestFixtures.create_exact_frame_gif(output_gif, frame_count=5)
            
            result = validator.validate_color_reduction(
                input_path=input_gif,
                output_path=output_gif,
                expected_colors=32,
                wrapper_metadata={"engine": "test"}
            )
            
            # Should handle exception gracefully
            assert result.is_valid is False
            assert result.validation_type == "color_count"
            assert "validation failed" in result.error_message.lower()
            
        finally:
            # Restore original method
            validator._count_unique_colors = original_method
```

### Pattern 2: Configuration Error Testing

```python
class TestConfigurationErrorHandling:
    """Test validation system with various configuration errors."""
    
    def test_invalid_tolerance_handling(self):
        """Test validation handles invalid tolerance values."""
        
        # Test with negative tolerance (should be handled)
        config = ValidationConfig(FRAME_RATIO_TOLERANCE=-0.1)
        validator = WrapperOutputValidator(config)
        
        # Should still work (internally handle invalid config)
        assert validator.config.FRAME_RATIO_TOLERANCE >= 0
    
    def test_extreme_configuration_values(self, tmp_path):
        """Test validation with extreme configuration values."""
        
        # Extremely permissive config
        permissive_config = ValidationConfig(
            FRAME_RATIO_TOLERANCE=10.0,      # 1000% tolerance
            COLOR_COUNT_TOLERANCE=1000,      # 1000 color tolerance
            FPS_TOLERANCE=100.0              # 10000% FPS tolerance
        )
        
        validator = WrapperOutputValidator(permissive_config)
        
        # Create very different input/output files
        input_gif = tmp_path / "input.gif"
        output_gif = tmp_path / "output.gif"
        
        ValidationTestFixtures.create_exact_frame_gif(input_gif, frame_count=100)
        ValidationTestFixtures.create_exact_frame_gif(output_gif, frame_count=1)  # Massive reduction
        
        result = validator.validate_frame_reduction(
            input_path=input_gif,
            output_path=output_gif,
            expected_ratio=0.01,  # 1% expected
            wrapper_metadata={"engine": "test"}
        )
        
        # Should pass due to extreme tolerance
        assert result.is_valid is True
```

---

## 📊 Test Data Management

### Pattern 1: Test Fixture Management

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_fixtures_dir():
    """Directory containing all test fixtures."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session") 
def validation_test_gifs(test_fixtures_dir, tmp_path_factory):
    """Create standard validation test GIFs."""
    
    fixtures_dir = tmp_path_factory.mktemp("validation_fixtures")
    fixtures = {}
    
    # Standard test cases
    test_cases = [
        {"name": "small_4_frames", "frames": 4, "colors": 16, "size": (50, 50)},
        {"name": "medium_20_frames", "frames": 20, "colors": 64, "size": (100, 100)},
        {"name": "large_100_frames", "frames": 100, "colors": 256, "size": (200, 200)},
        {"name": "high_color", "frames": 10, "colors": 256, "size": (150, 150)},
        {"name": "low_color", "frames": 10, "colors": 4, "size": (150, 150)},
        {"name": "single_frame", "frames": 1, "colors": 8, "size": (100, 100)},
    ]
    
    for case in test_cases:
        gif_path = fixtures_dir / f"{case['name']}.gif"
        
        # Create frame-focused fixture
        ValidationTestFixtures.create_exact_frame_gif(
            gif_path, 
            frame_count=case['frames'],
            width=case['size'][0],
            height=case['size'][1]
        )
        
        # Also create color-focused version
        color_gif_path = fixtures_dir / f"{case['name']}_colors.gif"
        ValidationTestFixtures.create_exact_color_gif(
            color_gif_path,
            color_count=case['colors'],
            frame_count=min(case['frames'], 10)  # Limit frames for color tests
        )
        
        fixtures[case['name']] = {
            'frame_focused': gif_path,
            'color_focused': color_gif_path,
            'properties': case
        }
    
    return fixtures

@pytest.fixture
def validation_config_variants():
    """Different validation configurations for testing."""
    return {
        'strict': ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.01,
            COLOR_COUNT_TOLERANCE=0,
            FPS_TOLERANCE=0.05
        ),
        'default': ValidationConfig(),
        'permissive': ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.2,
            COLOR_COUNT_TOLERANCE=10,
            FPS_TOLERANCE=0.3
        ),
        'performance': ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.15,
            COLOR_COUNT_TOLERANCE=5,
            LOG_VALIDATION_FAILURES=False
        )
    }
```

### Pattern 2: Parameterized Testing

```python
class TestValidationParameterized:
    """Parameterized tests for comprehensive validation coverage."""
    
    @pytest.mark.parametrize("frame_count,reduction_ratio", [
        (10, 0.5),   # 50% reduction
        (20, 0.3),   # 70% reduction  
        (50, 0.8),   # 20% reduction
        (100, 0.1),  # 90% reduction
    ])
    def test_frame_reduction_scenarios(self, tmp_path, frame_count, reduction_ratio):
        """Test frame reduction with various scenarios."""
        
        validator = WrapperOutputValidator()
        
        # Create input with specific frame count
        input_gif = tmp_path / "input.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            input_gif, frame_count=frame_count
        )
        
        # Create output with reduced frames
        expected_output_frames = max(1, int(frame_count * reduction_ratio))
        output_gif = tmp_path / "output.gif"
        ValidationTestFixtures.create_exact_frame_gif(
            output_gif, frame_count=expected_output_frames
        )
        
        result = validator.validate_frame_reduction(
            input_path=input_gif,
            output_path=output_gif,
            expected_ratio=reduction_ratio,
            wrapper_metadata={"engine": "test"}
        )
        
        # Should pass validation
        assert result.is_valid is True, f"Failed for {frame_count} frames, {reduction_ratio} ratio"
        assert abs(result.actual["frames"] - expected_output_frames) <= 1
    
    @pytest.mark.parametrize("config_name", ["strict", "default", "permissive"])
    def test_validation_with_different_configs(self, validation_config_variants, config_name, tmp_path):
        """Test validation behavior with different configurations."""
        
        config = validation_config_variants[config_name]
        validator = WrapperOutputValidator(config)
        
        # Create test scenario that might pass/fail based on config
        input_gif = tmp_path / "input.gif"
        output_gif = tmp_path / "output.gif"
        
        ValidationTestFixtures.create_exact_frame_gif(input_gif, frame_count=10)
        ValidationTestFixtures.create_exact_frame_gif(output_gif, frame_count=8)  # Slight difference
        
        result = validator.validate_frame_reduction(
            input_path=input_gif,
            output_path=output_gif,
            expected_ratio=0.5,  # Expect 50% reduction (5 frames)
            wrapper_metadata={"engine": "test"}
        )
        
        # Different configs should handle this differently
        if config_name == "strict":
            # Strict config might fail on the discrepancy
            pass  # Could pass or fail based on exact tolerance
        elif config_name == "permissive":
            # Permissive config should be more forgiving
            pass  # More likely to pass
        
        # Verify result structure is always valid
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'validation_type')
        assert result.validation_type == "frame_count"
```

---

## ✅ Testing Checklist

### Unit Test Coverage Checklist
- [ ] **Validation result accuracy** - Test correct validation logic
- [ ] **Error handling** - Test graceful failure scenarios  
- [ ] **Configuration handling** - Test various config combinations
- [ ] **Edge cases** - Test boundary conditions and unusual inputs
- [ ] **Performance** - Test validation overhead remains acceptable

### Integration Test Coverage Checklist
- [ ] **Wrapper integration** - Test validation integrates with wrappers correctly
- [ ] **Pipeline integration** - Test multi-stage pipeline validation
- [ ] **Configuration integration** - Test different configs work in practice
- [ ] **Real file testing** - Test with actual GIF files, not just fixtures

### Performance Test Coverage Checklist
- [ ] **Overhead measurement** - Measure validation vs baseline performance
- [ ] **Scalability testing** - Test performance with different file sizes
- [ ] **Memory usage** - Test validation doesn't leak memory
- [ ] **Concurrent validation** - Test validation under load

---

## 📚 Related Documentation

- [Wrapper Integration Guide](../guides/wrapper-validation-integration.md)
- [Configuration Reference](../reference/validation-config-reference.md)
- [Performance Optimization Guide](../technical/validation-performance-guide.md)
- [Testing Best Practices](../guides/testing-best-practices.md)

---

## 🔧 Test Utilities

All test utilities and fixtures mentioned in this guide are available in:
- `tests/fixtures/validation_test_fixtures.py` - Test fixture creation utilities
- `tests/utils/validation_test_helpers.py` - Helper functions for validation testing
- `tests/conftest.py` - Pytest fixtures and configuration