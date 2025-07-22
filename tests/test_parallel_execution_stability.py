from unittest.mock import MagicMock, patch

from src.giflab.experiment import ExperimentalConfig, ExperimentalPipeline


def test_run_experiment_uses_thread_pool(tmp_path):
    """Ensure ThreadPoolExecutor is used to avoid pickle issues on macOS/Windows."""
    # Minimal config to disable heavy processing
    cfg = ExperimentalConfig(ENABLE_DETAILED_ANALYSIS=False)
    runner = ExperimentalPipeline(cfg, workers=1)

    # Prepare dummy job and minimal mocks
    dummy_job = MagicMock(name="ExperimentJob")

    with patch.object(runner, "generate_sample_gifs", return_value=[]), \
         patch.object(runner, "generate_jobs", return_value=[dummy_job]), \
         patch.object(runner, "execute_job", return_value=MagicMock(success=True, job=dummy_job, metrics={}, compression_result={})), \
         patch.object(runner, "_write_result_to_csv"), \
         patch("src.giflab.experiment.ThreadPoolExecutor") as mock_executor:

        # Mock the context manager behaviour of ThreadPoolExecutor
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = MagicMock(
            success=True,
            job=dummy_job,
            metrics={},
            compression_result={},
        )
        # Mock as_completed to yield the mocked future
        with patch("src.giflab.experiment.as_completed", return_value=[mock_executor.return_value.__enter__.return_value.submit.return_value]):
            runner.run_experiment(sample_gifs=[])

        # Assert that ThreadPoolExecutor was used
        mock_executor.assert_called_once()
