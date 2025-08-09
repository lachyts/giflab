"""Multiprocessing support for GifLab frame generation and pipeline execution.

This module provides process-safe multiprocessing capabilities for:
1. Parallel frame generation in synthetic GIF creation
2. Parallel pipeline execution with proper DB write coordination
3. Process-safe queue management for cache system integration

Performance improvements target CPU-intensive operations while maintaining
data integrity through proper synchronization mechanisms.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .synthetic_gifs import SyntheticFrameGenerator, SyntheticGifSpec


@dataclass
class FrameGenerationTask:
    """Task specification for parallel frame generation."""

    content_type: str
    size: tuple[int, int]
    frame_index: int
    total_frames: int
    task_id: str  # Unique identifier for the task


@dataclass
class FrameGenerationResult:
    """Result from parallel frame generation."""

    task_id: str
    frame_index: int
    success: bool
    image_data: bytes | None = None
    error_message: str | None = None
    generation_time: float = 0.0


class ProcessSafeQueue:
    """Process-safe queue wrapper for DB write coordination.

    This queue ensures that database writes from multiple processes
    are properly serialized to prevent corruption and race conditions.
    """

    def __init__(self, maxsize: int = 0):
        """Initialize the process-safe queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue = mp.Queue(maxsize)
        self._lock = mp.Lock()

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        """Put an item in the queue safely."""
        with self._lock:
            self._queue.put(item, block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Get an item from the queue safely."""
        with self._lock:
            return self._queue.get(block, timeout)

    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return self._queue.empty()

    def qsize(self) -> int:
        """Get approximate queue size."""
        with self._lock:
            return self._queue.qsize()


class ParallelFrameGenerator:
    """Parallel frame generator using multiprocessing for performance.

    This class coordinates multiple worker processes to generate synthetic
    GIF frames in parallel, providing significant speedup for large datasets
    or high-resolution images.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        chunk_size: int = 1,
        logger: logging.Logger | None = None,
    ):
        """Initialize the parallel frame generator.

        Args:
            max_workers: Maximum number of worker processes (default: CPU count)
            chunk_size: Number of tasks per worker batch
            logger: Logger instance for debugging
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)

        # Performance tracking
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = 0.0

    def generate_frames_parallel(
        self,
        tasks: list[FrameGenerationTask],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[FrameGenerationResult]:
        """Generate multiple frames in parallel.

        Args:
            tasks: List of frame generation tasks
            progress_callback: Optional callback for progress updates (completed, total)

        Returns:
            List of frame generation results
        """
        if not tasks:
            return []

        self._total_tasks = len(tasks)
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = time.time()

        self.logger.info(
            f"Starting parallel frame generation: {self._total_tasks} tasks "
            f"with {self.max_workers} workers"
        )

        results = []

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(_generate_single_frame, task): task
                    for task in tasks
                }

                # Collect results as they complete
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)

                        if result.success:
                            self._completed_tasks += 1
                        else:
                            self._failed_tasks += 1
                            self.logger.warning(
                                f"Frame generation failed: {result.error_message}"
                            )

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                self._completed_tasks + self._failed_tasks,
                                self._total_tasks,
                            )

                    except Exception as e:
                        self._failed_tasks += 1
                        task = future_to_task[future]
                        self.logger.error(
                            f"Frame generation exception for task {task.task_id}: {e}"
                        )

                        results.append(
                            FrameGenerationResult(
                                task_id=task.task_id,
                                frame_index=task.frame_index,
                                success=False,
                                error_message=str(e),
                            )
                        )

        except Exception as e:
            self.logger.error(f"Critical error in parallel frame generation: {e}")
            raise

        elapsed_time = time.time() - self._start_time
        self.logger.info(
            f"Parallel frame generation completed: {self._completed_tasks} successful, "
            f"{self._failed_tasks} failed in {elapsed_time:.2f}s "
            f"({self._total_tasks/elapsed_time:.1f} frames/sec)"
        )

        return results

    def generate_gif_frames_parallel(
        self,
        spec: SyntheticGifSpec,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Image.Image]:
        """Generate all frames for a GIF specification in parallel.

        Args:
            spec: Synthetic GIF specification
            progress_callback: Optional progress callback

        Returns:
            List of PIL Images in frame order
        """
        # Create tasks for all frames
        tasks = [
            FrameGenerationTask(
                content_type=spec.content_type,
                size=spec.size,
                frame_index=frame_idx,
                total_frames=spec.frames,
                task_id=f"{spec.name}_frame_{frame_idx}",
            )
            for frame_idx in range(spec.frames)
        ]

        # Generate frames in parallel
        results = self.generate_frames_parallel(tasks, progress_callback)

        # Sort results by frame index and convert to images
        results.sort(key=lambda r: r.frame_index)
        images = []

        for result in results:
            if result.success and result.image_data:
                try:
                    # Convert bytes back to PIL Image
                    from io import BytesIO

                    img = Image.open(BytesIO(result.image_data))
                    images.append(img)
                except Exception as e:
                    self.logger.error(f"Failed to convert frame data to image: {e}")
                    # Create a fallback image
                    fallback = Image.new(
                        "RGB", spec.size, (255, 0, 0)
                    )  # Red error frame
                    images.append(fallback)
            else:
                self.logger.warning(
                    f"Using fallback for failed frame {result.frame_index}"
                )
                # Create a fallback image
                fallback = Image.new("RGB", spec.size, (255, 0, 0))  # Red error frame
                images.append(fallback)

        return images


def _generate_single_frame(task: FrameGenerationTask) -> FrameGenerationResult:
    """Worker function for generating a single frame.

    This function runs in a separate process and generates one frame
    based on the task specification.

    Args:
        task: Frame generation task

    Returns:
        Frame generation result
    """
    start_time = time.time()

    try:
        # Create frame generator in worker process
        generator = SyntheticFrameGenerator()

        # Generate the frame
        image = generator.create_frame(
            task.content_type, task.size, task.frame_index, task.total_frames
        )

        # Convert image to bytes for inter-process transfer
        from io import BytesIO

        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        generation_time = time.time() - start_time

        return FrameGenerationResult(
            task_id=task.task_id,
            frame_index=task.frame_index,
            success=True,
            image_data=img_data,
            generation_time=generation_time,
        )

    except Exception as e:
        generation_time = time.time() - start_time

        return FrameGenerationResult(
            task_id=task.task_id,
            frame_index=task.frame_index,
            success=False,
            error_message=str(e),
            generation_time=generation_time,
        )


class ParallelPipelineExecutor:
    """Parallel pipeline executor with process-safe DB coordination.

    This class manages parallel execution of compression pipelines while
    ensuring that database writes are properly coordinated to prevent
    race conditions and data corruption.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        db_write_queue_size: int = 1000,
        logger: logging.Logger | None = None,
    ):
        """Initialize the parallel pipeline executor.

        Args:
            max_workers: Maximum number of worker processes
            db_write_queue_size: Size of the DB write coordination queue
            logger: Logger instance
        """
        self.max_workers = max_workers or max(
            1, mp.cpu_count() - 1
        )  # Leave one CPU for coordination
        self.db_write_queue = ProcessSafeQueue(db_write_queue_size)
        self.logger = logger or logging.getLogger(__name__)

        # Coordination for graceful shutdown
        self._shutdown_event = mp.Event()
        self._writer_thread: threading.Thread | None = None

    def start_db_writer_thread(self, db_writer_func: Callable[[Any], None]) -> None:
        """Start the database writer thread for coordinated writes.

        Args:
            db_writer_func: Function to handle database writes
        """
        if self._writer_thread and self._writer_thread.is_alive():
            self.logger.warning("DB writer thread already running")
            return

        self._shutdown_event.clear()
        self._writer_thread = threading.Thread(
            target=self._db_writer_worker, args=(db_writer_func,), daemon=True
        )
        self._writer_thread.start()
        self.logger.info("Database writer thread started")

    def stop_db_writer_thread(self) -> None:
        """Stop the database writer thread gracefully."""
        if not self._writer_thread or not self._writer_thread.is_alive():
            return

        self._shutdown_event.set()

        # Add sentinel to wake up the writer thread
        try:
            self.db_write_queue.put(None, block=False)
        except queue.Full:
            pass

        self._writer_thread.join(timeout=5.0)

        if self._writer_thread.is_alive():
            self.logger.warning("DB writer thread did not stop gracefully")
        else:
            self.logger.info("Database writer thread stopped")

    def _db_writer_worker(self, db_writer_func: Callable[[Any], None]) -> None:
        """Worker function for the database writer thread.

        This function runs in a separate thread and processes database
        write requests from the queue in a serialized manner.
        """
        self.logger.info("DB writer worker started")

        while not self._shutdown_event.is_set():
            try:
                # Get write request from queue
                item = self.db_write_queue.get(timeout=1.0)

                # Check for shutdown sentinel
                if item is None:
                    break

                # Process the write request
                try:
                    db_writer_func(item)
                except Exception as e:
                    self.logger.error(f"Database write error: {e}")

            except queue.Empty:
                continue  # Timeout, check shutdown flag
            except Exception as e:
                self.logger.error(f"DB writer worker error: {e}")

        self.logger.info("DB writer worker stopped")

    def queue_db_write(self, data: Any, timeout: float = 5.0) -> bool:
        """Queue a database write operation.

        Args:
            data: Data to write to database
            timeout: Timeout for queue operation

        Returns:
            True if successfully queued, False otherwise
        """
        try:
            self.db_write_queue.put(data, block=True, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning("DB write queue is full, dropping write request")
            return False
        except Exception as e:
            self.logger.error(f"Failed to queue DB write: {e}")
            return False


def setup_multiprocessing_logging(log_level: int = logging.INFO) -> None:
    """Setup logging for multiprocessing workers.

    Args:
        log_level: Logging level for worker processes
    """
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_optimal_worker_count(task_type: str = "frame_generation") -> int:
    """Get optimal worker count for different task types.

    Args:
        task_type: Type of task ("frame_generation" or "pipeline_execution")

    Returns:
        Optimal number of worker processes
    """
    cpu_count = mp.cpu_count()

    if task_type == "frame_generation":
        # Frame generation is CPU-intensive, use all cores
        return cpu_count
    elif task_type == "pipeline_execution":
        # Pipeline execution involves I/O, leave one core for coordination
        return max(1, cpu_count - 1)
    else:
        # Default conservative approach
        return max(1, cpu_count // 2)


# Signal handlers for graceful shutdown
def _signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logging.getLogger(__name__).info(
        f"Received signal {signum}, shutting down gracefully..."
    )
    # The actual cleanup will be handled by the context managers


# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
