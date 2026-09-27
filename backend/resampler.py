"""Strict timestamp-based temporal resampler to guarantee exact 25.0 FPS video input."""

from typing import List, Optional, Tuple
import numpy as np


class TemporalResampler:
    """Resamples variable or non-25 FPS video streams to exact target FPS (default 25.0 FPS / 40ms)."""

    def __init__(self, target_fps: float = 25.0) -> None:
        self.target_fps = target_fps
        self.interval_ms = 1000.0 / target_fps
        self.buffer: List[Tuple[np.ndarray, float]] = []
        self.next_target_ms: Optional[float] = None
        self.start_timestamp_ms: Optional[float] = None

    def reset(self) -> None:
        """Reset internal timeline and buffers."""
        self.buffer.clear()
        self.next_target_ms = None
        self.start_timestamp_ms = None

    def add_frame(self, frame: np.ndarray, timestamp_ms: float) -> List[np.ndarray]:
        """Add an incoming frame with its capture timestamp in milliseconds.

        Returns zero, one, or more frames aligned to the exact 40ms grid.
        """
        self.buffer.append((frame, timestamp_ms))

        # Sort buffer by timestamp just in case packets arrive slightly out of order
        self.buffer.sort(key=lambda x: x[1])

        if self.next_target_ms is None:
            self.start_timestamp_ms = timestamp_ms
            self.next_target_ms = timestamp_ms

        emitted_frames: List[np.ndarray] = []

        # Keep emitting as long as the buffer has frames spanning beyond next_target_ms
        while len(self.buffer) >= 2 and self.buffer[-1][1] >= self.next_target_ms:
            # Find the frame closest to next_target_ms among available frames
            best_idx = 0
            best_diff = abs(self.buffer[0][1] - self.next_target_ms)
            for i, (_, ts) in enumerate(self.buffer):
                diff = abs(ts - self.next_target_ms)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            emitted_frames.append(self.buffer[best_idx][0])
            self.next_target_ms += self.interval_ms

            # Prune buffer of frames strictly older than next_target_ms - 2 * interval_ms
            cutoff_ms = self.next_target_ms - (self.interval_ms * 2)
            self.buffer = [item for item in self.buffer if item[1] >= cutoff_ms]

        return emitted_frames

    def flush(self) -> List[np.ndarray]:
        """Flush remaining frames at stream end to complete the sequence."""
        emitted_frames: List[np.ndarray] = []
        if not self.buffer or self.next_target_ms is None:
            return emitted_frames

        last_frame, last_ts = self.buffer[-1]
        while self.next_target_ms <= last_ts:
            best_idx = 0
            best_diff = abs(self.buffer[0][1] - self.next_target_ms)
            for i, (_, ts) in enumerate(self.buffer):
                diff = abs(ts - self.next_target_ms)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            emitted_frames.append(self.buffer[best_idx][0])
            self.next_target_ms += self.interval_ms

        self.reset()
        return emitted_frames
