"""Unit tests for TemporalResampler."""

import numpy as np
from backend.resampler import TemporalResampler


def test_resample_30fps_to_25fps():
    """Verify that 30 incoming frames across 1000ms are resampled to exactly 25 frames."""
    resampler = TemporalResampler(target_fps=25.0)
    output_frames = []

    # Simulate 30 frames spanning exactly 1000ms (~33.33ms intervals)
    for i in range(30):
        frame = np.full((96, 96, 3), i, dtype=np.uint8)
        ts = i * (1000.0 / 30.0)
        emitted = resampler.add_frame(frame, ts)
        output_frames.extend(emitted)

    output_frames.extend(resampler.flush())

    # 1 second of video at 25 FPS should yield approximately 25 frames
    assert len(output_frames) == 25
    # Verify frame shape is preserved
    assert output_frames[0].shape == (96, 96, 3)


def test_resample_variable_fps():
    """Verify handling of jittery frame rates (e.g., dropping from 30 FPS to 15 FPS)."""
    resampler = TemporalResampler(target_fps=25.0)
    output_frames = []

    timestamps = [0.0, 30.0, 70.0, 110.0, 200.0, 240.0, 280.0, 320.0, 400.0]
    for i, ts in enumerate(timestamps):
        frame = np.full((96, 96), i, dtype=np.uint8)
        output_frames.extend(resampler.add_frame(frame, ts))

    output_frames.extend(resampler.flush())
    # Up to 400ms at 40ms interval = 10 or 11 frames (t=0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400)
    assert 10 <= len(output_frames) <= 11
