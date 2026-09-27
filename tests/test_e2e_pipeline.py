"""End-to-end pipeline integration test."""

import pytest
import numpy as np
from backend.resampler import TemporalResampler
from backend.vad import VisualVAD
from backend.vsr_engine import merge_sliding_tokens
from backend.llm_resolver import format_sentence_case


def test_pipeline_streaming_flow():
    """Verify data flow across Resampler -> VAD -> Sliding Window Merger -> LLM Formatter."""
    resampler = TemporalResampler(target_fps=25.0)
    vad = VisualVAD(lar_threshold=0.15, silence_duration_ms=400.0)

    # 1. Simulate 30 incoming frames across 1000ms (~33.3ms interval)
    incoming_frames = []
    for i in range(30):
        # 96x96 synthetic crop
        frame = np.full((96, 96), i, dtype=np.uint8)
        ts = i * (1000.0 / 30.0)
        emitted = resampler.add_frame(frame, ts)
        incoming_frames.extend(emitted)
    incoming_frames.extend(resampler.flush())

    assert len(incoming_frames) == 25, "Must strictly resample 30 FPS to 25.0 FPS"

    # 2. Simulate VAD updates
    open_mouth = np.array([[50, 45], [50, 58], [30, 51], [70, 51]])
    is_speaking = False
    for i, frame in enumerate(incoming_frames):
        ts = i * 40.0
        is_speaking = vad.update(frame, open_mouth, ts)

    assert is_speaking, "VAD must detect active speech"

    # 3. Simulate sliding window output merging
    prev_chunk = "HELLO HOW"
    new_chunk = "HOW ARE YOU DOING"
    merged, committed = merge_sliding_tokens(prev_chunk, new_chunk)
    assert merged == "HELLO HOW ARE YOU DOING"

    # 4. Simulate sentence formatting
    formatted = format_sentence_case(merged)
    assert formatted == "Hello how are you doing"
