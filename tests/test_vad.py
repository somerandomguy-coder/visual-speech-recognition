"""Unit tests for VisualVAD."""

import numpy as np
from backend.vad import VisualVAD


def test_vad_speech_detection():
    vad = VisualVAD(lar_threshold=0.15, silence_duration_ms=400.0)

    # Mouth closed: upper and lower lip close together
    closed_landmarks = np.array([
        [50.0, 50.0],  # upper lip
        [50.0, 52.0],  # lower lip (2px apart)
        [30.0, 51.0],  # left corner (40px width)
        [70.0, 51.0],  # right corner
    ])
    # LAR = 2 / 40 = 0.05 (< 0.15)
    crop = np.zeros((96, 96), dtype=np.uint8)
    assert not vad.update(crop, closed_landmarks, timestamp_ms=0.0)
    assert not vad.is_speaking()

    # Mouth opens: vertical distance expands to 12px
    # LAR = 12 / 40 = 0.30 (>= 0.15)
    open_landmarks = np.array([
        [50.0, 45.0],
        [50.0, 57.0],
        [30.0, 51.0],
        [70.0, 51.0],
    ])
    assert vad.update(crop, open_landmarks, timestamp_ms=40.0)
    assert vad.is_speaking()


def test_vad_phrase_completion_after_silence():
    vad = VisualVAD(lar_threshold=0.15, silence_duration_ms=400.0)
    crop = np.zeros((96, 96), dtype=np.uint8)

    open_landmarks = np.array([[50, 45], [50, 57], [30, 51], [70, 51]])
    closed_landmarks = np.array([[50, 50], [50, 52], [30, 51], [70, 51]])

    # Speaking at t=0 and t=40
    vad.update(crop, open_landmarks, 0.0)
    vad.update(crop, open_landmarks, 40.0)
    assert vad.is_speaking()
    assert not vad.is_phrase_complete()

    # Mouth closes at t=80
    vad.update(crop, closed_landmarks, 80.0)
    # At t=200, only 160ms of silence elapsed (< 400ms)
    vad.update(crop, closed_landmarks, 200.0)
    assert not vad.is_phrase_complete()

    # At t=500, 460ms of silence elapsed (>= 400ms)
    vad.update(crop, closed_landmarks, 500.0)
    assert not vad.is_speaking()
    assert vad.is_phrase_complete()
