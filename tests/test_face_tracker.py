"""Unit tests for FaceTracker."""

import cv2
import numpy as np
import pytest
from backend.face_tracker import FaceTracker


def test_face_tracker_no_face():
    """Verify that FaceTracker returns None when provided an empty black frame."""
    tracker = FaceTracker()
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(empty_frame)

    assert result is None
    assert not tracker.is_face_detected()
    assert tracker.get_latest_mouth_landmarks() is None


def test_face_tracker_invalid_input():
    """Verify that FaceTracker handles None or empty input gracefully."""
    tracker = FaceTracker()
    assert tracker.process_frame(None) is None
    assert tracker.process_frame(np.array([])) is None
