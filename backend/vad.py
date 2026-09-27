"""Visual Voice Activity Detection (V-VAD) using mouth kinematics and lip aspect ratio."""

from typing import Optional
import numpy as np


class VisualVAD:
    """Detects visual speech activity (mouth motion and opening) to trigger phrase boundaries."""

    def __init__(
        self,
        lar_threshold: float = 0.15,
        motion_threshold: float = 2.0,
        silence_duration_ms: float = 400.0,
    ) -> None:
        self.lar_threshold = lar_threshold
        self.motion_threshold = motion_threshold
        self.silence_duration_ms = silence_duration_ms

        self.prev_crop: Optional[np.ndarray] = None
        self.last_speech_time_ms: Optional[float] = None
        self.speech_start_time_ms: Optional[float] = None
        self.current_time_ms: float = 0.0
        self.is_currently_speaking: bool = False
        self.has_spoken_in_turn: bool = False

    def reset(self) -> None:
        """Reset VAD internal state."""
        self.prev_crop = None
        self.last_speech_time_ms = None
        self.speech_start_time_ms = None
        self.is_currently_speaking = False
        self.has_spoken_in_turn = False

    def compute_lar(self, landmarks_mouth: Optional[np.ndarray]) -> float:
        """Compute Lip Aspect Ratio: vertical height / horizontal width.

        landmarks_mouth expects array of shape (4, 2):
        [0]: upper lip center (x, y)
        [1]: lower lip center (x, y)
        [2]: left mouth corner (x, y)
        [3]: right mouth corner (x, y)
        """
        if landmarks_mouth is None or len(landmarks_mouth) < 4:
            return 0.0

        p_upper = landmarks_mouth[0]
        p_lower = landmarks_mouth[1]
        p_left = landmarks_mouth[2]
        p_right = landmarks_mouth[3]

        vertical_dist = np.linalg.norm(p_upper - p_lower)
        horizontal_dist = np.linalg.norm(p_left - p_right)

        if horizontal_dist < 1e-5:
            return 0.0

        return float(vertical_dist / horizontal_dist)

    def compute_motion_delta(self, mouth_crop_96x96: np.ndarray) -> float:
        """Compute mean absolute difference between consecutive 96x96 crops."""
        if self.prev_crop is None:
            self.prev_crop = mouth_crop_96x96.copy()
            return 0.0

        diff = np.mean(np.abs(mouth_crop_96x96.astype(float) - self.prev_crop.astype(float)))
        self.prev_crop = mouth_crop_96x96.copy()
        return float(diff)

    def update(
        self,
        mouth_crop_96x96: np.ndarray,
        landmarks_mouth: Optional[np.ndarray],
        timestamp_ms: float,
    ) -> bool:
        """Process a new frame and return True if speech is detected in this frame."""
        self.current_time_ms = timestamp_ms

        lar = self.compute_lar(landmarks_mouth)
        motion = self.compute_motion_delta(mouth_crop_96x96)

        # Active if either mouth is open or kinematics show significant lip movement
        frame_speaking = (lar >= self.lar_threshold) or (motion >= self.motion_threshold)

        if frame_speaking:
            self.last_speech_time_ms = timestamp_ms
            if not self.is_currently_speaking:
                self.speech_start_time_ms = timestamp_ms
            self.is_currently_speaking = True
            self.has_spoken_in_turn = True
        else:
            # Check if silence exceeded
            if (
                self.last_speech_time_ms is not None
                and (timestamp_ms - self.last_speech_time_ms) >= self.silence_duration_ms
            ):
                self.is_currently_speaking = False

        return self.is_currently_speaking

    def is_speaking(self) -> bool:
        """Check if speaker is currently active."""
        return self.is_currently_speaking

    def is_phrase_complete(self) -> bool:
        """Check if a phrase was spoken and has now been followed by silence."""
        if not self.has_spoken_in_turn or self.last_speech_time_ms is None:
            return False

        silence_elapsed = self.current_time_ms - self.last_speech_time_ms
        return (not self.is_currently_speaking) and (silence_elapsed >= self.silence_duration_ms)
