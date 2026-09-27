"""Face tracking and canonical affine mouth cropping using MediaPipe Face Mesh."""

from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np


class FaceTracker:
    """Tracks face landmarks and produces canonically aligned 96x96 grayscale mouth crops."""

    def __init__(
        self,
        ema_alpha: float = 0.7,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Canonical destination points for 96x96 mouth ROI:
        # [0] Left corner, [1] Right corner, [2] Upper lip center
        self.dst_canonical_pts = np.float32([
            [24.0, 48.0],
            [72.0, 48.0],
            [48.0, 38.0],
        ])

        self.prev_mouth_pts: Optional[np.ndarray] = None
        self.last_head_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
        self.face_detected: bool = False
        self.latest_mouth_landmarks: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset temporal smoothing filters."""
        self.prev_mouth_pts = None
        self.last_head_pose = (0.0, 0.0, 0.0)
        self.face_detected = False
        self.latest_mouth_landmarks = None

    def is_face_detected(self) -> bool:
        """Check if a face was detected in the most recent processed frame."""
        return self.face_detected

    def get_head_pose(self) -> Tuple[float, float, float]:
        """Return the estimated (yaw, pitch, roll) angles in degrees from the latest frame."""
        return self.last_head_pose

    def get_latest_mouth_landmarks(self) -> Optional[np.ndarray]:
        """Return array of shape (4, 2): [upper_lip, lower_lip, left_corner, right_corner]."""
        return self.latest_mouth_landmarks

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Process an input BGR frame and return a canonical 96x96 grayscale mouth ROI.

        Returns None if no face is detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            self.face_detected = False
            return None

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self.face_detected = False
            return None

        self.face_detected = True
        landmarks = results.multi_face_landmarks[0].landmark

        # Extract pixel coordinates for key landmarks:
        # #61: left mouth corner, #291: right mouth corner
        # #0: upper lip center, #17: lower lip center
        # #33: left eye corner, #263: right eye corner, #1: nose tip
        def get_pt(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h], dtype=np.float32)

        pt_left_corner = get_pt(61)
        pt_right_corner = get_pt(291)
        pt_upper_lip = get_pt(0)
        pt_lower_lip = get_pt(17)
        pt_left_eye = get_pt(33)
        pt_right_eye = get_pt(263)
        pt_nose = get_pt(1)

        # Store for V-VAD usage
        self.latest_mouth_landmarks = np.array([
            pt_upper_lip,
            pt_lower_lip,
            pt_left_corner,
            pt_right_corner,
        ], dtype=np.float32)

        # Compute approximate head pose angles (degrees)
        dx = pt_right_eye[0] - pt_left_eye[0]
        dy = pt_right_eye[1] - pt_left_eye[1]
        roll = float(np.degrees(np.arctan2(dy, dx)))

        eye_center = (pt_left_eye + pt_right_eye) / 2.0
        mouth_center = (pt_left_corner + pt_right_corner) / 2.0
        face_center = (eye_center + mouth_center) / 2.0

        yaw = float(np.degrees(np.arctan2(pt_nose[0] - face_center[0], w * 0.2)))
        pitch = float(np.degrees(np.arctan2(pt_nose[1] - face_center[1], h * 0.2)))
        self.last_head_pose = (yaw, pitch, roll)

        # Source affine points: left corner, right corner, upper lip
        src_pts = np.float32([pt_left_corner, pt_right_corner, pt_upper_lip])

        # Apply Exponential Moving Average (EMA) to smooth landmark jitter
        if self.prev_mouth_pts is not None:
            src_pts = self.ema_alpha * src_pts + (1.0 - self.ema_alpha) * self.prev_mouth_pts
        self.prev_mouth_pts = src_pts.copy()

        # Compute affine transformation matrix mapping to canonical (96, 96) frontal view
        M = cv2.getAffineTransform(src_pts, self.dst_canonical_pts)

        # Warp input frame to standardized 96x96
        warped_bgr = cv2.warpAffine(
            frame_bgr,
            M,
            (96, 96),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Convert to grayscale
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

        return warped_gray
