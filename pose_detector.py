from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class PoseDetector:
    """MediaPipe pose detector for a single frame/region."""

    def __init__(self):
        self.enabled = bool(mp is not None and hasattr(mp, "solutions"))
        self.warning_text = None

        if self.enabled:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                # Real-time stream mode for smoother tracking across frames.
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.pose = None
            self.warning_text = "MediaPipe Pose unavailable on this Python build (try Python 3.11)."

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Returns:
            landmarks: np.ndarray shape (33, 2) with normalized (x, y), or None.
            output_frame: input frame with skeleton overlay if detected.
        """
        output = frame_bgr.copy()
        if not self.enabled:
            cv2.putText(
                output,
                self.warning_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            return None, output

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return None, output

        self.mp_drawing.draw_landmarks(
            output,
            result.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        points = []
        for lm in result.pose_landmarks.landmark:
            points.append((lm.x, lm.y))

        return np.array(points, dtype=np.float32), output

    def close(self):
        if self.pose is not None:
            self.pose.close()
