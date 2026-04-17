from typing import Optional, Tuple

import cv2
import numpy as np

mp_import_error = None
mp_solutions = None

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp_solutions = mp.solutions
    else:
        try:
            from mediapipe.python import solutions as mp_solutions
        except ImportError as exc:
            mp_import_error = exc
except Exception as exc:
    mp = None
    mp_import_error = exc


class PoseDetector:
    """MediaPipe pose detector for a single frame/region."""

    def __init__(self):
        self.enabled = bool(mp is not None and mp_solutions is not None)
        self.warning_text = None

        if self.enabled:
            self.mp_pose = mp_solutions.pose
            self.mp_drawing = mp_solutions.drawing_utils
            self.mp_drawing_styles = mp_solutions.drawing_styles
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
            if mp is None:
                error_text = str(mp_import_error) if mp_import_error is not None else ""
                if "module 'attr' has no attribute 's'" in error_text:
                    self.warning_text = (
                        "Broken attrs install in current venv. "
                        "Run with .\\.venv311\\Scripts\\python.exe or reinstall attrs."
                    )
                else:
                    self.warning_text = "MediaPipe is not installed. Use .\\.venv311\\Scripts\\python.exe."
            elif mp_import_error is not None:
                self.warning_text = (
                    "MediaPipe Pose API not available in this interpreter. "
                    "Use Python 3.11 with mediapipe==0.10.8."
                )
            else:
                self.warning_text = (
                    "MediaPipe Pose unavailable on this Python build. "
                    "Use .\\.venv311\\Scripts\\python.exe."
                )

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
