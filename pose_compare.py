from typing import Dict, Optional
import numpy as np

JOINT_TRIPLETS = {
    # BlazePose landmark indices (stable across MediaPipe pose variants).
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
}


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_joint_angles(landmarks: Optional[np.ndarray]) -> Dict[str, float]:
    if landmarks is None or len(landmarks) < 33:
        return {}

    angles = {}
    for joint_name, (i, j, k) in JOINT_TRIPLETS.items():
        angles[joint_name] = calculate_angle(landmarks[i], landmarks[j], landmarks[k])
    return angles


def score_pose(player_angles: Dict[str, float], target_angles: Dict[str, float]) -> float:
    """
    score = 100 - average_angle_difference
    clamped to 0..100
    """
    common = [j for j in target_angles.keys() if j in player_angles]
    if not common:
        return 0.0

    diffs = [abs(player_angles[j] - target_angles[j]) for j in common]
    avg_diff = float(np.mean(diffs))
    score = 100.0 - avg_diff
    return float(np.clip(score, 0.0, 100.0))


def performance_message(score: float) -> str:
    if score >= 85:
        return "Perfect Pose!"
    if score >= 60:
        return "Almost There!"
    return "What was that pose?"
