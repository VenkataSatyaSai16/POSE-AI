import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from pose_compare import JOINT_TRIPLETS, extract_joint_angles
from pose_detector import PoseDetector


@dataclass
class PoseReference:
    name: str
    display_name: str
    image_path: Optional[str]
    image: Optional[np.ndarray]
    target_angles: Dict[str, float]


DEFAULT_POSE_TARGETS = {
  "bharatanatyam": {
    "left_elbow": 95.0,
    "right_elbow": 95.0,
    "left_shoulder": 90.0,
    "right_shoulder": 90.0,
    "left_hip": 130.0,
    "right_hip": 130.0,
    "left_knee": 110.0,
    "right_knee": 110.0
  },
  "bodybuilderpose": {
    "left_elbow": 65.0,
    "right_elbow": 65.0,
    "left_shoulder": 70.0,
    "right_shoulder": 70.0,
    "left_hip": 170.0,
    "right_hip": 170.0,
    "left_knee": 170.0,
    "right_knee": 170.0
  },
  "child": {
    "left_elbow": 152.3325208783005,
    "right_elbow": 81.69704105096073,
    "left_shoulder": 133.2351056560541,
    "right_shoulder": 151.64656412561072,
    "left_hip": 116.82204421718191,
    "right_hip": 95.31088394288007,
    "left_knee": 155.99136611116154,
    "right_knee": 120.86947057228008
  },
  "childpose": {
    "left_elbow": 160.0,
    "right_elbow": 160.0,
    "left_shoulder": 60.0,
    "right_shoulder": 60.0,
    "left_hip": 45.0,
    "right_hip": 45.0,
    "left_knee": 160.0,
    "right_knee": 160.0
  },
  "classical": {
    "left_elbow": 152.72758687984805,
    "right_elbow": 60.53671792833116,
    "left_shoulder": 159.4235028446727,
    "right_shoulder": 167.61196907683143,
    "left_hip": 110.81497875918457,
    "right_hip": 129.30201760846504,
    "left_knee": 94.17292402430544,
    "right_knee": 166.08012915607623
  },
  "dancepose1": {
    "left_elbow": 130.0,
    "right_elbow": 95.0,
    "left_shoulder": 135.0,
    "right_shoulder": 95.0,
    "left_hip": 145.0,
    "right_hip": 155.0,
    "left_knee": 165.0,
    "right_knee": 165.0
  },
  "dancepose2": {
    "left_elbow": 138.7471776338149,
    "right_elbow": 79.85251078789479,
    "left_shoulder": 139.75094669399908,
    "right_shoulder": 157.7290986989306,
    "left_hip": 72.21536963222839,
    "right_hip": 144.070292581427,
    "left_knee": 92.05153725645069,
    "right_knee": 139.05041786372811
  },
  "dappose": {
    "left_elbow": 165.0,
    "right_elbow": 65.0,
    "left_shoulder": 155.0,
    "right_shoulder": 65.0,
    "left_hip": 165.0,
    "right_hip": 165.0,
    "left_knee": 170.0,
    "right_knee": 170.0
  },
  "fun": {
    "left_elbow": 165.53461940082286,
    "right_elbow": 170.93556264361723,
    "left_shoulder": 130.78454911796442,
    "right_shoulder": 146.56226765385057,
    "left_hip": 162.05029717515959,
    "right_hip": 151.59248101953904,
    "left_knee": 65.30742784940394,
    "right_knee": 168.54198884663396
  },
  "headdown": {
    "left_elbow": 71.26702432534931,
    "right_elbow": 132.47262700346351,
    "left_shoulder": 63.77248418206635,
    "right_shoulder": 134.50451179526132,
    "left_hip": 96.98715568738137,
    "right_hip": 170.5504123513156,
    "left_knee": 109.2897652956211,
    "right_knee": 170.59208963845185
  },
  "kick": {
    "left_elbow": 111.75842944449928,
    "right_elbow": 173.853064088997,
    "left_shoulder": 126.37545661931085,
    "right_shoulder": 101.00989627977395,
    "left_hip": 126.47531996962424,
    "right_hip": 111.29041384260881,
    "left_knee": 88.90633928170024,
    "right_knee": 77.4305553674189
  },
  "michael": {
    "left_elbow": 87.50228368818716,
    "right_elbow": 141.985736814924,
    "left_shoulder": 82.01426824058763,
    "right_shoulder": 163.21293934479132,
    "left_hip": 117.6957315961437,
    "right_hip": 61.75422915130014,
    "left_knee": 127.58796097149451,
    "right_knee": 104.44814781034427
  },
  "michael2": {
    "left_elbow": 165.39732725293902,
    "right_elbow": 60.92985401078082,
    "left_shoulder": 71.52268204233579,
    "right_shoulder": 119.86249663920412,
    "left_hip": 141.35203874455672,
    "right_hip": 71.1092336940782,
    "left_knee": 78.88547866460391,
    "right_knee": 66.64844081218236
  },
  "shahrukh": {
    "left_elbow": 92.61670057651014,
    "right_elbow": 117.41732270622835,
    "left_shoulder": 73.13644323126094,
    "right_shoulder": 110.28357419694976,
    "left_hip": 74.02013903663828,
    "right_hip": 161.39142495398116,
    "left_knee": 95.2264615596425,
    "right_knee": 174.22280283490824
  },
  "showthem": {
    "left_elbow": 114.35363709976772,
    "right_elbow": 72.16823699662574,
    "left_shoulder": 107.51340917237607,
    "right_shoulder": 147.17842793333938,
    "left_hip": 159.9375551702886,
    "right_hip": 119.04830792668614,
    "left_knee": 96.36020058857102,
    "right_knee": 72.75186085479541
  },
  "squatpose": {
    "left_elbow": 165.0,
    "right_elbow": 165.0,
    "left_shoulder": 110.0,
    "right_shoulder": 110.0,
    "left_hip": 85.0,
    "right_hip": 85.0,
    "left_knee": 85.0,
    "right_knee": 85.0
  },
  "stylish": {
    "left_elbow": 84.46151629992126,
    "right_elbow": 151.1085151677438,
    "left_shoulder": 128.4570971438538,
    "right_shoulder": 103.79726588081118,
    "left_hip": 163.5951413408822,
    "right_hip": 129.5217633887679,
    "left_knee": 152.28632043315298,
    "right_knee": 109.42586305722867
  },
  "treepose1": {
    "left_elbow": 165.0,
    "right_elbow": 165.0,
    "left_shoulder": 140.0,
    "right_shoulder": 140.0,
    "left_hip": 105.0,
    "right_hip": 170.0,
    "left_knee": 90.0,
    "right_knee": 175.0
  }
}



POSE_TARGETS_FILE = "pose_targets.json"


def _format_pose_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _load_pose_targets_config(config_path: Path) -> Dict[str, Dict[str, float]]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(name).lower(): {joint: float(value) for joint, value in angles.items()} for name, angles in data.items()}


def _fallback_target(name: str) -> Dict[str, float]:
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return {joint: rng.uniform(60.0, 175.0) for joint in JOINT_TRIPLETS.keys()}


def _image_target_angles(image: Optional[np.ndarray], detector: Optional[PoseDetector]) -> Optional[Dict[str, float]]:
    if image is None or detector is None:
        return None
    landmarks, _ = detector.detect(image)
    angles = extract_joint_angles(landmarks)
    return angles if angles else None


def load_pose_references(poses_dir: str = "poses") -> List[PoseReference]:
    directory = Path(poses_dir)
    config_targets = _load_pose_targets_config(directory.parent / POSE_TARGETS_FILE)
    files = []
    if directory.exists():
        files.extend(directory.glob("*.webp"))
        files.extend(directory.glob("*.avif"))
        files.extend(directory.glob("*.png"))
        files.extend(directory.glob("*.jpg"))
        files.extend(directory.glob("*.jpeg"))
    files = sorted(files)

    references: List[PoseReference] = []
    detector = PoseDetector()
    try:
        for p in files:
            name = p.stem.lower()
            image = cv2.imread(str(p))
            if image is None:
                continue

            target_angles = config_targets.get(name)
            if target_angles is None:
                image_angles = _image_target_angles(image, detector)
                if image_angles:
                    target_angles = image_angles
                else:
                    target_angles = DEFAULT_POSE_TARGETS.get(name, _fallback_target(name))

            references.append(
                PoseReference(
                    name=name,
                    display_name=_format_pose_name(name),
                    image_path=str(p),
                    image=image,
                    target_angles=target_angles,
                )
            )
    finally:
        detector.close()

    if references:
        return references

    # If no assets found, still provide playable rounds with placeholders.
    for default_name in DEFAULT_POSE_TARGETS.keys():
        references.append(
            PoseReference(
                name=default_name,
                display_name=_format_pose_name(default_name),
                image_path=None,
                image=None,
                target_angles=DEFAULT_POSE_TARGETS[default_name],
            )
        )
    return references


class PoseBattleGame:
    def __init__(self, all_poses: List[PoseReference], poses_per_round: int = 5):
        if not all_poses:
            raise ValueError("No poses available.")
        self.all_poses = all_poses
        self.poses_per_round = poses_per_round
        self.reset()

    def reset(self):
        if len(self.all_poses) >= self.poses_per_round:
            self.round_poses = random.sample(self.all_poses, self.poses_per_round)
        else:
            self.round_poses = [random.choice(self.all_poses) for _ in range(self.poses_per_round)]

        self.current_round = 0
        self.player_totals = {"p1": 0.0, "p2": 0.0}
        self.last_round_scores = {"p1": 0.0, "p2": 0.0}

    def get_current_pose(self) -> Optional[PoseReference]:
        if self.current_round >= self.poses_per_round:
            return None
        return self.round_poses[self.current_round]

    def register_round_scores(self, p1_score: float, p2_score: float):
        self.last_round_scores["p1"] = p1_score
        self.last_round_scores["p2"] = p2_score
        self.player_totals["p1"] += p1_score
        self.player_totals["p2"] += p2_score
        self.current_round += 1

    def is_finished(self) -> bool:
        return self.current_round >= self.poses_per_round

    def winner_text(self) -> str:
        p1 = self.player_totals["p1"]
        p2 = self.player_totals["p2"]
        if abs(p1 - p2) < 1e-6:
            return f"It's a Tie!  P1: {p1:.1f}  P2: {p2:.1f}"
        winner = "Player 1" if p1 > p2 else "Player 2"
        return f"{winner} Wins!  P1: {p1:.1f}  P2: {p2:.1f}"

    def round_label(self) -> str:
        return f"Round {min(self.current_round + 1, self.poses_per_round)}/{self.poses_per_round}"
