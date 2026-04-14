<div align="center">

# 🤸 POSE AI — Real-Time Pose Battle Game

**Strike the pose. Beat your opponent. Win the round.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **two-player pose matching game** powered by MediaPipe's BlazePose and OpenCV. Players compete side-by-side in front of a single webcam, striking target poses as fast and accurately as they can. An AI judge scores each round in real time based on **joint-angle similarity**.

---

</div>

## ✨ Features

| Feature | Description |
|---|---|
| 🎮 **Two-Player Split-Screen** | Single webcam divided into two halves — Player 1 (left) & Player 2 (right) |
| 🦴 **Real-Time Skeleton Tracking** | Full-body pose estimation via MediaPipe BlazePose (33 landmarks) |
| 📐 **Joint-Angle Scoring** | 8 key joint angles compared against target poses for precise scoring |
| 🎯 **20+ Target Poses** | Curated library including dance, yoga, bodybuilding & fun poses |
| 🖥️ **Responsive HUD** | Modern gradient UI with live scores, round info & performance feedback |
| 🏆 **Performance Feedback** | Instant ratings — *"Perfect Pose!"*, *"Almost There!"*, or *"What was that pose?"* |
| 🔄 **Replay System** | Instantly restart a new game after the winner is announced |

---

## 🎬 How It Works

```
┌─────────────────────────────────────────────────────┐
│                    WEBCAM FEED                       │
│  ┌──────────────┐          ┌──────────────┐         │  ┌────────────┐
│  │  PLAYER 1    │          │  PLAYER 2    │         │  │  TARGET    │
│  │  (Left Half) │          │ (Right Half) │         │  │   POSE     │
│  │              │          │              │         │  │            │
│  │  🦴 Skeleton │          │  🦴 Skeleton │         │  │   🖼️ Image │
│  │   Overlay    │          │   Overlay    │         │  │            │
│  └──────────────┘          └──────────────┘         │  └────────────┘
│              ┌──────────────────────┐                │
│              │    SCORE HUD BAR     │                │
│              │  P1: 87.3  P2: 72.1  │                │
│              │   Round 3/5          │                │
│              └──────────────────────┘                │
└─────────────────────────────────────────────────────┘
```

1. **Countdown** — Each round starts with a 5-second timer
2. **Pose** — When the timer hits zero, your current body pose is captured
3. **Score** — Joint angles are extracted and compared to the target pose
4. **Repeat** — 5 rounds per game, highest total score wins

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** (recommended — best MediaPipe compatibility)
- A working **webcam**
- Windows / macOS / Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/VenkataSatyaSai16/POSE-AI.git
cd POSE-AI

# Create a virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Game

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| <kbd>Space</kbd> | Start the game from the title screen |
| <kbd>R</kbd> | Restart after game over |
| <kbd>Q</kbd> | Quit the game |

---

## 📁 Project Structure

```
POSE-AI/
├── main.py              # 🎬 Entry point — game loop & state machine
├── camera.py            # 📷 Webcam capture manager
├── pose_detector.py     # 🦴 MediaPipe BlazePose wrapper
├── pose_compare.py      # 📐 Joint angle extraction & scoring engine
├── game_logic.py        # 🎮 Round management, scoring & pose loading
├── ui.py                # 🖥️ HUD, overlays & responsive UI rendering
├── pose_targets.json    # 🎯 Pre-computed target angles per pose
├── test.py              # 🧪 Test script
├── requirements.txt     # 📦 Python dependencies
└── poses/               # 🖼️ Target pose image library
    ├── Bharatanatyam.jpeg
    ├── bodybuilderpose.jpg
    ├── treepose1.jpeg
    ├── michael.webp
    ├── shahrukh.webp
    └── ... (20+ poses)
```

---

## 🧠 How Scoring Works

The game compares **8 key joint angles** between the player's pose and the target:

```
Left Elbow  ─┐                    ┌─ Right Elbow
Left Shoulder─┤   🦴 Body Model   ├─ Right Shoulder
Left Hip     ─┤                    ├─ Right Hip
Left Knee    ─┘                    └─ Right Knee
```

**Formula:**
```
Score = 100 - mean(|player_angle - target_angle| for each joint)
```

| Score Range | Feedback |
|---|---|
| **85 – 100** | ⭐ *Perfect Pose!* |
| **60 – 84** | 👍 *Almost There!* |
| **0 – 59** | 😅 *What was that pose?* |

---

## 🖼️ Pose Library

The game ships with **20+ diverse poses** spanning multiple categories:

| Category | Poses |
|---|---|
| 💃 **Dance** | Bharatanatyam, Classical, Dance Pose 1 & 2 |
| 🧘 **Yoga** | Tree Pose, Child Pose, Head Down |
| 💪 **Fitness** | Bodybuilder Pose, Squat Pose |
| 🕺 **Pop Culture** | Michael (Jackson), Shahrukh (Khan), Dab Pose |
| 🎉 **Fun** | Fun, Kick, Stylish, Show Them |

> **Adding your own poses:** Drop any image (`.jpg`, `.jpeg`, `.png`, `.webp`, `.avif`) into the `poses/` folder. The game will automatically detect the pose in the image and use it as a target. For fine-tuned control, add custom angles in `pose_targets.json`.

---

## ⚙️ Configuration

| Setting | Location | Default |
|---|---|---|
| Rounds per game | `main.py` → `TOTAL_ROUNDS` | `5` |
| Countdown duration | `main.py` → `COUNTDOWN_SECONDS` | `5.0s` |
| Result display time | `main.py` → `RESULT_SHOW_SECONDS` | `2.5s` |
| Window size | `main.py` → `cv2.resizeWindow()` | `1280×720` |
| Camera index | `camera.py` → `CameraManager()` | `0` |
| Pose detection confidence | `pose_detector.py` | `0.5` |

---

## 🛠️ Tech Stack

- **[MediaPipe](https://mediapipe.dev/)** — Google's ML framework for real-time pose estimation (BlazePose, 33 landmarks)
- **[OpenCV](https://opencv.org/)** — Computer vision library for camera capture, image processing & UI rendering
- **[NumPy](https://numpy.org/)** — Efficient numerical computation for angle calculations

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- 🌐 Add a web-based UI with Flask/Streamlit
- 📱 Mobile support via MediaPipe's lightweight models
- 🎵 Background music & sound effects
- 📊 Score history & leaderboard persistence
- 🤖 Single-player mode with AI opponent
- 🎨 Custom pose creation tool

```bash
# Fork & clone
git clone https://github.com/<your-username>/POSE-AI.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Commit & push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Open a Pull Request 🎉
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ and AI**

⭐ Star this repo if you like it!

</div>
