# 🐍 Hand-Controlled Snake Game

A computer vision-powered Snake game built with **Python**, **OpenCV**, and **MediaPipe**. Instead of using your keyboard, you control the snake's direction in real-time by moving your index finger in front of your webcam.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-007f00?style=for-the-badge&logo=google&logoColor=white)

---

## 🚀 Features
- **Gesture Control**: Intuitive hand-tracking to guide the snake.
- **Scoring System**: Real-time score tracking and persistent High Score for the session.
- **Visual Feedback**: Red hand-landmark overlays to show what the AI is tracking.
- **Responsive Gameplay**: Optimized for 8 FPS to balance control and difficulty.

## 🛠️ Tech Stack
| Library | Purpose |
| :--- | :--- |
| **MediaPipe** | AI Hand Landmark detection (Tracking index finger). |
| **Pygame** | Game engine for rendering the snake, food, and UI. |
| **OpenCV** | Capturing and processing the webcam video stream. |
| **NumPy** | Managing coordinate arrays for the snake body. |

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Jemelimercy/Hand-Controlled-Snake-.git](https://github.com/Jemelimercy/Hand-Controlled-Snake-.git)
   cd Hand-Controlled-Snake-

   Set up a Virtual Environment (Optional but Recommended):
   python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

Install Dependencies:

pip install opencv-python mediapipe pygame numpy

Run the Game:
python snake_game.py

🎮 How to Play
Calibration: Ensure your hand is visible to the webcam. You will see red dots on your knuckles.

Movement: Move your Index Finger Tip (Landmark 8) to steer.

Move finger Up/Down/Left/Right relative to the snake's head to change direction.

Objective: Eat the red food squares to grow and increase your score.

Game Over: Hitting the wall or your own tail will reset your score (but your High Score remains!).

Quit: Press 'Q' or close the window to exit.



🌟 Acknowledgments
Inspired by the classic Dino and Snake games.

Powered by Google's MediaPipe Hand Tracking.