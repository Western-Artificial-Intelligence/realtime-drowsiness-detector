📘 Overview

Drowsy and distracted driving are leading causes of accidents worldwide.
This project aims to enhance driver safety by using computer vision to:

Detect drowsiness based on eye closure, blinking rate, and yawning.

Detect distraction through head pose and gaze tracking (optional phone detection).

Alert the driver with on-screen and audio cues to re-engage attention.

🧠 Core Features

- Real-time detection of drowsiness and distraction

- Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) analysis

- Head pose estimation for gaze direction

- Visual and audio alert system

- Fully on-device processing for user privacy

- Optional YOLO-based phone detection

🧰 Tech Stack
Computer Vision:	OpenCV, MediaPipe
ML Models:	PyTorch, YOLOv8, ONNX Runtime (potentially)
Frontend/UI: Streamlit (most likely)


⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/Western-Artificial-Intelligence/realtime-drowsiness-detector.git
cd wai-drowsy


📂 Repository Structure (once set up)

```bash
wai-drowsy/
├── data/                   # Sample demo clips
├── models/                 # YOLO/ONNX models (ignored in git)
├── src/
│   ├── wai/
│   │   ├── camera.py        # Frame capture
│   │   ├── landmarks.py     # Facial landmarks via MediaPipe
│   │   ├── signals.py       # EAR/MAR/head pose logic
│   │   ├── fusion.py        # Combines multiple signals
│   │   ├── alerts.py        # Visual/audio alerts
│   │   └── ui.py            # Streamlit interface
├── tests/                  # Basic smoke tests
├── requirements.txt
├── README.md
└── LICENSE
```

