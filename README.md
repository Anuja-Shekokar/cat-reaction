🐱 Cat Reaction

A fun real-time computer vision project that makes a cat react to your hand gestures using your laptop camera.

🎥 Demo

https://youtube.com/shorts/S--ZBGXXfoo?si=gk5viH9vjxMlJcno

👋 Gestures
Gesture	Reaction
☝️ Pointing Cat
🤟 Shaka Cat
🤫 Shush Cat
👍 Thumbs-Up Cat
✋ Default Cat

🛠️ Tech Stack
Python
OpenCV
MediaPipe
NumPy
Matplotlib

⚙️ How It Works
Camera captures your hand.
MediaPipe detects hand landmarks.
The program recognizes the gesture.
The matching cat reaction appears in a separate window.

🚀 How to Run
1. Clone the repository
git clone https://github.com/Anuja-Shekokar/cat-reaction.git
cd cat-reaction
2. Install dependencies
pip install opencv-python mediapipe numpy matplotlib
3. Run the project
python main.py

Allow camera access if prompted.

📁 Project Structure
cat-reaction/
├── main.py
├── hand_landmarker.task
└── cats/
    ├── default.png
    ├── point.png
    ├── shaka.png
    ├── shush.png
    └── thumbs_up.png
    
👩‍💻 Author

Anuja Shekokar 🐱

Made with Python, computer vision, hand gestures, and an unreasonable amount of cats.
