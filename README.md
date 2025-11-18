# ☕ Mug Detection System (YOLO + Streamlit)

A complete Mug Detection system using **YOLO11 (Ultralytics)** for detection and **Streamlit** for live web-based demo.  
This project demonstrates real-time image/video mug detection using a custom-trained model.

---

## 🚀 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Programming language |
| ⚡ Ultralytics YOLO11 | Object Detection Model |
| 🎨 Streamlit | Web App Interface |
| 👁️ OpenCV | Frame processing & video handling |
| 🖼️ Roboflow | Dataset creation & auto-labeling |
| 🎥 Pexels | Source of raw mug-holding videos |

---

## 📌 Project Workflow

### 📹 1. Data Collection
- Downloaded free mug-holding videos from **Pexels**  
  👉 https://www.pexels.com/search/videos/coffee%20mug/

Example video used:
👉 https://www.pexels.com/video/person-holding-a-coffee-mug-7986492/

---

### 🖼️ 2. Frame Extraction

Extracted frames from videos using OpenCV:

```python
import cv2, os

video_path = "input_video.mp4"
output_dir = "data/raw_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 5 == 0:  # Save every 5th frame
        cv2.imwrite(f"{output_dir}/frame_{frame_idx}.jpg", frame)
    frame_idx += 1

cap.release()
print("Frames saved!")
🏷️ 3. Annotation & Dataset Creation (Roboflow)
Uploaded extracted frames to Roboflow

Used Auto Label + Manual correction

Roboflow auto-split data into train, valid, test

Exported in YOLO11 format

📥 Roboflow dataset download code:

python
Copy code
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("mug-detection")
dataset = project.version(1).download("yolov11")
🎯 4. Training YOLO11 Model
python
Copy code
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # YOLO11 nano model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs/detect",
    name="mug_model_yolo11"
)
📌 Output structure (auto-generated):

bash
Copy code
runs/detect/mug_model_yolo11/
└── weights/
    ├── best.pt   ← used for inference
🧪 5. Streamlit Web App Demo (app.py)
Run this file using:

bash
Copy code
streamlit run app.py
✔ Upload image or video
✔ Streamlit shows detection results live
✔ Saves detection video to streamlit_outputs/

📂 Project Structure
bash
Copy code
Mug_detection_system/
├── app.py                    # Streamlit detection demo app
├── video_demo.py             # Simple CLI-based video detection script
├── extract_frames.py         # Frame extraction from raw videos
├── train_mug_yolo11.py       # Model training
├── roboflow.ipynb            # Dataset download notebook
├── requirements.txt          # Dependencies
├── .gitignore
├── README.md
⚙️ Installation & Setup
bash
Copy code
# Clone repository
git clone https://github.com/Nellutla123/Mug_detection_system.git
cd Mug_detection_system

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)

# Install dependencies
pip install -r requirements.txt
▶️ How to Run
🔹 Run Web App (streamlit)
bash
Copy code
streamlit run app.py
🔹 Run CLI video detection
bash
Copy code
python video_demo.py
🎥 Test Data Source (Try These Videos!)
Type	Source
Mug-Holding Videos	https://www.pexels.com/search/videos/coffee%20mug/
Example Video	https://www.pexels.com/video/person-holding-a-coffee-mug-7986492/
Mug Images	https://www.pexels.com/search/mug/

💡 You can download and upload directly in app.py.

🛠️ Requirements (requirements.txt)
nginx
Copy code
ultralytics
streamlit
opencv-python
pillow
numpy
torch
torchvision






