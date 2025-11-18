


# ☕ Mug Detection System (YOLO11 + Streamlit)

A simple **mug detection system** built using **YOLO11 (Ultralytics)** and a **Streamlit** web app.  
The project demonstrates:

- Custom data collection from videos  
- Dataset creation & annotation using **Roboflow**  
- Training a **YOLO11** model on the custom dataset  
- Running **image & video mug detection** via a web interface  

---

## 🧰 Tech Stack & Tools

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Ultralytics-YOLO11-0A1A2F?logo=ultralytics&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Roboflow-Dataset%20&%20Labeling-111111?logo=roboflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Pexels-Video%20Source-05A081?logo=pexels&logoColor=white" />
</p>

---

## 📂 Project Structure
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
# Clone repository
git clone https://github.com/Nellutla123/Mug_detection_system.git
cd Mug_detection_system

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)

# Install dependencies
pip install -r requirements.txt

---
 📌 Project Workflow

## 🧬 1. Data Collection & Dataset Creation

### 1.1 Raw Video Source (Pexels)

I used free videos from **[Pexels](https://www.pexels.com/)** that show people **holding or using mugs**.

You can find similar videos here:

- Pexels search – coffee mug videos:  
  👉 https://www.pexels.com/search/videos/coffee%20mug/

Example type of clip used for this project:

- Person holding a coffee mug:  
  👉 https://www.pexels.com/video/person-holding-a-coffee-mug-7986492/

These videos were downloaded and later used to generate frames.

---

### 1.2 Converting Videos to Frames

To train YOLO, I first converted the downloaded videos into **image frames** (JPEGs).  
(You can use your own script or `extract_frames.py` / a simple OpenCV snippet like below.)

Example frame extraction logic:

```python
import cv2
import os

video_path = "input_videos/sample_mug_video.mp4"
output_dir = "data/raw_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
frame_interval = 5  # save every 5th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, frame)

    frame_idx += 1

cap.release()
print("Frames saved to:", output_dir)


🏷️ 3. Annotation & Dataset Creation (Roboflow)

Uploaded extracted frames to Roboflow

Used Auto Label + Manual correction

Roboflow auto-split data into train, valid, test

Exported in YOLO11 format


🎯 4. Training YOLO11 Model
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

runs/detect/mug_model_yolo11/
└── weights/
    ├── best.pt   ← used for inference

🧪 5. Streamlit Web App Demo (app.py)

Run this file using:

streamlit run app.py


✔ Upload image or video
✔ Streamlit shows detection results live
✔ Saves detection video to streamlit_outputs/




