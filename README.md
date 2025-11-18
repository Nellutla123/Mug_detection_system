# ☕ Mug Detection System (YOLO + Streamlit)

A complete Mug Detection system using **YOLO11 (Ultralytics)** for detection and **Streamlit** for live web-based demo.  
This project demonstrates real-time image/video mug detection using a custom-trained model.



<img width="1920" height="1080" alt="Screenshot (45)" src="https://github.com/user-attachments/assets/4233cc95-d274-46c6-9a29-902a4024e2da" />




<img width="1920" height="1080" alt="Screenshot (47)" src="https://github.com/user-attachments/assets/e09f151c-4c6b-4022-b13e-8ab808d30beb" />



<img width="1920" height="1080" alt="Screenshot (48)" src="https://github.com/user-attachments/assets/1930c958-0507-481c-954d-5d2e249bab92" />

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







