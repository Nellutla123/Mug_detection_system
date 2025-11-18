import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================

# Change this if your best.pt is in a different location
MODEL_PATH = r"runs/detect/mug_yolo11n_results/weights/best.pt"
# Example absolute path if needed:
# MODEL_PATH = r"D:\mug_detection_project\runs\detect\mug_yolo11n_results\weights\best.pt"


# ==========================
# LOAD MODEL (CACHED)
# ==========================

@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model once and cache it."""
    model = YOLO(model_path)
    return model


# ==========================
# STREAMLIT PAGE SETUP
# ==========================

st.set_page_config(page_title="Mug Detection - YOLO11", layout="wide")

st.title("☕ Mug Detection using YOLO11")
st.write(
    "Upload an **image** or **video**, and this app will run your trained "
    "YOLO11 model to detect mugs."
)

# Load model
with st.spinner("Loading YOLO11 model..."):
    model = load_model(MODEL_PATH)

# Sidebar
st.sidebar.header("Settings")
task_type = st.sidebar.radio("Choose input type:", ["Image", "Video"])
conf_threshold = st.sidebar.slider(
    "Confidence threshold", 0.1, 1.0, 0.5, 0.05,
    help="Higher = fewer but more confident detections"
)


# ==========================
# IMAGE INFERENCE
# ==========================

if task_type == "Image":
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        # Read image
        image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        # Run YOLO prediction
        with st.spinner("Running detection on image..."):
            results = model.predict(
                image,
                conf=conf_threshold,
                verbose=False
            )

        # Draw boxes on image
        result_image = results[0].plot()  # BGR numpy array
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detection Result")
            st.image(result_image, use_column_width=True)

        # Show detection info
        st.subheader("Detections")
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.write("No objects detected.")
        else:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
                st.write(
                    f"**#{i+1}** - Class ID: `{cls_id}`, "
                    f"Confidence: `{conf:.2f}`, Box: `{xyxy}`"
                )


# ==========================
# VIDEO INFERENCE
# ==========================

else:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        # Save uploaded video to a temp file (keep original extension)
        suffix = Path(uploaded_video.name).suffix
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_video.read())
        video_input_path = tfile.name

        st.subheader("Original Video")
        st.video(video_input_path)

        # Run YOLO prediction on video
        with st.spinner("Running detection on video... this may take some time..."):
            results = model.predict(
                source=video_input_path,
                conf=conf_threshold,
                save=True,
                project="streamlit_outputs",  # output folder in your project
                name="mug_video",             # subfolder for outputs
                exist_ok=True,
                verbose=False,
                imgsz=480,        # smaller size for speed
                vid_stride=3      # process every 3rd frame
            )

        # Get the folder where YOLO saved results
        save_dir = Path(results[0].save_dir)

        # Debug (optional – helps you see where things are saved)
        st.write(f"Results saved in: `{save_dir}`")

        pred_video_path = None
        video_extensions = (".mp4", ".avi", ".mov", ".mkv")

        # Find a video file in that folder
        for file in os.listdir(save_dir):
            if file.lower().endswith(video_extensions):
                pred_video_path = save_dir / file
                break

        if pred_video_path and pred_video_path.exists():
            st.subheader("🎯 Detection Result Video")

            # Read as bytes and send to st.video()
            with open(pred_video_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            st.success(f"Processed video saved at: `{pred_video_path}`")

            # Optional: Download button
            with open(pred_video_path, "rb") as vid_file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=vid_file,
                    file_name="mug_detection_output.mp4",
                    mime="video/mp4"
                )
        else:
            st.error(
                "⚠️ Could not find the processed video file. "
                "Check the 'streamlit_outputs/mug_video' folder."
            )

