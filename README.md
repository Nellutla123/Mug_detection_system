Mug Detection System

A small object-detection project for detecting mugs (cup-like objects).
This repository contains training and inference scripts (YOLOv11-style naming), utilities to extract frames from videos, a Roboflow notebook for dataset operations, and a minimal app for serving predictions. Files in this repo include app.py, train_mug_yolo11.py, predict_mug_yolo11.py, extract_frames.py, result.py, and roboflow.ipynb. 
GitHub

Table of contents

About

Features

Repo structure

Requirements

Installation

Usage

1) Prepare dataset (Roboflow / manual)

2) Extract frames from video

3) Train model

4) Run inference / predict

5) Run the app (serve predictions)

Examples

Configuration

Troubleshooting

Contributing

License

About

This project implements an object detection pipeline focused on detecting mugs. It includes scripts to extract frames from videos, train a detection model (files named train_mug_yolo11.py suggest a YOLO-style training pipeline), make predictions (predict_mug_yolo11.py), and serve predictions through a small app (app.py). A Jupyter notebook roboflow.ipynb is included to help with dataset import/export and labeling workflows. 
GitHub
+1

Features

Frame extraction from video files for dataset creation (extract_frames.py). 
GitHub

Training script for a YOLO-style object detector (train_mug_yolo11.py). 
GitHub

Prediction/inference script to run detection on images/video (predict_mug_yolo11.py). 
GitHub

Minimal serving app (likely Flask or FastAPI) in app.py to accept images and return detection results. 
GitHub

Notebook for Roboflow dataset operations and walkthrough (roboflow.ipynb). 
GitHub

Repo structure (what I observed)
Mug_detection_system/
├─ app.py
├─ extract_frames.py
├─ predict_mug_yolo11.py
├─ train_mug_yolo11.py
├─ result.py
├─ roboflow.ipynb
├─ requirements.txt
└─ .gitignore


If any file names differ or you add scripts, adjust this section accordingly. 
GitHub

Requirements

A requirements.txt is present in the repo. Typical dependencies for this kind of project include:

Python 3.8+

PyTorch (or other DL framework if used)

OpenCV (opencv-python) for frame extraction and image I/O

torchvision / albumentations (optional, for transforms)

Flask or FastAPI (if app.py is a web app)

Roboflow client (if the notebook uses Roboflow)

Install from the included requirements.txt:

python -m venv venv
source venv/bin/activate      # Linux / macOS
# .\venv\Scripts\activate     # Windows PowerShell

pip install -r requirements.txt


(If requirements.txt is missing packages you need, add them — e.g. torch, torchvision, opencv-python, flask, roboflow.) 
GitHub

Installation

Clone the repo:

git clone https://github.com/Nellutla123/Mug_detection_system.git
cd Mug_detection_system


Create and activate a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Prepare dataset (see next section).

Usage

The commands below are templated based on typical scripts named like the ones in this repo. Replace flags/arguments with the exact ones used in your script if they differ.

Prepare dataset (Roboflow / manual)

If you use Roboflow, open roboflow.ipynb and follow the notebook cells to import/export your dataset and download in YOLO format. Otherwise, prepare dataset in YOLO format with images and annotations in labels/. 
GitHub

Extract frames from a video

extract_frames.py likely extracts frames from a video for dataset creation.

Example:

python extract_frames.py --video path/to/video.mp4 --output data/frames --fps 1


(Adjust CLI flags to the ones implemented in your script.)

Train the model

Run the training script:

python train_mug_yolo11.py --data data/mug_dataset.yaml --cfg yolovX.cfg --epochs 50 --batch-size 8


Replace --cfg, --data, and other flags with the parameters your script expects. The training script name suggests YOLO-style training; verify that it uses PyTorch/Ultralytics or another implementation. 
GitHub

Run inference / predict

Use the prediction script to run detection on an image, folder, or video:

python predict_mug_yolo11.py --source path/to/image_or_video --weights runs/exp/weights/best.pt --conf 0.25


Outputs will typically be saved to a runs/predict folder or printed to stdout. Check result.py if that script handles formatting or post-processing. 
GitHub

Run the app (serve predictions)

If app.py is a Flask/FastAPI server, start it like:

python app.py
# or
FLASK_APP=app.py flask run


Then open http://127.0.0.1:5000 (or the printed host/port) and use the provided web endpoint(s) to upload images and receive JSON detection results. Inspect app.py to confirm routes and payload format. 
GitHub

Examples

Detect a mug in an image

python predict_mug_yolo11.py --source examples/mug.jpg --weights runs/exp/weights/best.pt
# Output (image with bounding boxes) saved to runs/predict/...


Extract frames

python extract_frames.py --video sample_video.mp4 --output frames/ --start 0 --end 60 --fps 1


Train

python train_mug_yolo11.py --epochs 100 --batch-size 16 --data data/mug_data.yaml


(Adjust to exact flags in your scripts.)

Configuration

requirements.txt — install dependencies from here. 
GitHub

data/*.yaml — expected dataset config file (if following YOLO conventions). If not present, create one listing train/val image directories and class names.

runs/ — typical output directory for training/prediction results (weights, logs, prediction images).

Add a config.example.yaml or document exact config variables if your scripts expect specific environment variables or file names.

Troubleshooting

CUDA / PyTorch mismatch: ensure your torch build matches your CUDA version. If training fails with CUDA errors, either install CPU-only or a torch build that matches your GPU drivers.

Missing packages: ModuleNotFoundError → add the missing package to requirements.txt and reinstall.

App not starting: inspect app.py for required environment variables (e.g., model weights path). Run python app.py from a terminal to see tracebacks.

Models not found: ensure --weights path passed to predict_* scripts points to an existing .pt file created by training.





