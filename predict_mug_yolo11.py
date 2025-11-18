from ultralytics import YOLO

model = YOLO("runs/detect/mug_yolo11n_results/weights/best.pt")



model.predict(
    source=r"D:\mug_detection_project\videos\4107562-uhd_3840_2160_25fps.mp4",
    save=True,
    conf=0.5
)
