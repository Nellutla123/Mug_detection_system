from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model = model.train( data = r"D:\mug_detection_project\Mug_object_detection-1\data.yaml",
                    epochs = 10,
                    imgsz = 640,
                    batch = 8,
                    name = "mug_yolo11n_results")