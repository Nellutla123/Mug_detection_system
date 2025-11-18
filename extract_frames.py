import cv2
import os
import glob

VIDEO_DIR = "videos"
OUTPUT_DIR = "frames"

FRAME_EVERY_N = 10

os.makedirs(OUTPUT_DIR,exist_ok=True)

video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))

for video_path in video_paths:
    print(f"processing:{video_path}")
    cap = cv2.VideoCapture(video_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    frame_idx = 0
    saved_count = 0
    
    
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        
        if frame_idx % FRAME_EVERY_N == 0:
            frame_filename = f"{video_name}_frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            cv2.imwrite(frame_path,frame)
            saved_count += 1
        
        frame_idx += 1
    cap.release()
    
    print(f"saved {saved_count} frames for{video_name}")

print("done extracting frames")
    

