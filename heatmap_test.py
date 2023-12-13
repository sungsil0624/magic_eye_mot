import supervision as sv
import cv2
import numpy as np
from utils import for_webcam as forw
import os
from datetime import datetime, timedelta
from supervision import get_video_frames_generator
from ultralytics import YOLO


model = YOLO('yolov8n.pt')
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness=1,
    text_scale=0.5
)
heatmap_annotator = sv.HeatMapAnnotator()

# 1시간 단위라면
# save_interval_minutes = 60
# 아래는 10초 단위
save_interval_seconds = 10
output_directory = 'heatmap_images/'

# 초기화
last_save_time = datetime.now()

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    global last_save_time

    results = model(frame, tracker="botsort.yaml")[0]

    # detect
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels
    )
    annotated_frame = heatmap_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    cv2.imshow('Annotated Frame', annotated_frame)

    current_time = datetime.now()
    elapsed_time = (current_time - last_save_time).total_seconds()

    if elapsed_time >= save_interval_seconds:
        forw.save_screenshot(annotated_frame, current_time)
        last_save_time = current_time
        heatmap_annotator.heat_mask = None

    return annotated_frame


forw.process_video(
    target_path="results/heatmap.mp4",
    callback=callback,
    webcam_index=0,
)
