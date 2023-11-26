import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()


# 프레임 별 사람 수 출력 예시
# 우선 로컬 영상에서 적용

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    people_count = sum(1 for class_id in detections.class_id if results.names[class_id] == "person")

    # 콘솔에 프레임 별 사람 수 출력
    print(f"Frame {frame_index}: People Count - {people_count}")

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

    return trace_annotator.annotate(
        annotated_frame, detections=detections)



sv.process_video(
    source_path="assets/people-walking.mp4",
    target_path="result.mp4",
    callback=callback
)