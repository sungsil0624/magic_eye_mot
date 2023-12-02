import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

# line settings
LINE_START = sv.Point(466, 360)
LINE_END = sv.Point(546, 624)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

# annotator 초기화
line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)
# label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# 프레임 별 사람 수 출력 예시
# 우선 로컬 영상에서 적용

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    results = model(frame, tracker="botsort.yaml")[0]

    # detect
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0] # person만 detect
    detections = tracker.update_with_detections(detections)

    people_count = sum(1 for class_id in detections.class_id)

    ## 감지된 객체의 x,y 값 뽑아내기
    ## 사람만 감지될 수 있도록 코드도 변경해야할듯
    # for bbox in detections.xyxy:
    #     x_center = (bbox[0] + bbox[2]) / 2
    #     y_center = (bbox[1] + bbox[3]) / 2
    #     print(f"객체 - x: {x_center}, y: {y_center}")

    # 콘솔에 프레임 별 사람 수 출력
    print(f"Frame {frame_index}: People Count - {people_count}")

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # annotate
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels
    )
    # annotated_frame = label_annotator.annotate(
    #     annotated_frame, detections=detections, labels=labels)
    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

    return trace_annotator.annotate(
        annotated_frame, detections=detections)


sv.process_video(
    source_path="assets/people-walking.mp4",
    target_path="result.mp4",
    callback=callback
)