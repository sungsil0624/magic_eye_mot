import numpy as np
import supervision as sv
from ultralytics import YOLO
from deepface import DeepFace
import cv2

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
trace_annotator = sv.TraceAnnotator()

# 객체 체류시간을 저장할 딕셔너리 초기화
stay_duration = {}

# 얼굴 정보를 저장할 딕셔너리 초기화
faces_info = {}

# 프레임 레이트
frame_rate = 24.0


def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    results = model(frame, tracker="botsort.yaml")[0]

    # detect
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # person만 detect
    detections = tracker.update_with_detections(detections)

    people_count = sum(1 for class_id in detections.class_id)

    # 이전 프레임의 객체 ID
    previous_ids = set(stay_duration.keys())

    # 현재 프레임의 객체 ID
    current_ids = set(detections.tracker_id)

    # 등장한 객체 ID 기록
    new_appearances = current_ids - previous_ids

    # 사라진 객체 ID 기록
    disappeared_objects = previous_ids - current_ids

    # 등장한 객체에 대해서 현재 프레임 인덱스 기록
    for tracker_id in new_appearances:
        stay_duration[tracker_id] = frame_index

    # 사라진 객체에 대해서 체류시간 계산 및 출력
    for tracker_id in disappeared_objects:
        start_frame = stay_duration.pop(tracker_id)
        end_frame = frame_index
        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / frame_rate
        print(f"Object #{tracker_id} stayed for {duration_seconds:.2f} seconds.")

    # 얼굴 인식 부분 추가
    img_with_faces = frame.copy()
    for (x, y, x2, y2), tracker_id in zip(detections.xyxy, detections.tracker_id):
        # 얼굴 정보를 이미 저장한 객체인지 확인
        if tracker_id not in faces_info:
            face_crop = frame[int(y):int(y2), int(x):int(x2)]
            # 얼굴 인식
            try:
                result = DeepFace.analyze(face_crop, actions=['age', 'gender'])
                age = result[0]["age"]
                gender = result[0]["gender"]
                # 얼굴 정보 저장
                faces_info[tracker_id] = f"Age: {age}, Gender: {gender}"
            except:
                pass

        # 얼굴 정보 표시
        if tracker_id in faces_info:
            cv2.putText(img_with_faces, faces_info[tracker_id], (int(x), int(y) - 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # 콘솔에 프레임 별 사람 수 출력
    print(f"Frame {frame_index}: People Count - {people_count}")

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # annotate
    annotated_frame = box_annotator.annotate(
        scene=img_with_faces,
        detections=detections,
        labels=labels
    )
    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

    return trace_annotator.annotate(
        annotated_frame, detections=detections)


sv.process_video(
    source_path="assets/example1.mp4",
    target_path="result.mp4",
    callback=callback
)
