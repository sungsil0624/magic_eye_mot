import time
import cv2
from utils import for_webcam as forw
import supervision as sv
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

faces_info = {}

# clip 이벤트
event_for_clip = False

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    # 여기에서 모델 적용 및 처리
    results = model(frame, tracker="botsort.yaml")[0]

    # detect
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # person만 detect
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # 얼굴 인식 부분
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
                if gender["Woman"] > gender["Man"]:
                    gender = "Woman"
                else:
                    gender = "Man"

                current_time = time.localtime()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', current_time)
                data = {
                    timestamp: {
                        'age': age,
                        'gender': gender
                    }
                }
                # firebase_manager.update_data('/users/aKSaa0in0dgZeQXMnFybODKSMdp1/people_info', data)
                print(age)
                print(gender)
            except:
                pass

        # 얼굴 정보 표시
        if tracker_id in faces_info:
            cv2.putText(img_with_faces, faces_info[tracker_id], (int(x), int(y) - 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    annotated_frame = box_annotator.annotate(
        scene=img_with_faces,
        detections=detections,
        labels=labels
    )

    # annotate한 화면
    cv2.imshow('Annotated Frame', annotated_frame)

    return annotated_frame

forw.process_video(
    target_path="output_webcam_deepface.mp4",  # You can change the file format based on the codec used
    callback=callback,
    webcam_index=0,  # Adjust this index based on the webcam you want to use
)
