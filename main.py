import time
import supervision as sv
from ultralytics import YOLO
from deepface import DeepFace
import cv2
import os
from dotenv import load_dotenv
from firebase.firebase_manager import FireBaseManager
from rtsp.rtsp_steram import RTSPVideoCapture

load_dotenv()

credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
database_url = os.getenv("FIREBASE_DATABASE_URL")
bucket_name = os.getenv("FIRESTORE_BUCKET_NAME")

firebase_manager = FireBaseManager(credentials_path, database_url, bucket_name)

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

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

# RTSP URL
rtsp_url = "rtsp://admin:1q2w3e4r!@192.9.200.142:554/ch04/0"
rtsp_capture = RTSPVideoCapture(rtsp_url)

try:
    while True:
        frame = rtsp_capture.read_frame()

        if rtsp_capture.get_frame_count() % 20 == 0:

            if frame is not None:
                frame_index = rtsp_capture.get_frame_count()  # You may need to implement get_frame_count in your RTSPVideoCapture class
                results = model(frame, tracker="botsort.yaml")[0]

                # detect
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.class_id == 0]  # person만 detect
                detections = tracker.update_with_detections(detections)

                people_count = sum(1 for class_id in detections.class_id)

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
                            firebase_manager.update_data('/users/aKSaa0in0dgZeQXMnFybODKSMdp1/people_info', data)
                            print(age)
                            print(gender)
                        except:
                            pass

                    # 얼굴 정보 표시
                    if tracker_id in faces_info:
                        cv2.putText(img_with_faces, faces_info[tracker_id], (int(x), int(y) - 50),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                current_time = time.localtime()
                if current_time.tm_sec == 0:
                    # Firebase에 데이터 업로드
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', current_time)
                    data = {
                        timestamp: {
                            'people_count': people_count
                        }
                    }
                    firebase_manager.update_data('/users/aKSaa0in0dgZeQXMnFybODKSMdp1/people_count', data)

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

                # Display the annotated frame
                cv2.imshow("Annotated Frame", annotated_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 루프가 끝나면 RTSP 연결 해제
    rtsp_capture.release()
    cv2.destroyAllWindows()
