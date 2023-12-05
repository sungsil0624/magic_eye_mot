# import cv2
# import numpy as np
# from utils.heatmap_generator import HeatmapGenerator
#
# # 사용 예시
# heatmap_generator = HeatmapGenerator(width=1280, height=853)
#
# # 50개의 랜덤 (x, y) 좌표에 랜덤한 값을 누적
# # 실제 사용에서는 객체 검출 시의 x,y 값을 통해 범위를 정해서 누적
# num_points = 50
# for _ in range(num_points):
#     random_x = np.random.randint(0, 1280)
#     random_y = np.random.randint(0, 853)
#     random_value = np.random.rand() * 15  # 0에서 10 사이의 랜덤한 값
#     heatmap_generator.add_heat(x=random_x, y=random_y, value=random_value)
#
# heatmap_image = heatmap_generator.generate_heatmap()
#
# # 새로운 배경 이미지 불러오기
# background_image = cv2.imread("assets/people.jpg")
#
# # 히트맵 이미지와 배경 이미지 합치기
# heatmap_with_background = cv2.addWeighted(heatmap_image, 0.7, background_image, 0.3, 0)
#
# # 히트맵 이미지를 화면에 표시
# cv2.imshow("Heatmap with Background", heatmap_with_background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from supervision import get_video_frames_generator

import supervision as sv
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

heatmap_annotator = sv.HeatMapAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path='assets/example_2.mp4')
frames_generator = get_video_frames_generator(source_path='assets/exmaple_2.mp4')
with sv.VideoSink(target_path='result_heatmap_example_2.mp4', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == 0]
        annotated_frame = heatmatp_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        sink.write_frame(frame=annotated_frame)
