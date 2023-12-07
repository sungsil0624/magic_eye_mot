from supervision import get_video_frames_generator
import supervision as sv
from ultralytics import YOLO

model = YOLO('../yolov8n.pt')

heatmap_annotator = sv.HeatMapAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path='../assets/example1.mp4')
frames_generator = get_video_frames_generator(source_path='../assets/example1.mp4')
with sv.VideoSink(target_path='../result_heatmap_example1.mp4', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == 0]
        annotated_frame = heatmap_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        sink.write_frame(frame=annotated_frame)