from pip._vender.typing_extensions import Callable, Generator
import supervision as sv
import cv2
import numpy as np

class VideoSink:
    def __init__(self, target_path: str, video_info: sv.VideoInfo, codec: str = "mp4v"):
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__writer = None

    def __enter__(self):
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*self.__codec)
        except TypeError as e:
            print(str(e) + ". Defaulting to mp4v...")
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        return self

    def write_frame(self, frame: np.ndarray):
        self.__writer.write(frame)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__writer.release()

def process_video(
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    webcam_index: int = 0,
) -> None:
    webcam_info = sv.VideoInfo(width=640, height=480, fps=30)  # Adjust these values based on your webcam
    with VideoSink(target_path=target_path, video_info=webcam_info) as sink:
        for index, frame in enumerate(get_webcam_frames_generator(webcam_index)):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)

def get_webcam_frames_generator(webcam_index: int) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(webcam_index)

    if not cap.isOpened():
        raise RuntimeError(f"Error opening webcam with index {webcam_index}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # detection 작업 전 그냥 웹캠 화면
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        yield frame

    cap.release()
    cv2.destroyAllWindows()