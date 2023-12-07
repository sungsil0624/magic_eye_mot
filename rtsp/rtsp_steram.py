import cv2


class RTSPVideoCapture:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.frame_index = 0  # Initialize frame index
        self.start_time = cv2.getTickCount()

        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening RTSP stream: {rtsp_url}")

    def read_frame(self):
        ret, frame = self.cap.read()

        if ret:
            # 프레임의 높이와 너비 출력
            height, width, _ = frame.shape
            # Increment frame index
            self.frame_index += 1

        return frame

    def get_frame_count(self):
        return self.frame_index

    def get_fps(self):
        # 현재 시간에서 시작 시간을 뺀 후, 프레임 수로 나눠서 FPS 계산
        elapsed_time = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
        fps = self.frame_index / elapsed_time
        return fps

    def release(self):
        self.cap.release()
