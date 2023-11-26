import cv2


class RTSPVideoCapture:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)

        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening RTSP stream: {rtsp_url}")

    def read_frame(self):
        ret, frame = self.cap.read()

        if ret:
            # 프레임의 높이와 너비 출력
            height, width, _ = frame.shape
            print(f"Frame Height: {height}, Frame Width: {width}")

        return frame

    def release(self):
        self.cap.release()
