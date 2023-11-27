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

# # 사용 예시
# rtsp_url = "rtsp://your_rtsp_server_address/your_stream_path"
# rtsp_capture = RTSPVideoCapture(rtsp_url)
#
# try:
#     while True:
#         frame = rtsp_capture.read_frame()
#
#         # 프레임이 None이 아닌 경우에만 화면에 표시
#         if frame is not None:
#             cv2.imshow("RTSP Frame", frame)
#
#         # 'q' 키를 누르면 루프 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# finally:
#     # 루프가 끝나면 RTSP 연결 해제
#     rtsp_capture.release()
#     cv2.destroyAllWindows()