import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def reset_heatmap(self):
        """
        누적된 값 초기화
        """
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)

    def add_heat(self, x, y, value=1):
        """
        x,y 좌표 값 누적

        Args:
            x (int): x 좌표
            y (int): y 좌표
            value (float): 누적할 값 (기본값: 1)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.heatmap[y, x] += value

    def generate_heatmap(self):
        # 빈도수의 범위를 확인하여 가장 큰 값으로 나누어 강도를 조절
        max_value = np.max(self.heatmap)
        if max_value == 0:
            max_value = 1  # 0으로 나누는 것을 방지

        # 빈도수를 색상 강도로 변환하여 히트맵 생성
        heatmap_image = cv2.applyColorMap(
            np.uint8(255 * self.heatmap / max_value), cv2.COLORMAP_HOT
        )

        # 각 좌표에 대해 큰 크기로 점을 그림
        for y, row in enumerate(self.heatmap):
            for x, value in enumerate(row):
                if value > 0:
                    # 점 크기를 값에 비례하도록 설정 (원하는 크기로 조절 가능)
                    radius = int(value)
                    # 히트맵 이미지에 점을 그릴 때 빈도수를 고려하여 색상을 지정
                    color_intensity = int(255 * value / max_value)
                    color = (0, 0, color_intensity)  # 빨간색 계열로 설정
                    cv2.circle(heatmap_image, (x, y), radius, color, thickness=-1)  # thickness=-1은 원을 채워 그림

        return heatmap_image
