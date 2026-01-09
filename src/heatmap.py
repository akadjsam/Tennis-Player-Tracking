# heatmap.py
import numpy as np
import cv2
from mini_court import MiniCourt

# 기존 MiniCourt 클래스를 상속받음
class CenteredMiniCourt(MiniCourt):
    def __init__(self, frame):
        # 부모 클래스의 초기화 메소드를 먼저 호출
        super().__init__(frame)

    # 위치 계산 메소드만 오버라이딩(재정의)하여 중앙으로 옮김
    def set_canvas_background_box_position(self, frame):
        frame_height, frame_width, _ = frame.shape

        # 프레임의 정중앙 좌표를 계산
        center_x = frame_width // 2
        center_y = frame_height // 2

        # 중앙 좌표를 기준으로 사각형의 시작점과 끝점을 계산
        self.start_x = center_x - (self.drawing_rectangle_width // 2)
        self.end_x = center_x + (self.drawing_rectangle_width // 2)
        self.start_y = center_y - (self.drawing_rectangle_height // 2)
        self.end_y = center_y + (self.drawing_rectangle_height // 2)


def create_heatmap_overlay(base_image, positions):
    # 히트맵 밀도 레이어 생성
    heatmap_layer = np.zeros_like(base_image[:, :, 0], dtype=np.float32)
    height, width, _ = base_image.shape

    # 튜닝 파라미터
    point_radius = 5 # int(width / 150)  # 반지름을 코트 크기에 비례하도록 수정
    point_intensity = 15

    # 전달받은 좌표 그대로 밀도 누적 (좌표 보정 필요 없음)
    for pos in positions:
        # 캔버스 경계 내에 있는지 확인
        if 0 <= pos[0] < width and 0 <= pos[1] < height:
            cv2.circle(heatmap_layer, (int(pos[0]), int(pos[1])), point_radius, point_intensity, -1)

    # 블러 및 컬러맵 적용
    blur_kernel_size = (int(width / 30) | 1, int(width / 30) | 1)
    heatmap_layer = cv2.GaussianBlur(heatmap_layer, blur_kernel_size, 0)
    heatmap_layer = cv2.GaussianBlur(heatmap_layer, (21,21), 0)
    normalized_heatmap = cv2.normalize(heatmap_layer, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    # 마스킹을 이용한 오버레이
    final_image = base_image.copy()
    mask = normalized_heatmap > 0
    alpha = 0.5
    final_image[mask] = cv2.addWeighted(base_image[mask], 1 - alpha, color_heatmap[mask], alpha, 0)

    return final_image