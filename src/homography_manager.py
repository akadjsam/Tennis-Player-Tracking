# homography_manager.py
import cv2
import numpy as np
from utils import constants


class HomographyManager:
    def __init__(self):
        # 실제 테니스 코트의 꼭짓점 4개에 대한 좌표 (미터 단위)
        # constants.py 에 정의된 값을 사용
        self.real_world_points = np.array([
            [0, 0],  # Top-left
            [constants.DOUBLE_LINE_WIDTH, 0],  # Top-right
            [0, constants.HALF_COURT_LINE_HEIGHT * 2],  # Bottom-left
            [constants.DOUBLE_LINE_WIDTH, constants.HALF_COURT_LINE_HEIGHT * 2]  # Bottom-right
        ], dtype=np.float32)

        self.homography_matrix = None

    def compute_homography_matrix(self, video_keypoints):
        court_corner_indices = [
            0, 1, 2, 3  # 4개 코너
        ]
        video_points = np.array([
            [video_keypoints[i * 2], video_keypoints[i * 2 + 1]]
            for i in court_corner_indices
        ], dtype=np.float32)

        # 호모그래피 행렬 계산
        self.homography_matrix, _ = cv2.findHomography(video_points, self.real_world_points)

        if self.homography_matrix is None:
            print("Error: Homography matrix could not be computed.")
            return False
        return True

    def transform_pixel_to_real(self, pixel_coords):
        if self.homography_matrix is None:
            raise ValueError("Homography matrix is not computed yet.")

        # 좌표를 올바른 형태로 변환 (1, N, 2)
        pixel_coords_np = np.array([[pixel_coords]], dtype=np.float32)

        # 원근 변환 적용
        transformed_points = cv2.perspectiveTransform(pixel_coords_np, self.homography_matrix)

        return transformed_points[0][0]

