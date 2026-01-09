import numpy as np
from collections import deque

class DirectionAnalyzer:
    def __init__(self, min_distance_threshold=0.1):
        self.min_distance_threshold = min_distance_threshold
        # 각 선수의 모든 유효한 이동 방향 각도를 저장합니다.
        self.player_directions = {}

    def update_direction(self, player_id, prev_pos, current_pos):
        # 이동 벡터 및 거리 계산
        movement_vector = current_pos - prev_pos
        distance = np.linalg.norm(movement_vector)

        # 최소 이동 거리보다 짧으면, 제자리걸음으로 간주하고 무시
        if distance < self.min_distance_threshold:
            return

        # 이동 방향 각도 계산
        dx = movement_vector[0]
        # y축 방향을 뒤집어, 이미지 위쪽 방향이 수학적인 양의 y축이 되도록 함
        dy = -movement_vector[1]

        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360  # 각도를 0-360 범위로 변환

        # 계산된 각도를 해당 선수의 리스트에 추가
        if player_id not in self.player_directions:
            self.player_directions[player_id] = []
        self.player_directions[player_id].append(angle)

    def _circular_mean(self, angles):
        if not angles:
            return 0
        # 각도를 라디안으로 변환 후 복소 평면상의 벡터로 변환
        angles_rad = np.radians(angles)
        # 모든 벡터의 평균 계산
        mean_vector = np.mean(np.exp(1j * angles_rad))
        # 평균 벡터의 각도를 다시 도(degree)로 변환
        mean_angle_deg = np.degrees(np.angle(mean_vector))

        return (mean_angle_deg + 360) % 360

    def analyze_patterns(self, player_id):
        directions = self.player_directions.get(player_id)
        if not directions:
            return {
                "error": "No movement data available for this player."
            }

        # 원형 평균을 사용해 전체적인 평균 이동 방향 계산
        avg_direction = self._circular_mean(directions)

        # 방향의 일관성/분포를 확인하기 위한 간단한 분산 계산
        direction_variance = np.std(directions)

        return {
            "total_movements": len(directions),
            "average_direction_deg": f"{avg_direction:.2f}°",
            "direction_variance": f"{direction_variance:.2f}"
        }

    def get_histogram_data(self, player_id, bins=8):
        directions = self.player_directions.get(player_id)
        if not directions:
            return None, None, None

        # 0도에서 360도까지 지정된 수의 구간으로 나눔
        bin_edges = np.linspace(0, 360, bins + 1)
        # 각 구간에 속하는 방향 데이터의 개수를 셈
        hist, _ = np.histogram(directions, bins=bin_edges)

        # 히스토그램 각 막대의 라벨 생성 (예: N, NE, E 등등)
        labels = []
        if bins == 8:
            # 8방위 라벨 (북동쪽부터 시작)
            dir_labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
            angles = np.arange(22.5, 360, 45)
            # 데이터 순서에 맞게 라벨 재정렬
            ordered_labels = [dir_labels[int(a / 45)] for a in angles]
            labels = ordered_labels
        else:
            labels = [f"{(bin_edges[i] + bin_edges[i + 1]) / 2:.1f}°" for i in range(bins)]

        return hist, bin_edges, labels