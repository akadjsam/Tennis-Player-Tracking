import matplotlib.pyplot as plt
import numpy as np

def plot_direction_histogram(directions, pid, output_path, name):
    # 8방향 (0, 45, 90, ...)으로 구간을 나눔
    bins = np.arange(0, 361, 45)
    # 각 구간에 속하는 각도의 개수를 계산
    hist, bin_edges = np.histogram(directions, bins=bins)

    # 시각화
    plt.figure(figsize=(8, 8), dpi=100)
    ax = plt.subplot(111, polar=True)  # 원형(polar) 좌표계 사용

    # 막대 그래프 그리기
    # 각 막대의 각도 (라디안으로 변환)와 너비 설정
    angles = np.deg2rad(bin_edges[:-1] + 22.5)  # 각 구간의 중간 각도
    width = np.deg2rad(45)  # 각 막대의 너비
    bars = ax.bar(angles, hist, width=width, bottom=0.0, alpha=0.7)

    # 라벨 및 제목 설정
    ax.set_rgrids([])
    ax.set_theta_zero_location('N')  # 0도를 북쪽(위)으로 설정
    ax.set_theta_direction(-1)  # 시계 방향으로 각도 증가
    ax.set_thetagrids(np.arange(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    plt.title(f'Player {pid} Movement Direction Histogram')

    # 파일로 저장
    save_path = f"{output_path}/{name}_direction_histogram_player_{pid}.png"
    plt.savefig(save_path)
    print(f"Direction histogram for Player {pid} saved to {save_path}")
    plt.close()