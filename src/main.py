import cv2
import os
import argparse
import numpy as np
from collections import deque
from pykalman import KalmanFilter

# Custom Modules
from tracking import init_tracker, process_frame
from court_line_detector import CourtLineDetector
from homography_manager import HomographyManager
from mini_court import MiniCourt
from heatmap import create_heatmap_overlay, CenteredMiniCourt
from make_histogram import plot_direction_histogram

# 상수 설정
POS_WINDOW_SIZE = 15
SPEED_WINDOW_SIZE = 15
COOLDOWN_FRAMES = 15
ABNORMAL_THRESH = 7
MIN_MOVEMENT_DIST = 0.05

TENNIS_COURT_WIDTH_M = 10.97
TENNIS_COURT_HEIGHT_M = 23.77

# 칼만필터 파라미터
DT = 1.0
KF_TRANSITION_MATRIX = [[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]]
KF_OBSERVATION_MATRIX = [[1, 0, 0, 0], [0, 1, 0, 0]]
KF_TRANS_COV = np.eye(4) * 0.03
KF_OBS_COV = np.eye(2) * 80

def transform_real_to_minicourt(real_pos, real_dims, mc_dims, mc_origin):
    real_w, real_h = real_dims
    mc_w, mc_h = mc_dims
    ox, oy = mc_origin
    x_mc = (real_pos[0] / real_w) * mc_w
    y_mc = (real_pos[1] / real_h) * mc_h
    return (int(ox + x_mc), int(oy + y_mc))

def initialize_kalman_filter():
    return KalmanFilter(
        transition_matrices=KF_TRANSITION_MATRIX,
        observation_matrices=KF_OBSERVATION_MATRIX,
        transition_covariance=KF_TRANS_COV,
        observation_covariance=KF_OBS_COV
    )

def draw_player_stats(frame, stats, fps, frame_count):
    start_x, start_y = 30, 50
    line_height = 35
    gap = 180
    sorted_pids = sorted(stats['distances'].keys())

    for i, pid in enumerate(sorted_pids):
        base_y = start_y + i * gap
        dist = stats['distances'].get(pid, 0)
        max_speed = stats['max_speeds'].get(pid, 0) * 3.6  # km/h
        avg_speed = 0
        if frame_count > 0:
            avg_speed = (dist / (frame_count / fps)) * 3.6

        texts = [
            (f"--- Player {pid} ---", (0, 255, 0)),
            (f"Distance: {dist:.2f} m", (255, 255, 255)),
            (f"Top Speed: {max_speed:.2f} km/h", (255, 255, 255)),
            (f"Avg Speed: {avg_speed:.2f} km/h", (255, 255, 255))
        ]
        for j, (text, color) in enumerate(texts):
            cv2.putText(frame, text, (start_x, base_y + line_height * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def generate_post_processing_outputs(stats, base_frame, old_mc, project_name, output_dir):
    print("Generating heatmaps and histograms...")
    heatmap_canvas = np.full((720, 1280, 3), (128, 0, 0), dtype=np.uint8)
    centered_mc = CenteredMiniCourt(heatmap_canvas)
    heatmap_canvas = centered_mc.draw_background_rectangle(heatmap_canvas)
    heatmap_canvas = centered_mc.draw_court(heatmap_canvas)

    old_off_x, old_off_y = old_mc.court_start_x, old_mc.court_start_y
    new_off_x, new_off_y = centered_mc.court_start_x, centered_mc.court_start_y

    transformed_pos = {}
    for pid, pos_list in stats['minicourt_positions'].items():
        new_list = []
        for (x, y) in pos_list:
            new_x = x - old_off_x + new_off_x
            new_y = y - old_off_y + new_off_y
            new_list.append((new_x, new_y))
        transformed_pos[pid] = new_list

    for pid, positions in transformed_pos.items():
        heatmap_img = create_heatmap_overlay(heatmap_canvas.copy(), positions)
        save_path = os.path.join(output_dir, f"{project_name}_heatmap_player_{pid}.png")
        cv2.imwrite(save_path, heatmap_img)

    for pid, directions in stats['movement_directions'].items():
        if directions:
            plot_direction_histogram(directions, pid, output_dir, project_name)


def init_resources(args):
    # 디렉토리 생성
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    heatmap_dir = os.path.join(output_dir, "heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)

    # 모델 로드
    print(f"Loading models...\n Tracker: {args.tracker_model}\n Court: {args.court_model}")
    model, tracker = init_tracker(args.tracker_model)
    court_detector = CourtLineDetector(args.court_model)
    kf = initialize_kalman_filter()

    # 비디오 로드
    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise IOError(f"Error: Failed to open video file: {args.input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    print(f"Video Info: {width}x{height} @ {fps} FPS")

    return cap, out, fps, model, tracker, court_detector, kf, heatmap_dir


def setup_court_system(cap, court_detector):
    """첫 프레임을 읽어 코트 감지 및 미니코트 설정"""
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Error: Cannot read first frame.")

    # 호모그래피 계산
    original_court_key_points = court_detector.predict(first_frame)
    homography_manager = HomographyManager()
    if not homography_manager.compute_homography_matrix(original_court_key_points):
        raise ValueError("Error: Could not compute homography matrix.")

    # 미니코트 초기화
    mini_court = MiniCourt(first_frame)
    mc_height_px = mini_court.convert_meters_to_pixels(TENNIS_COURT_HEIGHT_M)

    court_info = {
        'real_dims': (TENNIS_COURT_WIDTH_M, TENNIS_COURT_HEIGHT_M),
        'mc_dims': (mini_court.court_drawing_width, mc_height_px),
        'mc_origin': (mini_court.court_start_x, mini_court.court_start_y)
    }

    return first_frame, mini_court, homography_manager, court_info


def process_player_physics(pid, bbox, stats, kf, homography_manager, fps):
    x1, y1, x2, y2 = map(int, bbox)
    raw_foot_pos = ((x1 + x2) // 2, y2)
    current_y_center = (y1 + y2) / 2

    # 1. 칼만필터 업데이트
    measurement = np.array([raw_foot_pos[0], raw_foot_pos[1]])
    if pid not in stats['kalman_states']:
        stats['kalman_states'][pid] = ([measurement[0], measurement[1], 0, 0], np.eye(4))

    prev_mean, prev_cov = stats['kalman_states'][pid]
    new_mean, new_cov = kf.filter_update(prev_mean, prev_cov, observation=measurement)
    stats['kalman_states'][pid] = (new_mean, new_cov)

    stable_foot_pos = (int(new_mean[0]), int(new_mean[1]))
    foot_pos_real = homography_manager.transform_pixel_to_real(stable_foot_pos)

    # 2. 거리계산
    if pid in stats['prev_real_pos']:
        dist_delta = np.linalg.norm(foot_pos_real - stats['prev_real_pos'][pid])
        if dist_delta < 2.0:  # Prevent teleportation artifacts
            stats['distances'][pid] = stats['distances'].get(pid, 0) + dist_delta
    stats['prev_real_pos'][pid] = foot_pos_real

    # 3. 이상 움직임 탐지 및 스무딩
    pos_history = stats['real_pos_history'].setdefault(pid, deque(maxlen=POS_WINDOW_SIZE))

    # Check cooldown or abrupt vertical movement
    is_abnormal = False
    if stats['cooldown_counter'].get(pid, 0) > 0:
        is_abnormal = True
        stats['cooldown_counter'][pid] -= 1
    elif pid in stats['last_y_center']:
        v_velocity = current_y_center - stats['last_y_center'][pid]
        if abs(v_velocity) > ABNORMAL_THRESH:
            is_abnormal = True
            stats['cooldown_counter'][pid] = COOLDOWN_FRAMES

    stats['last_y_center'][pid] = current_y_center

    if is_abnormal:
        # Reset history on anomaly
        if pid in stats['smoothed_real_pos']: del stats['smoothed_real_pos'][pid]
        if pid in stats['speeds_history']: stats['speeds_history'][pid].clear()
    else:
        # Normal processing
        if len(pos_history) < POS_WINDOW_SIZE:
            pos_history.append(foot_pos_real)
        else:
            pos_history.append(foot_pos_real)
            curr_smooth_pos = np.mean(pos_history, axis=0)

            if pid in stats['smoothed_real_pos']:
                prev_smooth_pos = stats['smoothed_real_pos'][pid]

                # 방향
                dx = curr_smooth_pos[0] - prev_smooth_pos[0]
                dy = -(curr_smooth_pos[1] - prev_smooth_pos[1])  # Invert Y
                if np.sqrt(dx ** 2 + dy ** 2) > MIN_MOVEMENT_DIST:
                    math_angle = np.degrees(np.arctan2(dy, dx))
                    compass_angle = (math_angle - 90 + 360) % 360
                    stats['movement_directions'].setdefault(pid, []).append(compass_angle)

                # 속도
                move_dist = np.linalg.norm(curr_smooth_pos - prev_smooth_pos)
                inst_speed = move_dist * fps

                spd_hist = stats['speeds_history'].setdefault(pid, deque(maxlen=SPEED_WINDOW_SIZE))
                spd_hist.append(inst_speed)

                if len(spd_hist) == SPEED_WINDOW_SIZE:
                    median_speed = np.median(spd_hist)
                    if median_speed < 11.0:
                        curr_max = stats['max_speeds'].get(pid, 0)
                        if median_speed > curr_max:
                            stats['max_speeds'][pid] = median_speed

            stats['smoothed_real_pos'][pid] = curr_smooth_pos

    return stable_foot_pos, foot_pos_real


def main(args):
    # 1. 초기화 및 리소스 로드
    try:
        cap, out, fps, model, tracker, court_detector, kf, heatmap_dir = init_resources(args)

        # 2. 코트 시스템 설정 (첫 프레임 처리)
        first_frame, mini_court, homography_manager, court_info = setup_court_system(cap, court_detector)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 상태 저장소 초기화
    stats = {
        'distances': {}, 'prev_real_pos': {}, 'max_speeds': {},
        'positions_history': {}, 'movement_directions': {},
        'minicourt_positions': {}, 'kalman_states': {},
        'real_pos_history': {}, 'smoothed_real_pos': {},
        'speeds_history': {}, 'last_y_center': {},
        'cooldown_counter': {}, 'stable_foot_pos': {}
    }

    frame_count = 0
    print("Start processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 3. Tracking 및 미니코트 그리기
        processed_frame, player_boxes, _ = process_frame(frame.copy(), model, tracker)
        processed_frame = mini_court.draw_background_rectangle(processed_frame)
        processed_frame = mini_court.draw_court(processed_frame)

        if len(player_boxes) == 2:
            for pid, bbox in player_boxes.items():
                # 4. 개별 플레이어 물리 연산 처리
                stable_foot_pos, foot_pos_real = process_player_physics(
                    pid, bbox, stats, kf, homography_manager, fps
                )

                # 시각화 데이터 계산
                foot_pos_mc = transform_real_to_minicourt(
                    foot_pos_real,
                    court_info['real_dims'],
                    court_info['mc_dims'],
                    court_info['mc_origin']
                )
                stats['minicourt_positions'].setdefault(pid, []).append(foot_pos_mc)

                # 그리기
                cv2.circle(processed_frame, stable_foot_pos, 8, (0, 255, 255), -1)
                cv2.circle(processed_frame, foot_pos_mc, 5, (0, 0, 255), -1)

        # 5. 통계 표시 및 저장
        draw_player_stats(processed_frame, stats, fps, frame_count)
        out.write(processed_frame)

        if frame_count % 50 == 0:
            print(f"Processed frame {frame_count}...")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 후처리 (히트맵, 히스토그램 등)
    generate_post_processing_outputs(stats, first_frame, mini_court, args.name, heatmap_dir)
    print(f"Done! Output saved to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tennis Player Tracking and Analysis")
    default_input = '../testvideo/shelton/bandicam 2025-08-09 21-34-55-803.avi'
    default_tracker = '../weights/best.pt'
    default_court = '../weights/keypoints_model_50.pth'
    default_name = 'result'

    parser.add_argument('--input_path', type=str, default=default_input)
    parser.add_argument('--tracker_model', type=str, default=default_tracker)
    parser.add_argument('--court_model', type=str, default=default_court)
    parser.add_argument('--name', type=str, default=default_name)

    args = parser.parse_args()
    args.output_path = f'../testoutput/{args.name}.mp4'

    main(args)