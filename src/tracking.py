# tracking.py
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 클래스 이름 매핑 및 추적 대상 클래스 설정
CLASS_NAMES = {0: 'players', 1: 'tennis_ball'}
TARGET_CLASS = 'players'


def init_tracker(model_path: str):
    print("Initializing YOLO model")
    model = YOLO(model_path)
    print("Initializing DeepSort tracker")
    tracker = DeepSort(max_age=60, n_init=3)
    return model, tracker


def process_frame(frame, model, tracker):
    # YOLO 추론 (CPU 모드)
    results = model.predict(frame, imgsz=640, device='cpu', verbose=False)[0]

    # 검출 결과 중 TARGET_CLASS 만 필터링
    detections = []
    for *box, score, cls in results.boxes.data.tolist():
        cls_id = int(cls)
        class_name = CLASS_NAMES.get(cls_id, 'unknown')
        if class_name != TARGET_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box)
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, cls_id))

    # DeepSORT 트래킹 업데이트
    tracks = tracker.update_tracks(detections, frame=frame)

    player_centers_current_frame = []  # 현재 프레임의 선수 중앙 좌표 저장
    player_boxes = {}
    # 트랙 결과 그리기
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        if int(tid) >= 3:
            continue
        x, y, w, h = map(int, track.to_tlwh())
        player_boxes[tid] = (x, y, x + w, y + h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bottom_x = x + w // 2
        bottom_y = y + h
        cv2.circle(frame, (bottom_x, bottom_y), 5, (255, 255, 0), -1)
        cv2.putText(frame, f'Player ID:{tid}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        player_centers_current_frame.append((tid, bottom_x, bottom_y))
    print(player_boxes)
    return frame, player_boxes, player_centers_current_frame