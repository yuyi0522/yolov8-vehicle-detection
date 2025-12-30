from ultralytics import YOLO
import os
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
#from StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT



# --------------------------
# Initialize DeepSORT and Model
# --------------------------
# max_age: 幾幀沒有匹配仍保持 track
# n_init: track 啟動前需要匹配幾幀
# embedder: 'mobilenet' 為預訓練 CNN ReID
model = YOLO("weights/best.pt")
tracker = DeepSort(max_age=150, n_init=30,nms_max_overlap=1.0,max_cosine_distance=0.8, embedder="mobilenet")


# --------------------------
# basic initialize
# --------------------------
video_files = './videos/p2.mp4'
save_dir = "./videos/runs"
save_name = "XYWH0.8_p2_result_video.mp4"
os.makedirs(save_dir, exist_ok=True)

class_names = {
    0: "Human",
    1: "Car",
    2: "Truck",
    3: "Ship",
    4: "Ferry",
    5: "Freighter",
    6: "Fishing Boat",
    7: "Speed Boat",
    8: "Inflatable Boat",
    9: "Raft"
}

# 可以隨意設定顏色 (BGR)
colors = {
    0: (0, 255, 0),       # Human - 綠
    1: (255, 0, 0),       # Car - 藍
    2: (0, 0, 255),       # Truck - 紅
    3: (255, 255, 0),     # Ship - 黃
    4: (0, 255, 255),     # Ferry - 青
    5: (255, 0, 255),     # Freighter - 紫
    6: (128, 128, 0),     # Fishing Boat - 橄欖
    7: (0, 128, 128),     # Speed Boat - 深青
    8: (128, 0, 128),     # Inflatable Boat - 深紫
    9: (50, 50, 50)       # Raft - 灰
}
# --------------------------
# Video input/output
# --------------------------
cap = cv2.VideoCapture(video_files)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(save_dir, save_name), fourcc, fps, (width, height))

frame_count = 0

# --------------------------
# 累計總數的字典
# --------------------------
total_counts = {cls_id: set() for cls_id in class_names.keys()}  # 用 set 追蹤已出現的 track_id


# --------------------------
# Main loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 偵測
    results = model.predict(frame, iou=0.7, conf=0.4, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        conf = float(box.conf)
        cls_id = int(box.cls)
        detections.append(([x1, y1, w, h], conf, cls_id))


    # DeepSORT 更新 track
    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cls_id = track.det_class if hasattr(track, "det_class") else -1
        color = colors.get(cls_id, (255, 255, 255))

        label = f"{class_names.get(cls_id, 'id_' + str(cls_id))} | ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 累計總數
        if cls_id >= 0:
            total_counts[cls_id].add(track_id)

        # --------------------------
        # 顯示左上角累計數
        # --------------------------
        h, w = frame.shape[:2]
        # 動態設定字體大小（fontScale）
        font_scale = max(0.5, min(2.5, h / 1000))  # 依圖片高度自動調整
        thickness = max(1, int(h / 600))  # 線條粗細
        y_offset = 30
        for cls_id, track_ids in total_counts.items():
            text = f"{class_names[cls_id]}: {len(track_ids)}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[cls_id], thickness)
            y_offset += int(25 * font_scale)

    out.write(frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"已處理 {frame_count} 幀")

cap.release()
out.release()
print(f"✅ 影片結果已存到 {save_dir}/result_video.mp4")