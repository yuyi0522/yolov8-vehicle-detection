from ultralytics import YOLO
import cv2
import numpy as np

def predict(image_bytes):

# 模型
    model = YOLO("weights/best.pt")

    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = model.track(
        source=img,
        tracker="bytetrack.yaml",
        save=True,
        save_txt=True,
        conf=0.4,
        iou=0.3,
        agnostic_nms=False,
        track_thresh: 0.7,
        track_buffer: 80,
        match_thresh: 0.8,
        mot20: True
        )


    objects = []

    for result in results:
        boxes = result.boxes

        for i, box in enumerate(boxes):
            tlbr = box.xyxy[0].tolist()
            obj = {
                "id": int(box.id.item()) if box.id is not None else -1,  # track ID
                "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])],
                "score": float(box.conf.item()),
                "class": int(box.cls.item())
            }
            objects.append(obj)

    return objects

'''
    all_data = []
    class_names = model.names
    df = pd.DataFrame(all_data)
    df.to_csv("prediction_results.csv", index=False)
'''

'''
    for i, result in enumerate(results):
        im = result.plot()
        counts = {cls_name: 0 for cls_name in class_names.values()}

        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = class_names.get(cls_id, f"id_{cls_id}")
            counts[cls_name] += 1


        y0 = 15
        for cls_name, count in counts.items():
            if count > 0:
                text = f"{cls_name}: {count}"
                cv2.putText(im, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0, 255, 0), 1, cv2.LINE_AA)
                y0 += 20


        save_path = os.path.join(save_dir, f"result_{i}.jpg")
        cv2.imwrite(save_path, im)
    print(f"✅ 所有結果已存到資料夾: {save_dir}")
'''


if __name__ == '__main__':
    predict("./test", "./result")