import cv2
from collections import deque, Counter
from pathlib import Path
from ultralytics import YOLO

def main():
    weights_path = Path("runs/detect/isl_public_yolov8/weights/best.pt")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. "
            f"Make sure training finished successfully."
        )

    model = YOLO(str(weights_path))
    class_names = model.names  # index -> class label from dataset

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("Webcam started — press 'q' to quit.")

    conf_threshold = 0.5
    iou_threshold = 0.45
    history = deque(maxlen=12)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # Run YOLO
        results = model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        result = results[0]
        detections = result.boxes

        current_label = None
        if detections is not None and len(detections) > 0:
            confs = detections.conf.cpu().numpy()
            clss = detections.cls.cpu().numpy().astype(int)

            best_idx = confs.argmax()
            best_class_id = clss[best_idx]
            best_conf = confs[best_idx]

            if best_conf >= conf_threshold:
                current_label = class_names.get(best_class_id, str(best_class_id))
                history.append(current_label)

        # Temporal smoothing
        stable_label = None
        if history:
            stable_label = Counter(history).most_common(1)[0][0]

        annotated = result.plot()

        if stable_label is not None:
            cv2.rectangle(annotated, (10, 10), (520, 70), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                f"Detected: {stable_label}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

        cv2.imshow("Public ISL Detection (YOLOv8)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")

if __name__ == "__main__":
    main()
