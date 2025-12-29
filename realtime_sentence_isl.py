import cv2
from collections import deque, Counter
from pathlib import Path
from ultralytics import YOLO
import time

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

    print("=" * 60)
    print("ISL SENTENCE BUILDER - WEBCAM STARTED")
    print("=" * 60)
    print("Controls:")
    print("  SPACE  - Add current sign to sentence")
    print("  ENTER  - Add space between words")
    print("  'c'    - Clear sentence")
    print("  'd'    - Delete last sign")
    print("  's'    - Save sentence to file")
    print("  'q'    - Quit")
    print("=" * 60)

    conf_threshold = 0.6
    iou_threshold = 0.45
    history = deque(maxlen=15)
    
    # Sentence building
    sentence = []
    current_word = []
    last_detection_time = time.time()
    auto_add_delay = 2.0  # seconds to wait before auto-adding sign
    last_added_sign = None
    
    # For stable detection
    detection_count = {}
    min_detections = 5  # Need 5 consistent detections to add sign

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
                
                # Count detections for stability
                if current_label in detection_count:
                    detection_count[current_label] += 1
                else:
                    detection_count = {current_label: 1}

        # Temporal smoothing
        stable_label = None
        if history:
            label_counts = Counter(history)
            stable_label = label_counts.most_common(1)[0][0]

        # Auto-add sign if stable for long enough
        if stable_label and stable_label != last_added_sign:
            if stable_label in detection_count and detection_count[stable_label] >= min_detections:
                current_time = time.time()
                if current_time - last_detection_time >= auto_add_delay:
                    current_word.append(stable_label)
                    last_added_sign = stable_label
                    last_detection_time = current_time
                    detection_count = {}
                    print(f"Auto-added: {stable_label}")

        annotated = result.plot()
        h, w = annotated.shape[:2]

        # Create info panel
        panel_height = 250
        panel = annotated.copy()
        cv2.rectangle(panel, (0, 0), (w, panel_height), (0, 0, 0), -1)

        # Display current detection
        if stable_label:
            cv2.putText(
                panel,
                f"Detected: {stable_label}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
            
            # Progress bar for auto-add
            if stable_label in detection_count:
                progress = min(detection_count[stable_label] / min_detections, 1.0)
                bar_width = 400
                cv2.rectangle(panel, (20, 70), (20 + int(bar_width * progress), 85), (0, 255, 255), -1)
                cv2.rectangle(panel, (20, 70), (20 + bar_width, 85), (255, 255, 255), 2)

        # Display current word being built
        current_word_text = " + ".join(current_word) if current_word else "(empty)"
        cv2.putText(
            panel,
            f"Current Word: {current_word_text}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Display full sentence
        sentence_text = " ".join(sentence) if sentence else "(no sentence yet)"
        cv2.putText(
            panel,
            f"Sentence: {sentence_text}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Controls help
        cv2.putText(
            panel,
            "SPACE=Add | ENTER=Space | C=Clear | D=Delete | S=Save | Q=Quit",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("ISL Sentence Builder", panel)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space - add current sign to word
            if stable_label:
                current_word.append(stable_label)
                last_added_sign = stable_label
                detection_count = {}
                print(f"Added to word: {stable_label}")
        elif key == 13:  # Enter - complete word and add to sentence
            if current_word:
                word = "".join(current_word)
                sentence.append(word)
                current_word = []
                last_added_sign = None
                print(f"Word completed: {word}")
                print(f"Sentence: {' '.join(sentence)}")
        elif key == ord('c'):  # Clear everything
            sentence = []
            current_word = []
            last_added_sign = None
            detection_count = {}
            print("Cleared sentence and word")
        elif key == ord('d'):  # Delete last sign from current word
            if current_word:
                removed = current_word.pop()
                print(f"Removed: {removed}")
        elif key == ord('s'):  # Save sentence to file
            if sentence:
                full_sentence = " ".join(sentence)
                with open("detected_sentences.txt", "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {full_sentence}\n")
                print(f"Saved: {full_sentence}")
            else:
                print("No sentence to save")

    cap.release()
    cv2.destroyAllWindows()
    print("\nExiting...")
    
    # Print final sentence
    if sentence:
        print(f"\nFinal Sentence: {' '.join(sentence)}")

if __name__ == "__main__":
    main()
