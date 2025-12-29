from pathlib import Path
from ultralytics import YOLO

def main():
    #Change this if your dataset.location printed a different folder name
    DATASET_DIR = Path("Indian-Sign-Language-Detection-1")
    data_yaml = DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find data.yaml at {data_yaml}. "
            f"Check dataset.location from download_public_isl.py."
        )

    print("Starting YOLOv8 training on public ISL dataset...")

    # Base YOLOv8 model (small)
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=80,        # increase to 100+ if you want maximum accuracy
        imgsz=640,
        batch=8,
        workers=0,        # safer on Windows
        project="runs",
        name="isl_public_yolov8",
        verbose=True
    )

    print("\nTraining finished.")
    print("Check 'runs/detect/isl_public_yolov8/weights/best.pt' for the trained model.")

if __name__ == "__main__":
    main()
