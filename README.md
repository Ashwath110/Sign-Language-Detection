# Indian Sign Language Detection

A real-time Indian Sign Language (ISL) detection system using YOLOv8 for gesture recognition through webcam input.

## Demo

<p align="center">
  <img src="Screenshot 2025-12-29 190329.png" alt="Indian Sign Language Detection - Real-time Detection showing letter 'E'" width="800">
</p>

## Features

- 🤖 YOLOv8-based sign language detection model
- 📹 Real-time webcam detection
- 💬 Sentence formation from detected gestures
- 📊 Custom dataset training support
- 🎯 High accuracy gesture recognition

## Project Structure

```
SignLanguageDetection/
├── train_public_isl.py          # Main training script
├── realtime_public_isl.py       # Real-time single gesture detection
├── realtime_sentence_isl.py     # Real-time sentence formation
├── download_isl_dataset.py      # Dataset download utility
├── merge_datasets.py            # Merge multiple datasets
├── train_in_colab.ipynb         # Google Colab training notebook
├── best.pt                      # Trained model weights
├── yolov8s.pt                   # Base YOLOv8 small model
├── Indian-Sign-Language-Detection-1/  # Dataset directory
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
└── runs/                        # Training results and logs
```

## Requirements

- Python 3.11+
- OpenCV
- Ultralytics YOLOv8
- Webcam for real-time detection

## Installation

1. **Clone or download this repository**

2. **Install required packages:**
```bash
pip install ultralytics opencv-python
```

## Dataset Setup

The project uses the Indian Sign Language Detection dataset. The dataset should be organized in the following structure:

```
Indian-Sign-Language-Detection-1/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

If you need to download or merge datasets, use the provided utility scripts:
- `download_isl_dataset.py` - Download ISL dataset
- `merge_datasets.py` - Merge multiple datasets

## Training

### Train the Model

Run the training script to train YOLOv8 on the ISL dataset:

```bash
python train_public_isl.py
```

**Training Parameters:**
- Epochs: 80 (configurable)
- Image size: 640x640
- Batch size: 8
- Workers: 0 (for Windows compatibility)

The trained model will be saved at: `runs/detect/isl_public_yolov8/weights/best.pt`

### Training in Google Colab

Alternatively, use the provided Jupyter notebook for training in Google Colab:
- Open `train_in_colab.ipynb`
- Follow the instructions in the notebook

## Usage

### Real-time Gesture Detection

Run real-time sign language detection with your webcam:

```bash
python realtime_public_isl.py
```

**Controls:**
- Press `q` to quit the application

**Features:**
- Displays detected sign language gesture in real-time
- Shows confidence scores
- Smooths predictions using temporal averaging

### Real-time Sentence Formation

For sentence formation from continuous gestures:

```bash
python realtime_sentence_isl.py
```

This mode allows you to form complete sentences by detecting multiple signs in sequence.

## Model Details

- **Base Model:** YOLOv8 Small (yolov8s.pt)
- **Framework:** Ultralytics YOLOv8
- **Input Size:** 640x640
- **Detection:** Real-time object detection for hand gestures
- **Confidence Threshold:** 0.5
- **IOU Threshold:** 0.45

## Training Results

After training completes, you can find:
- **Best weights:** `runs/detect/isl_public_yolov8/weights/best.pt`
- **Last weights:** `runs/detect/isl_public_yolov8/weights/last.pt`
- **Training metrics:** `runs/detect/isl_public_yolov8/results.csv`
- **Plots:** Various performance plots in the runs directory

## Performance Tips

1. **GPU Acceleration:** Training and inference will be much faster with CUDA-enabled GPU
2. **Batch Size:** Increase batch size if you have more GPU memory
3. **Epochs:** Train for more epochs (100+) for better accuracy
4. **Dataset Quality:** Ensure good lighting and clear hand gestures in training data

## Troubleshooting

### Webcam Issues
- If webcam fails to open, check if it's being used by another application
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for external webcams

### Model Not Found
- Ensure training completed successfully
- Check that `best.pt` exists in the expected location
- Update the path in the detection scripts if necessary

### Low Accuracy
- Train for more epochs
- Increase dataset size
- Improve image quality and variety in training data
- Adjust confidence threshold

## Dataset Information

The Indian Sign Language Detection dataset contains labeled images of various ISL gestures. Each gesture is annotated with bounding boxes in YOLO format.

- **Train Set:** Training images and labels
- **Valid Set:** Validation images and labels  
- **Test Set:** Test images and labels
- **data.yaml:** Dataset configuration with class names and paths

## Contributing

Feel free to contribute to this project by:
- Adding more ISL gestures
- Improving detection accuracy
- Enhancing the user interface
- Adding new features

## License

This project is for educational purposes. Please check the dataset license in `Indian-Sign-Language-Detection-1/README.roboflow.txt`

## Acknowledgments

- YOLOv8 by Ultralytics
- Indian Sign Language Detection dataset from Roboflow
- OpenCV community

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Sign Language Detection! 👋**
