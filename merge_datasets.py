import os
import shutil
from pathlib import Path
import yaml

def merge_yolo_datasets(alpha_dir, gesture_dir, output_dir):
    alpha_dir = Path(alpha_dir)
    gesture_dir = Path(gesture_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Folders structure for YOLOv8
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "images" / "valid").mkdir(parents=True)
    (output_dir / "labels" / "train").mkdir(parents=True)
    (output_dir / "labels" / "valid").mkdir(parents=True)

    # Merge folder contents
    def copy_contents(src, dst):
        if not Path(src).exists():
            return
        for split in ["train", "valid"]:
            if (Path(src)/"images"/split).exists():
                # Copy images
                for f in (Path(src)/"images"/split).iterdir():
                    shutil.copy(f, Path(dst)/"images"/split)
                # Copy labels
                for f in (Path(src)/"labels"/split).iterdir():
                    shutil.copy(f, Path(dst)/"labels"/split)

    print("Merging Alphabet Dataset")
    copy_contents(alpha_dir, output_dir)

    print("Merging Gesture Dataset")
    copy_contents(gesture_dir, output_dir)

    # Read class names from both datasets
    alpha_yaml = yaml.safe_load(open(alpha_dir/"data.yaml"))
    gest_yaml = yaml.safe_load(open(gesture_dir/"data.yaml"))

    combined_names = alpha_yaml["names"] + gest_yaml["names"]

    combined_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "names": combined_names
    }

    yaml.dump(combined_yaml, open(output_dir/"data.yaml", "w"))

    print("\nDATASETS MERGED SUCCESSFULLY")
    print(f"Combined dataset saved at: {output_dir}")


if __name__ == "__main__":
    merge_yolo_datasets(
        "indian-sign-language-letters-1",
        "indian-sign-language-gestures-1",
        "isl_combined_dataset"
    )
