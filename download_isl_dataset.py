from roboflow import Roboflow

def main():
    # 🔴 Put your Roboflow API key here (from your account settings)
    rf = Roboflow(api_key="xwbQNSeTGsWM0tWBRIjv")

    # ⚠ Update these two strings to match the URL of the project you forked.
    # Example Universe URL:
    # https://universe.roboflow.com/ashwath-oh952/indian-sign-language-detection-dxgag
    workspace_slug = "ashwath-oh952"
    project_slug = "indian-sign-language-detection-dxgag"

    project = rf.workspace(workspace_slug).project(project_slug)
    dataset = project.version(1).download("yolov8")

    print("\nDataset downloaded.")
    print(f"Dataset location: {dataset.location}")

if __name__ == "__main__":
    main()
