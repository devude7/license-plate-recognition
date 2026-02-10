from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "data/yolo/data.yaml"
BASE_MODEL = "yolo11s.pt"
IMG_SIZE = 960
EPOCHS = 60
BATCH = 16
DEVICE = "0"
WORKERS = 4
SEED = 42

PROJECT_DIR = "runs"
RUN_NAME = "yolo11s"

PATIENCE = 20
COS_LR = True


def main() -> None:
    repo_root = Path.cwd()

    data_path = (repo_root / DATA_YAML).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    project_dir = (repo_root / PROJECT_DIR).resolve()

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(data_path),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        project=str(project_dir),  
        name=RUN_NAME,             
        exist_ok=False,            
        patience=PATIENCE,
        cos_lr=COS_LR,
        verbose=True,
    )


if __name__ == "__main__":
    main()