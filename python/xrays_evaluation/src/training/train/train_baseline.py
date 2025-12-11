import os
import mlflow
from ultralytics import YOLO, settings

# 1. model config
MODEL_NAME = "yolo11x-cls"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. paths
PRETRAINED_MODEL = os.path.join(CURRENT_DIR, "pretrained_models", f"{MODEL_NAME}.pt")
DATA_PATH = os.path.join(CURRENT_DIR, "../../../../../datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121/split_data/")
LOCAL_RUNS_DIR = os.path.join(CURRENT_DIR, "runs/YOLO")

# 3. MLFow config
MLFLOW_DB_PATH = os.path.join(CURRENT_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

# ultralytics
settings.update({"mlflow": True, "runs_dir": LOCAL_RUNS_DIR})


def train_baseline():
    print(f"🚀 Iniciando entrenamiento Baseline con {MODEL_NAME}...")

    # load model
    model = YOLO(PRETRAINED_MODEL).to('cuda')

    # name experiment
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Xrays_Evaluation_Experiments"
    os.environ["MLFLOW_RUN_NAME"] = "01_Baseline_Nano"

    # train
    model.train(
        data=DATA_PATH,
        epochs=50,
        batch=64,
        imgsz=224,
        patience=5,
        task='classify',
        project=LOCAL_RUNS_DIR,
        name=f"{MODEL_NAME}",
        exist_ok=True
    )

    print("✅ Entrenamiento Baseline finalizado.")


if __name__ == '__main__':
    train_baseline()

"""
usage:
# move to ingeniia_services
dvc pull datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121.dvc
python src/training/train/train_baseline.py

visualize:
mlflow ui --backend-store-uri sqlite:///src/training/train/mlflow.db
"""