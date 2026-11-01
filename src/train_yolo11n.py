from pathlib import Path
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Admin\Downloads\NEU_FYP")

DATASET_YAML = (
    PROJECT_ROOT
    / "results"
    / "detection_dataset"
    / "yolo_dataset"
    / "neu_defects.yaml"
)

RUNS_DIR = PROJECT_ROOT / "results" / "model_results"

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "yolo11n.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 5

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("YOLO11N DEFECT LOCALIZATION TRAINING")
print("=" * 70)

print()
print("Dataset :", DATASET_YAML)
print("Model   :", MODEL_NAME)
print("Image   :", IMAGE_SIZE)
print("Batch   :", BATCH_SIZE)
print("Epochs  :", EPOCHS)
print("Device  : CPU")

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading pretrained YOLO11n...")

model = YOLO(MODEL_NAME)

print("Model loaded successfully.")

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

results = model.train(
    data=str(DATASET_YAML),

    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,

    epochs=EPOCHS,
    patience=PATIENCE,

    device="cpu",
    workers=0,

    pretrained=True,

    project=str(RUNS_DIR),
    name="yolo11n_baseline",

    exist_ok=True,

    save=True,
    save_period=-1,

    plots=True,

    verbose=True
)

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("YOLO11N TRAINING COMPLETE")
print("=" * 70)

print()
print("Training results saved to:")

print(
    RUNS_DIR / "yolo11n_baseline"
)

print()
print("Best model should be:")
print(
    RUNS_DIR
    / "yolo11n_baseline"
    / "weights"
    / "best.pt"
)

print("=" * 70)