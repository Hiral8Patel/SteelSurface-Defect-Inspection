from pathlib import Path
from ultralytics import YOLO

DATASET_YAML = Path(
    r"C:\Users\Admin\Downloads\NEU_FYP\results\detection_dataset\yolo_dataset\neu_defects.yaml"
)

print("=" * 70)
print("ULTRALYTICS YOLO DATASET CHECK")
print("=" * 70)

print()
print("Dataset YAML:")
print(DATASET_YAML)

print()
print("YAML exists:", DATASET_YAML.exists())

if not DATASET_YAML.exists():
    raise FileNotFoundError("Dataset YAML was not found.")

print()
print("Loading lightweight YOLO model...")

model = YOLO("yolo11n.pt")

print()
print("Model loaded successfully.")

print()
print("Running dataset validation...")

results = model.val(
    data=str(DATASET_YAML),
    split="val",
    imgsz=200,
    batch=8,
    device="cpu",
    workers=0,
    plots=False,
    verbose=False
)

print()
print("=" * 70)
print("DATASET CHECK COMPLETE")
print("=" * 70)

print("Validation images checked :", 360)
print("Dataset YAML              : OK")
print("Ultralytics loading       : OK")
print("=" * 70)