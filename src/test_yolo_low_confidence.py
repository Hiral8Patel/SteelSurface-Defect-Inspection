from ultralytics import YOLO
from pathlib import Path


# ============================================================
# YOLO11N LOW-CONFIDENCE CRAZING TEST
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Admin\Downloads\NEU_FYP")

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_results"
    / "yolo11n_baseline"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "results"
    / "detection_dataset"
    / "yolo_dataset"
    / "images"
    / "validation"
)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


print("=" * 70)
print("YOLO11N LOW-CONFIDENCE CRAZING TEST")
print("=" * 70)

model = YOLO(str(MODEL_PATH))

crazing_images = sorted(
    IMAGE_DIR.glob("crazing_*.jpg")
)

print(f"\nCrazing images found: {len(crazing_images)}")

thresholds = [0.25, 0.10, 0.05]

for conf_threshold in thresholds:

    print("\n" + "-" * 70)
    print(f"CONFIDENCE THRESHOLD: {conf_threshold}")
    print("-" * 70)

    images_with_crazing = 0
    total_crazing_predictions = 0
    total_predictions = 0

    for image_path in crazing_images:

        results = model.predict(
            source=str(image_path),
            imgsz=224,
            conf=conf_threshold,
            iou=0.45,
            verbose=False,
            device="cpu",
        )

        result = results[0]
        boxes = result.boxes

        found_crazing = False

        if boxes is not None and len(boxes) > 0:

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                total_predictions += 1

                if class_id == 0:

                    found_crazing = True
                    total_crazing_predictions += 1

        if found_crazing:
            images_with_crazing += 1

    print(
        f"Images with crazing prediction : "
        f"{images_with_crazing}/60"
    )

    print(
        f"Total crazing predictions       : "
        f"{total_crazing_predictions}"
    )

    print(
        f"Total predictions               : "
        f"{total_predictions}"
    )


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)