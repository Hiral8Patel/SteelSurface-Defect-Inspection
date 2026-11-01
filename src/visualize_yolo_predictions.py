from ultralytics import YOLO
from pathlib import Path
import random
import cv2


# ============================================================
# YOLO11N PREDICTION VISUALIZATION
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "yolo_predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


print("=" * 70)
print("YOLO11N PREDICTION VISUALIZATION")
print("=" * 70)

print(f"Model      : {MODEL_PATH}")
print(f"Image dir  : {IMAGE_DIR}")
print(f"Output dir : {OUTPUT_DIR}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading YOLO11n model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ------------------------------------------------------------
# Select validation images
# ------------------------------------------------------------

image_paths = list(IMAGE_DIR.glob("*.jpg"))

print(f"\nValidation images found: {len(image_paths)}")

random.seed(42)

selected_images = random.sample(
    image_paths,
    min(12, len(image_paths))
)

print(f"Selected images: {len(selected_images)}")


# ------------------------------------------------------------
# Run predictions
# ------------------------------------------------------------

print("\nRunning predictions...\n")

for index, image_path in enumerate(selected_images, start=1):

    print(f"[{index}/{len(selected_images)}] {image_path.name}")

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device="cpu"
    )

    result = results[0]

    # Save annotated prediction image
    annotated = result.plot()

    output_path = OUTPUT_DIR / f"prediction_{index}_{image_path.stem}.jpg"

    cv2.imwrite(
        str(output_path),
        annotated
    )

    # Print prediction details
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print("    No detections")

    else:

        print(f"    Detections: {len(boxes)}")

        for box_index, box in enumerate(boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = (
                CLASS_NAMES[class_id]
                if class_id < len(CLASS_NAMES)
                else f"class_{class_id}"
            )

            print(
                f"       {box_index + 1}. "
                f"{class_name:<18} "
                f"confidence={confidence:.3f}"
            )


print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(f"\nGenerated images:")
print(OUTPUT_DIR)

for path in sorted(OUTPUT_DIR.glob("*.jpg")):
    print(f"  - {path.name}")