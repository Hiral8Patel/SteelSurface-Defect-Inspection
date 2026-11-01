from ultralytics import YOLO
from pathlib import Path
from collections import Counter, defaultdict
import json


# ============================================================
# YOLO11N SYSTEMATIC ERROR ANALYSIS
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

LABEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "detection_dataset"
    / "yolo_dataset"
    / "labels"
    / "validation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_results"
    / "yolo11n_baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


# ------------------------------------------------------------
# Helper: read YOLO ground-truth labels
# ------------------------------------------------------------

def read_ground_truth(label_path):

    counts = Counter()

    if not label_path.exists():
        return counts

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) >= 5:

                class_id = int(parts[0])

                if 0 <= class_id < len(CLASS_NAMES):
                    counts[CLASS_NAMES[class_id]] += 1

    return counts


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("=" * 70)
print("YOLO11N SYSTEMATIC ERROR ANALYSIS")
print("=" * 70)

print(f"Model : {MODEL_PATH}")
print(f"Images: {IMAGE_DIR}")
print(f"Labels: {LABEL_DIR}")

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ------------------------------------------------------------
# Validation images
# ------------------------------------------------------------

image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

print(f"\nValidation images: {len(image_paths)}")


# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

ground_truth_total = Counter()
prediction_total = Counter()

images_with_predictions = 0
images_without_predictions = 0

confidence_by_class = defaultdict(list)

no_detection_by_class = Counter()

prediction_image_counts = Counter()

total_predictions = 0


# ------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------

print("\nRunning predictions on entire validation set...\n")

for index, image_path in enumerate(image_paths, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_counts = read_ground_truth(label_path)

    ground_truth_total.update(gt_counts)

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device="cpu"
    )

    result = results[0]

    boxes = result.boxes

    predicted_classes = []

    if boxes is not None and len(boxes) > 0:

        images_with_predictions += 1

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id < len(CLASS_NAMES):

                class_name = CLASS_NAMES[class_id]

                prediction_total[class_name] += 1
                confidence_by_class[class_name].append(confidence)

                predicted_classes.append(class_name)

                total_predictions += 1

    else:

        images_without_predictions += 1

        # Record all ground-truth classes for images
        # where YOLO detected nothing.
        for class_name in gt_counts:

            no_detection_by_class[class_name] += gt_counts[class_name]

    for class_name in set(predicted_classes):

        prediction_image_counts[class_name] += 1

    if index % 50 == 0:

        print(
            f"Processed {index}/{len(image_paths)} images..."
        )


# ------------------------------------------------------------
# Calculate confidence statistics
# ------------------------------------------------------------

average_confidence = {}

for class_name in CLASS_NAMES:

    values = confidence_by_class[class_name]

    if values:

        average_confidence[class_name] = {
            "count": len(values),
            "average": round(sum(values) / len(values), 4),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4),
        }

    else:

        average_confidence[class_name] = {
            "count": 0,
            "average": None,
            "minimum": None,
            "maximum": None,
        }


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET-LEVEL RESULTS")
print("=" * 70)

print(f"\nTotal validation images       : {len(image_paths)}")
print(f"Images with predictions       : {images_with_predictions}")
print(f"Images with NO predictions    : {images_without_predictions}")
print(f"Total predicted boxes         : {total_predictions}")


print("\n" + "-" * 70)
print("GROUND-TRUTH BOX COUNTS")
print("-" * 70)

for class_name in CLASS_NAMES:

    print(
        f"{class_name:<20}: "
        f"{ground_truth_total[class_name]}"
    )


print("\n" + "-" * 70)
print("PREDICTED BOX COUNTS")
print("-" * 70)

for class_name in CLASS_NAMES:

    print(
        f"{class_name:<20}: "
        f"{prediction_total[class_name]}"
    )


print("\n" + "-" * 70)
print("AVERAGE PREDICTION CONFIDENCE")
print("-" * 70)

for class_name in CLASS_NAMES:

    stats = average_confidence[class_name]

    if stats["count"] > 0:

        print(
            f"{class_name:<20}: "
            f"avg={stats['average']:.3f} | "
            f"min={stats['minimum']:.3f} | "
            f"max={stats['maximum']:.3f} | "
            f"n={stats['count']}"
        )

    else:

        print(
            f"{class_name:<20}: "
            f"No predictions"
        )


print("\n" + "-" * 70)
print("GROUND-TRUTH BOXES IN IMAGES WITH ZERO PREDICTIONS")
print("-" * 70)

for class_name in CLASS_NAMES:

    print(
        f"{class_name:<20}: "
        f"{no_detection_by_class[class_name]}"
    )


# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

analysis = {

    "validation_images": len(image_paths),

    "images_with_predictions":
        images_with_predictions,

    "images_without_predictions":
        images_without_predictions,

    "total_predictions":
        total_predictions,

    "ground_truth_boxes":
        dict(ground_truth_total),

    "predicted_boxes":
        dict(prediction_total),

    "average_confidence":
        average_confidence,

    "ground_truth_boxes_in_zero_prediction_images":
        dict(no_detection_by_class),

}


output_json = OUTPUT_DIR / "yolo11n_error_analysis.json"

with open(output_json, "w") as f:

    json.dump(
        analysis,
        f,
        indent=4
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(output_json)