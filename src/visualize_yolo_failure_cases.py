from ultralytics import YOLO
from pathlib import Path
import cv2
import random


# ============================================================
# YOLO11N FAILURE-CASE VISUALIZATION
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
    / "figures"
    / "yolo_failure_cases"
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
# Read ground-truth boxes
# ------------------------------------------------------------

def read_labels(label_path):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            boxes.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    return boxes


# ------------------------------------------------------------
# Draw ground truth
# ------------------------------------------------------------

def draw_ground_truth(image, boxes):

    output = image.copy()

    h, w = output.shape[:2]

    for class_id, xc, yc, bw, bh in boxes:

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        class_name = CLASS_NAMES[class_id]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            output,
            f"GT: {class_name}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("=" * 70)
print("YOLO11N FAILURE-CASE VISUALIZATION")
print("=" * 70)

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")


# ------------------------------------------------------------
# Find crazing validation images
# ------------------------------------------------------------

crazing_images = []

for image_path in sorted(IMAGE_DIR.glob("*.jpg")):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    boxes = read_labels(label_path)

    contains_crazing = any(
        class_id == 0
        for class_id, *_ in boxes
    )

    if contains_crazing:
        crazing_images.append(image_path)


print(
    f"\nCrazing validation images found: "
    f"{len(crazing_images)}"
)


# ------------------------------------------------------------
# Find failure cases
# ------------------------------------------------------------

zero_prediction_cases = []
wrong_class_cases = []

print("\nSearching for difficult crazing cases...\n")


for image_path in crazing_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_boxes = read_labels(label_path)

    image = cv2.imread(str(image_path))

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device="cpu",
    )

    result = results[0]

    boxes = result.boxes

    prediction_count = 0
    predicted_classes = []

    if boxes is not None and len(boxes) > 0:

        prediction_count = len(boxes)

        for box in boxes:

            class_id = int(box.cls[0])

            if class_id < len(CLASS_NAMES):

                predicted_classes.append(
                    CLASS_NAMES[class_id]
                )

    if prediction_count == 0:

        zero_prediction_cases.append(
            image_path
        )

    elif "crazing" not in predicted_classes:

        wrong_class_cases.append(
            image_path
        )


# ------------------------------------------------------------
# Select cases
# ------------------------------------------------------------

random.seed(42)

zero_selected = random.sample(
    zero_prediction_cases,
    min(6, len(zero_prediction_cases))
)

wrong_selected = random.sample(
    wrong_class_cases,
    min(6, len(wrong_class_cases))
)


print(
    f"Zero-prediction crazing cases : "
    f"{len(zero_prediction_cases)}"
)

print(
    f"Wrong-class/no-crazing cases  : "
    f"{len(wrong_class_cases)}"
)


# ------------------------------------------------------------
# Generate visualizations
# ------------------------------------------------------------

def create_visualization(
    image_path,
    output_name,
    category
):

    image = cv2.imread(str(image_path))

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_boxes = read_labels(label_path)

    ground_truth_image = draw_ground_truth(
        image,
        gt_boxes
    )

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device="cpu",
    )

    prediction_image = results[0].plot()

    # Resize both to same dimensions
    h, w = image.shape[:2]

    ground_truth_image = cv2.resize(
        ground_truth_image,
        (w, h)
    )

    prediction_image = cv2.resize(
        prediction_image,
        (w, h)
    )

    combined = cv2.hconcat(
        [
            ground_truth_image,
            prediction_image,
        ]
    )

    title_height = 40

    canvas = cv2.copyMakeBorder(
        combined,
        title_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(40, 40, 40),
    )

    cv2.putText(
        canvas,
        f"LEFT: Ground Truth    |    RIGHT: YOLO Prediction    |    {category}",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path = OUTPUT_DIR / output_name

    cv2.imwrite(
        str(output_path),
        canvas
    )

    print(f"  {output_path.name}")


# ------------------------------------------------------------
# Generate zero-prediction cases
# ------------------------------------------------------------

print("\nGenerating zero-prediction cases...")

for index, image_path in enumerate(
    zero_selected,
    start=1
):

    create_visualization(
        image_path,
        f"crazing_zero_prediction_{index}.jpg",
        "ZERO PREDICTIONS",
    )


# ------------------------------------------------------------
# Generate wrong-class cases
# ------------------------------------------------------------

print("\nGenerating wrong-class cases...")

for index, image_path in enumerate(
    wrong_selected,
    start=1
):

    create_visualization(
        image_path,
        f"crazing_wrong_class_{index}.jpg",
        "NO CRAZING PREDICTION",
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FAILURE-CASE VISUALIZATION COMPLETE")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.glob("*.jpg")):
    print(f"  - {path.name}")