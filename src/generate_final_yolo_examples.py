from ultralytics import YOLO
from pathlib import Path
import cv2


# ============================================================
# FINAL YOLO11N DEMONSTRATION EXAMPLES
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "final_yolo_examples"
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
# One representative image per class
# ------------------------------------------------------------

selected_images = {
    "crazing": "crazing_253.jpg",
    "inclusion": "inclusion_252.jpg",
    "patches": "patches_261.jpg",
    "pitted_surface": None,
    "rolled-in_scale": "rolled-in_scale_280.jpg",
    "scratches": "scratches_287.jpg",
}


print("=" * 70)
print("FINAL YOLO11N DEMONSTRATION EXAMPLES")
print("=" * 70)

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

print("\nConfidence threshold: 0.10")

for class_name, filename in selected_images.items():

    if filename is None:
        continue

    image_path = IMAGE_DIR / filename

    if not image_path.exists():

        print(
            f"\nWARNING: {filename} not found."
        )

        continue

    print(f"\nProcessing: {filename}")

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=0.10,
        iou=0.45,
        verbose=False,
        device="cpu",
    )

    result = results[0]

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print("  No detections.")

    else:

        print(
            f"  Detections: {len(boxes)}"
        )

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id < len(CLASS_NAMES):

                print(
                    f"    {CLASS_NAMES[class_id]:<20}"
                    f" confidence={confidence:.3f}"
                )

    annotated = result.plot()

    output_path = (
        OUTPUT_DIR
        / f"{class_name}_example.jpg"
    )

    cv2.imwrite(
        str(output_path),
        annotated
    )

    print(
        f"  Saved: {output_path.name}"
    )


print("\n" + "=" * 70)
print("FINAL EXAMPLES COMPLETE")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)