from pathlib import Path
from collections import defaultdict
import json
import statistics


# ============================================================
# DETECTION BOUNDING-BOX STATISTICS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Admin\Downloads\NEU_FYP")

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
    / "dataset_analysis"
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

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

class_stats = defaultdict(lambda: {
    "box_count": 0,
    "image_count": 0,
    "widths": [],
    "heights": [],
    "areas": [],
    "area_percentages": [],
})

boxes_per_image = defaultdict(list)


# ------------------------------------------------------------
# Read validation annotations
# ------------------------------------------------------------

label_files = sorted(LABEL_DIR.glob("*.txt"))

print("=" * 70)
print("DETECTION BOUNDING-BOX STATISTICS")
print("=" * 70)

print(f"Label directory : {LABEL_DIR}")
print(f"Label files     : {len(label_files)}")


for label_file in label_files:

    image_classes = set()
    image_box_count = 0

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue

            class_name = CLASS_NAMES[class_id]

            # Convert normalized YOLO dimensions
            width = width_norm * IMAGE_WIDTH
            height = height_norm * IMAGE_HEIGHT

            area = width * height
            area_percentage = (area / IMAGE_AREA) * 100

            class_stats[class_name]["box_count"] += 1

            class_stats[class_name]["widths"].append(width)
            class_stats[class_name]["heights"].append(height)
            class_stats[class_name]["areas"].append(area)
            class_stats[class_name]["area_percentages"].append(
                area_percentage
            )

            image_classes.add(class_name)
            image_box_count += 1

    for class_name in image_classes:
        class_stats[class_name]["image_count"] += 1

    boxes_per_image[label_file.stem].append(image_box_count)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("PER-CLASS BOX STATISTICS")
print("-" * 70)

results = {}


for class_name in CLASS_NAMES:

    stats = class_stats[class_name]

    if stats["box_count"] == 0:
        continue

    avg_width = statistics.mean(stats["widths"])
    avg_height = statistics.mean(stats["heights"])
    avg_area = statistics.mean(stats["areas"])
    avg_area_pct = statistics.mean(stats["area_percentages"])

    median_area = statistics.median(stats["areas"])

    # Small boxes = less than 5% of image
    small_boxes = sum(
        1 for x in stats["area_percentages"]
        if x < 5
    )

    # Very small boxes = less than 2% of image
    very_small_boxes = sum(
        1 for x in stats["area_percentages"]
        if x < 2
    )

    small_percentage = (
        small_boxes / stats["box_count"]
    ) * 100

    very_small_percentage = (
        very_small_boxes / stats["box_count"]
    ) * 100

    print(f"\n{class_name.upper()}")

    print(f"  Images containing class : {stats['image_count']}")
    print(f"  Total boxes             : {stats['box_count']}")
    print(f"  Average width           : {avg_width:.2f} px")
    print(f"  Average height          : {avg_height:.2f} px")
    print(f"  Average box area        : {avg_area:.2f} px²")
    print(f"  Median box area         : {median_area:.2f} px²")
    print(f"  Average image coverage  : {avg_area_pct:.2f}%")
    print(f"  Boxes < 5% image        : {small_boxes} ({small_percentage:.2f}%)")
    print(f"  Boxes < 2% image        : {very_small_boxes} ({very_small_percentage:.2f}%)")

    results[class_name] = {
        "image_count": stats["image_count"],
        "box_count": stats["box_count"],
        "average_width_px": round(avg_width, 2),
        "average_height_px": round(avg_height, 2),
        "average_area_px2": round(avg_area, 2),
        "median_area_px2": round(median_area, 2),
        "average_image_coverage_percent": round(avg_area_pct, 2),
        "boxes_below_5_percent": small_boxes,
        "boxes_below_5_percent_ratio": round(small_percentage, 2),
        "boxes_below_2_percent": very_small_boxes,
        "boxes_below_2_percent_ratio": round(very_small_percentage, 2),
    }


# ------------------------------------------------------------
# Overall box-per-image statistics
# ------------------------------------------------------------

all_box_counts = [
    values[0]
    for values in boxes_per_image.values()
]

print("\n" + "-" * 70)
print("BOXES PER IMAGE")
print("-" * 70)

print(
    f"Average boxes/image : "
    f"{statistics.mean(all_box_counts):.2f}"
)

print(
    f"Median boxes/image  : "
    f"{statistics.median(all_box_counts):.2f}"
)

print(
    f"Maximum boxes/image : "
    f"{max(all_box_counts)}"
)


# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

output = {
    "image_size": {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
    },
    "classes": results,
    "overall": {
        "validation_images": len(label_files),
        "average_boxes_per_image":
            round(statistics.mean(all_box_counts), 3),
        "median_boxes_per_image":
            round(statistics.median(all_box_counts), 3),
        "maximum_boxes_per_image":
            max(all_box_counts),
    },
}


output_path = (
    OUTPUT_DIR
    / "detection_box_statistics.json"
)

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(output_path)