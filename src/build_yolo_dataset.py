from pathlib import Path
import shutil
import json

# ============================================================
# PATHS
# ============================================================

SOURCE_DATASET = Path(
    r"C:\Users\Admin\Downloads\archive\NEU-DET"
)

YOLO_SOURCE = Path(
    r"C:\Users\Admin\Downloads\NEU_FYP\results\detection_dataset"
)

OUTPUT_DATASET = YOLO_SOURCE / "yolo_dataset"

# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in ["train", "validation"]:
    (OUTPUT_DATASET / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DATASET / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

print("=" * 70)
print("BUILDING YOLO DATASET")
print("=" * 70)

total_images = {
    "train": 0,
    "validation": 0
}

total_labels = {
    "train": 0,
    "validation": 0
}

# ============================================================
# PROCESS TRAIN + VALIDATION
# ============================================================

for split in ["train", "validation"]:

    print()
    print(f"Processing: {split.upper()}")
    print("-" * 40)

    source_images = SOURCE_DATASET / split / "images"
    source_labels = YOLO_SOURCE / split / "labels"

    output_images = OUTPUT_DATASET / "images" / split
    output_labels = OUTPUT_DATASET / "labels" / split

    # --------------------------------------------------------
    # Process every class folder
    # --------------------------------------------------------

    for class_folder in sorted(source_images.iterdir()):

        if not class_folder.is_dir():
            continue

        for image_path in sorted(class_folder.glob("*.jpg")):

            image_name = image_path.name
            label_name = image_path.stem + ".txt"

            label_path = source_labels / label_name

            if not label_path.exists():
                print(f"WARNING: Missing label: {label_name}")
                continue

            # ------------------------------------------------
            # Create hard link for image
            # ------------------------------------------------

            output_image = output_images / image_name

            if not output_image.exists():
                output_image.hardlink_to(image_path)

            # ------------------------------------------------
            # Copy YOLO label
            # ------------------------------------------------

            output_label = output_labels / label_name

            if not output_label.exists():
                shutil.copy2(label_path, output_label)

            total_images[split] += 1
            total_labels[split] += 1

    print(f"Images processed : {total_images[split]}")
    print(f"Labels processed : {total_labels[split]}")


# ============================================================
# CREATE DATASET YAML
# ============================================================

yaml_content = """path: C:/Users/Admin/Downloads/NEU_FYP/results/detection_dataset/yolo_dataset

train: images/train
val: images/validation

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""

yaml_path = OUTPUT_DATASET / "neu_defects.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")

# ============================================================
# CREATE SUMMARY
# ============================================================

summary = {
    "train_images": total_images["train"],
    "train_labels": total_labels["train"],
    "validation_images": total_images["validation"],
    "validation_labels": total_labels["validation"],
    "classes": CLASS_NAMES,
    "image_storage_method": "Windows hard links",
    "source_dataset": str(SOURCE_DATASET),
}

summary_path = OUTPUT_DATASET / "dataset_build_summary.json"

summary_path.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8"
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("YOLO DATASET BUILD COMPLETE")
print("=" * 70)

print(f"Train images       : {total_images['train']}")
print(f"Train labels       : {total_labels['train']}")
print(f"Validation images  : {total_images['validation']}")
print(f"Validation labels  : {total_labels['validation']}")

print()
print("Dataset location:")
print(OUTPUT_DATASET)

print()
print("YAML file:")
print(yaml_path)

print()
print("Images use Windows hard links.")
print("No duplicate image data was created.")
print("=" * 70)