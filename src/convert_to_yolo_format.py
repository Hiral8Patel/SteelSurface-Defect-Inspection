import os
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Admin\Downloads\NEU_FYP"

DETECTION_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "detection_dataset"
)

TRAIN_JSON = os.path.join(
    DETECTION_DIR,
    "train_annotations.json"
)

VALIDATION_JSON = os.path.join(
    DETECTION_DIR,
    "validation_annotations.json"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

TRAIN_LABELS_DIR = os.path.join(
    DETECTION_DIR,
    "train",
    "labels"
)

VALIDATION_LABELS_DIR = os.path.join(
    DETECTION_DIR,
    "validation",
    "labels"
)

os.makedirs(
    TRAIN_LABELS_DIR,
    exist_ok=True
)

os.makedirs(
    VALIDATION_LABELS_DIR,
    exist_ok=True
)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ============================================================
# CONVERSION FUNCTION
# ============================================================

def convert_annotations(
    json_path,
    output_dir,
    split_name
):

    print("\n" + "=" * 70)
    print(
        f"CONVERTING {split_name.upper()} "
        f"ANNOTATIONS TO YOLO FORMAT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        annotations = json.load(f)

    total_images = 0
    total_boxes = 0

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for annotation in annotations:

        filename = annotation["filename"]

        image_width = annotation["width"]
        image_height = annotation["height"]

        objects = annotation["objects"]

        # ----------------------------------------------------
        # Create label filename
        # ----------------------------------------------------

        image_name = os.path.splitext(
            filename
        )[0]

        label_filename = (
            image_name + ".txt"
        )

        label_path = os.path.join(
            output_dir,
            label_filename
        )

        # ----------------------------------------------------
        # Write YOLO annotations
        # ----------------------------------------------------

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as label_file:

            for obj in objects:

                class_id = obj["class_id"]

                xmin = obj["bbox"][0]
                ymin = obj["bbox"][1]
                xmax = obj["bbox"][2]
                ymax = obj["bbox"][3]

                # --------------------------------------------
                # Convert Pascal VOC
                # to YOLO normalized format
                # --------------------------------------------

                box_width = xmax - xmin
                box_height = ymax - ymin

                x_center = (
                    (xmin + xmax) / 2.0
                ) / image_width

                y_center = (
                    (ymin + ymax) / 2.0
                ) / image_height

                normalized_width = (
                    box_width / image_width
                )

                normalized_height = (
                    box_height / image_height
                )

                # --------------------------------------------
                # Validate normalized values
                # --------------------------------------------

                values = [
                    x_center,
                    y_center,
                    normalized_width,
                    normalized_height
                ]

                for value in values:

                    if value < 0 or value > 1:

                        raise ValueError(
                            f"Invalid normalized "
                            f"coordinate in {label_filename}: "
                            f"{value}"
                        )

                # --------------------------------------------
                # Write line
                # --------------------------------------------

                label_file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{normalized_width:.6f} "
                    f"{normalized_height:.6f}\n"
                )

                total_boxes += 1

                class_name = CLASS_NAMES[
                    class_id
                ]

                class_counts[
                    class_name
                ] += 1

        total_images += 1

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Images converted : {total_images}"
    )

    print(
        f"Boxes converted  : {total_boxes}"
    )

    print("\nClass distribution:")

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<20} : "
            f"{class_counts[class_name]}"
        )

    print(
        f"\nLabel directory:"
    )

    print(output_dir)

    return {
        "images": total_images,
        "boxes": total_boxes,
        "class_counts": class_counts
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEU-DET → YOLO FORMAT CONVERSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_summary = convert_annotations(
        TRAIN_JSON,
        TRAIN_LABELS_DIR,
        "train"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_summary = convert_annotations(
        VALIDATION_JSON,
        VALIDATION_LABELS_DIR,
        "validation"
    )

    # --------------------------------------------------------
    # Save conversion summary
    # --------------------------------------------------------

    summary = {
        "format": "YOLO",
        "classes": CLASS_NAMES,
        "train": train_summary,
        "validation": validation_summary
    }

    summary_path = os.path.join(
        DETECTION_DIR,
        "yolo_conversion_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("YOLO CONVERSION COMPLETE")
    print("=" * 70)

    print(
        "\nTraining labels:"
    )

    print(TRAIN_LABELS_DIR)

    print(
        "\nValidation labels:"
    )

    print(VALIDATION_LABELS_DIR)

    print(
        "\nConversion summary:"
    )

    print(summary_path)

    print("\n" + "=" * 70)