import os
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_IMAGES = os.path.join(
    DATASET_DIR,
    "train",
    "images"
)

TRAIN_LABELS = (
    r"C:\Users\Admin\Downloads\NEU_FYP"
    r"\results\detection_dataset\train\labels"
)

OUTPUT_DIR = (
    r"C:\Users\Admin\Downloads\NEU_FYP"
    r"\results\figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CLASSES
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
# SETTINGS
# ============================================================

NUM_EXAMPLES = 6
RANDOM_SEED = 42


# ============================================================
# FIND IMAGE FILE
# ============================================================

def find_image(filename):

    # Search through the six class folders.
    for class_name in CLASS_NAMES:

        path = os.path.join(
            TRAIN_IMAGES,
            class_name,
            filename
        )

        if os.path.exists(path):
            return path

    return None


# ============================================================
# READ YOLO LABEL
# ============================================================

def read_yolo_label(label_path):

    annotations = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    for line in lines:

        parts = line.split()

        if len(parts) != 5:

            raise ValueError(
                f"Invalid YOLO line in {label_path}: "
                f"{line}"
            )

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # ----------------------------------------------------
        # Validate normalized values
        # ----------------------------------------------------

        values = [
            x_center,
            y_center,
            box_width,
            box_height
        ]

        for value in values:

            if value < 0 or value > 1:

                raise ValueError(
                    f"Invalid normalized value "
                    f"{value} in {label_path}"
                )

        if class_id < 0 or class_id >= len(CLASS_NAMES):

            raise ValueError(
                f"Invalid class ID {class_id} "
                f"in {label_path}"
            )

        annotations.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height
            }
        )

    return annotations


# ============================================================
# CONVERT YOLO → PIXEL COORDINATES
# ============================================================

def yolo_to_pixels(
    annotation,
    image_width,
    image_height
):

    x_center = (
        annotation["x_center"]
        * image_width
    )

    y_center = (
        annotation["y_center"]
        * image_height
    )

    box_width = (
        annotation["width"]
        * image_width
    )

    box_height = (
        annotation["height"]
        * image_height
    )

    xmin = x_center - box_width / 2
    ymin = y_center - box_height / 2

    xmax = x_center + box_width / 2
    ymax = y_center + box_height / 2

    return (
        xmin,
        ymin,
        xmax,
        ymax
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("YOLO ANNOTATION VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Find label files
    # --------------------------------------------------------

    label_files = [
        file
        for file in os.listdir(TRAIN_LABELS)
        if file.lower().endswith(".txt")
    ]

    print(
        f"\nYOLO label files found : "
        f"{len(label_files)}"
    )

    # --------------------------------------------------------
    # Select random examples
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    selected_labels = random.sample(
        label_files,
        min(
            NUM_EXAMPLES,
            len(label_files)
        )
    )

    print(
        f"Examples selected     : "
        f"{len(selected_labels)}"
    )

    print("\nGenerating verification images...\n")

    # --------------------------------------------------------
    # Process each example
    # --------------------------------------------------------

    for example_number, label_filename in enumerate(
        selected_labels,
        start=1
    ):

        # ----------------------------------------------------
        # Label path
        # ----------------------------------------------------

        label_path = os.path.join(
            TRAIN_LABELS,
            label_filename
        )

        # ----------------------------------------------------
        # Image filename
        # ----------------------------------------------------

        image_filename = (
            os.path.splitext(
                label_filename
            )[0]
            + ".jpg"
        )

        image_path = find_image(
            image_filename
        )

        if image_path is None:

            print(
                f"WARNING: Image not found for "
                f"{label_filename}"
            )

            continue

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image_width, image_height = (
            image.size
        )

        # ----------------------------------------------------
        # Read annotations
        # ----------------------------------------------------

        annotations = read_yolo_label(
            label_path
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.imshow(image)

        for index, annotation in enumerate(
            annotations,
            start=1
        ):

            (
                xmin,
                ymin,
                xmax,
                ymax
            ) = yolo_to_pixels(
                annotation,
                image_width,
                image_height
            )

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            rectangle = Rectangle(
                (
                    xmin,
                    ymin
                ),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                linewidth=2
            )

            ax.add_patch(
                rectangle
            )

            # ----------------------------------------------
            # Label
            # ----------------------------------------------

            label = (
                f"{index}: "
                f"{annotation['class_name']}"
            )

            ax.text(
                xmin,
                max(ymin - 3, 5),
                label,
                fontsize=9,
                fontweight="bold",
                backgroundcolor="white"
            )

        ax.set_title(
            "YOLO Annotation Verification\n"
            f"{image_filename} | "
            f"{len(annotations)} defects"
        )

        ax.axis("off")

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"yolo_verification_{example_number}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print(
            f"Example {example_number}: "
            f"{image_filename}"
        )

        print(
            f"  Defects : "
            f"{len(annotations)}"
        )

        for annotation in annotations:

            (
                xmin,
                ymin,
                xmax,
                ymax
            ) = yolo_to_pixels(
                annotation,
                image_width,
                image_height
            )

            print(
                f"  - "
                f"{annotation['class_name']:<20} "
                f"Box: "
                f"({xmin:.1f}, "
                f"{ymin:.1f}, "
                f"{xmax:.1f}, "
                f"{ymax:.1f})"
            )

        print(
            f"  Saved   : "
            f"{output_path}"
        )

        print()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("=" * 70)
    print("YOLO VERIFICATION COMPLETE")
    print("=" * 70)

    print(
        "\nVerification images saved to:"
    )

    print(OUTPUT_DIR)