import os
import json
import xml.etree.ElementTree as ET
from collections import Counter


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\Admin\Downloads\archive\NEU-DET"

OUTPUT_DIR = r"C:\Users\Admin\Downloads\NEU_FYP\results\detection_dataset"

TRAIN_ANNOTATIONS = os.path.join(
    DATASET_DIR,
    "train",
    "annotations"
)

VAL_ANNOTATIONS = os.path.join(
    DATASET_DIR,
    "validation",
    "annotations"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_TO_ID = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARSE ONE XML FILE
# ============================================================

def parse_annotation(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    filename_element = root.find("filename")

    size_element = root.find("size")

    width = int(
        size_element.find("width").text
    )

    height = int(
        size_element.find("height").text
    )

    filename = filename_element.text

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    objects = []

    for obj in root.findall("object"):

        name_element = obj.find("name")

        bbox = obj.find("bndbox")

        if (
            name_element is None
            or bbox is None
        ):
            continue

        class_name = name_element.text.strip()

        if class_name not in CLASS_TO_ID:

            raise ValueError(
                f"Unknown class '{class_name}' "
                f"in {xml_path}"
            )

        xmin = int(
            bbox.find("xmin").text
        )

        ymin = int(
            bbox.find("ymin").text
        )

        xmax = int(
            bbox.find("xmax").text
        )

        ymax = int(
            bbox.find("ymax").text
        )

        # ----------------------------------------------------
        # Validate bounding box
        # ----------------------------------------------------

        if xmin < 0 or ymin < 0:
            raise ValueError(
                f"Negative coordinate in {xml_path}"
            )

        if xmax > width or ymax > height:
            raise ValueError(
                f"Bounding box outside image "
                f"in {xml_path}"
            )

        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                f"Invalid bounding box "
                f"in {xml_path}"
            )

        # ----------------------------------------------------
        # Normalized coordinates
        #
        # x_center, y_center, box_width, box_height
        #
        # Values are between 0 and 1.
        # ----------------------------------------------------

        box_width = xmax - xmin
        box_height = ymax - ymin

        x_center = (
            xmin + xmax
        ) / 2.0 / width

        y_center = (
            ymin + ymax
        ) / 2.0 / height

        normalized_width = (
            box_width / width
        )

        normalized_height = (
            box_height / height
        )

        objects.append(
            {
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],

                "bbox": [
                    xmin,
                    ymin,
                    xmax,
                    ymax
                ],

                "normalized_bbox": [
                    round(x_center, 6),
                    round(y_center, 6),
                    round(normalized_width, 6),
                    round(normalized_height, 6)
                ]
            }
        )

    # --------------------------------------------------------
    # Return structured annotation
    # --------------------------------------------------------

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    annotation_dir
):

    print("\n" + "=" * 70)
    print(
        f"{split_name.upper()} DETECTION DATA PREPARATION"
    )
    print("=" * 70)

    xml_files = sorted(
        [
            file
            for file in os.listdir(annotation_dir)
            if file.lower().endswith(".xml")
        ]
    )

    annotations = []

    class_counter = Counter()

    total_boxes = 0

    image_box_counter = Counter()

    # --------------------------------------------------------
    # Process XML files
    # --------------------------------------------------------

    for xml_file in xml_files:

        xml_path = os.path.join(
            annotation_dir,
            xml_file
        )

        annotation = parse_annotation(
            xml_path
        )

        annotations.append(
            annotation
        )

        number_of_boxes = len(
            annotation["objects"]
        )

        image_box_counter[
            number_of_boxes
        ] += 1

        total_boxes += number_of_boxes

        for obj in annotation["objects"]:

            class_counter[
                obj["class_name"]
            ] += 1

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name.lower()}_annotations.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            annotations,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"XML files processed : "
        f"{len(xml_files)}"
    )

    print(
        f"Images represented  : "
        f"{len(annotations)}"
    )

    print(
        f"Total bounding boxes: "
        f"{total_boxes}"
    )

    print("\nBounding boxes per image:")

    for box_count, image_count in sorted(
        image_box_counter.items()
    ):

        print(
            f"  {box_count} box(es) : "
            f"{image_count} images"
        )

    print("\nClass distribution:")

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<20} : "
            f"{class_counter[class_name]}"
        )

    print(
        f"\nSaved annotations:"
    )

    print(output_file)

    return {
        "images": len(annotations),
        "boxes": total_boxes,
        "class_distribution": dict(
            class_counter
        )
    }


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

def save_class_mapping():

    mapping_path = os.path.join(
        OUTPUT_DIR,
        "class_mapping.json"
    )

    mapping = {
        "classes": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "id_to_class": {
            str(index): class_name
            for index, class_name
            in enumerate(CLASS_NAMES)
        }
    }

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mapping,
            f,
            indent=2
        )

    print(
        f"\nClass mapping saved:"
    )

    print(mapping_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEU-DET DETECTION DATASET PREPARATION")
    print("=" * 70)

    print("\nClass mapping:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_id} -> {class_name}"
        )

    # --------------------------------------------------------
    # Process training data
    # --------------------------------------------------------

    train_summary = process_split(
        "train",
        TRAIN_ANNOTATIONS
    )

    # --------------------------------------------------------
    # Process validation data
    # --------------------------------------------------------

    validation_summary = process_split(
        "validation",
        VAL_ANNOTATIONS
    )

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    save_class_mapping()

    # --------------------------------------------------------
    # Save overall summary
    # --------------------------------------------------------

    summary = {
        "dataset": "NEU-DET",
        "classes": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "train": train_summary,
        "validation": validation_summary
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "detection_dataset_summary.json"
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
    print("DETECTION DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput directory:"
    )

    print(OUTPUT_DIR)

    print("\nFiles created:")

    print("  train_annotations.json")
    print("  validation_annotations.json")
    print("  class_mapping.json")
    print("  detection_dataset_summary.json")

    print("\n" + "=" * 70)