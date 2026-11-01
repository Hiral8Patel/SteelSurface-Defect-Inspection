import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, "train", "annotations")

VAL_IMAGES = os.path.join(DATASET_DIR, "validation", "images")
VAL_ANNOTATIONS = os.path.join(DATASET_DIR, "validation", "annotations")


# ============================================================
# EXPECTED CLASSES
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
# ANNOTATION ANALYSIS FUNCTION
# ============================================================

def analyze_split(split_name, image_dir, annotation_dir):

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} ANNOTATION ANALYSIS")
    print("=" * 70)

    image_files = []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(
                    os.path.join(root, file)
                )

    xml_files = [
        file for file in os.listdir(annotation_dir)
        if file.lower().endswith(".xml")
    ]

    print(f"Images found      : {len(image_files)}")
    print(f"XML annotations   : {len(xml_files)}")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    class_counter = Counter()

    boxes_per_image = Counter()

    invalid_boxes = []
    outside_boxes = []

    image_size_counter = Counter()

    total_boxes = 0

    annotation_image_names = set()
    image_names = set()

    # --------------------------------------------------------
    # Image names
    # --------------------------------------------------------

    for image_path in image_files:
        image_names.add(
            os.path.splitext(os.path.basename(image_path))[0]
        )

    for xml_file in xml_files:

        xml_path = os.path.join(
            annotation_dir,
            xml_file
        )

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

        except Exception as e:

            print(f"\nERROR reading: {xml_file}")
            print(e)
            continue

        filename_element = root.find("filename")

        if filename_element is not None:
            annotation_image_names.add(
                os.path.splitext(filename_element.text)[0]
            )

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        size_element = root.find("size")

        if size_element is not None:

            width_element = size_element.find("width")
            height_element = size_element.find("height")

            if (
                width_element is not None
                and height_element is not None
            ):

                width = int(width_element.text)
                height = int(height_element.text)

                image_size_counter[
                    (width, height)
                ] += 1

        # ----------------------------------------------------
        # Objects / Bounding boxes
        # ----------------------------------------------------

        objects = root.findall("object")

        boxes_per_image[len(objects)] += 1

        total_boxes += len(objects)

        for obj in objects:

            # ----------------------------------------------
            # Class name
            # ----------------------------------------------

            name_element = obj.find("name")

            if name_element is None:
                continue

            class_name = name_element.text.strip()

            class_counter[class_name] += 1

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            bbox = obj.find("bndbox")

            if bbox is None:
                invalid_boxes.append(
                    (xml_file, "Missing bndbox")
                )
                continue

            try:

                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

            except Exception:

                invalid_boxes.append(
                    (xml_file, "Invalid coordinate")
                )
                continue

            # ----------------------------------------------
            # Validate coordinates
            # ----------------------------------------------

            if xmin >= xmax or ymin >= ymax:

                invalid_boxes.append(
                    (
                        xml_file,
                        f"Invalid box: "
                        f"({xmin},{ymin},{xmax},{ymax})"
                    )
                )

            # ----------------------------------------------
            # Check if box exceeds image boundaries
            # ----------------------------------------------

            if size_element is not None:

                width = int(size_element.find("width").text)
                height = int(size_element.find("height").text)

                if (
                    xmin < 0
                    or ymin < 0
                    or xmax > width
                    or ymax > height
                ):

                    outside_boxes.append(
                        (
                            xml_file,
                            f"Box: "
                            f"({xmin},{ymin},{xmax},{ymax}) "
                            f"Image: {width}x{height}"
                        )
                    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n--- IMAGE / ANNOTATION MATCH ---")

    missing_annotations = image_names - annotation_image_names
    extra_annotations = annotation_image_names - image_names

    print(
        f"Images without annotation : "
        f"{len(missing_annotations)}"
    )

    print(
        f"Annotations without image : "
        f"{len(extra_annotations)}"
    )

    # --------------------------------------------------------
    # Image sizes
    # --------------------------------------------------------

    print("\n--- IMAGE DIMENSIONS FROM XML ---")

    for size, count in sorted(
        image_size_counter.items()
    ):
        print(
            f"{size[0]} x {size[1]} : {count} images"
        )

    # --------------------------------------------------------
    # Number of boxes
    # --------------------------------------------------------

    print("\n--- BOUNDING BOXES PER IMAGE ---")

    for box_count, image_count in sorted(
        boxes_per_image.items()
    ):
        print(
            f"{box_count} box(es) : "
            f"{image_count} images"
        )

    # --------------------------------------------------------
    # Total boxes
    # --------------------------------------------------------

    print("\n--- TOTAL BOUNDING BOXES ---")

    print(f"Total bounding boxes : {total_boxes}")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n--- DEFECT CLASS DISTRIBUTION ---")

    for class_name in CLASS_NAMES:

        count = class_counter[class_name]

        print(
            f"{class_name:<20} : {count}"
        )

    # --------------------------------------------------------
    # Unexpected classes
    # --------------------------------------------------------

    unexpected_classes = set(
        class_counter.keys()
    ) - set(CLASS_NAMES)

    if unexpected_classes:

        print("\n--- UNEXPECTED CLASSES ---")

        for class_name in sorted(
            unexpected_classes
        ):
            print(
                f"{class_name} : "
                f"{class_counter[class_name]}"
            )

    else:

        print("\nUnexpected classes : None")

    # --------------------------------------------------------
    # Invalid boxes
    # --------------------------------------------------------

    print("\n--- BOX VALIDATION ---")

    print(
        f"Invalid bounding boxes : "
        f"{len(invalid_boxes)}"
    )

    print(
        f"Boxes outside image   : "
        f"{len(outside_boxes)}"
    )

    # --------------------------------------------------------
    # Print examples if problems exist
    # --------------------------------------------------------

    if invalid_boxes:

        print("\nExamples of invalid boxes:")

        for item in invalid_boxes[:10]:
            print(
                f"  {item[0]} -> {item[1]}"
            )

    if outside_boxes:

        print("\nExamples of boxes outside image:")

        for item in outside_boxes[:10]:
            print(
                f"  {item[0]} -> {item[1]}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n--- SUMMARY ---")

    if (
        len(missing_annotations) == 0
        and len(extra_annotations) == 0
        and len(invalid_boxes) == 0
        and len(outside_boxes) == 0
    ):

        print("✓ Annotation structure looks valid.")

    else:

        print(
            "⚠ Some annotation issues were detected."
        )

    print(
        f"Images        : {len(image_files)}"
    )

    print(
        f"Annotations   : {len(xml_files)}"
    )

    print(
        f"Total boxes   : {total_boxes}"
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

if __name__ == "__main__":

    analyze_split(
        "Training",
        TRAIN_IMAGES,
        TRAIN_ANNOTATIONS
    )

    analyze_split(
        "Validation",
        VAL_IMAGES,
        VAL_ANNOTATIONS
    )

    print("\n" + "=" * 70)
    print("LOCALIZATION ANNOTATION ANALYSIS COMPLETE")
    print("=" * 70)