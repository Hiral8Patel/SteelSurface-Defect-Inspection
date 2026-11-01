import os
import xml.etree.ElementTree as ET
from PIL import Image

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "validation")


# ============================================================
# FUNCTION TO INSPECT ONE DATASET SPLIT
# ============================================================

def inspect_split(split_name, split_path):

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} DATASET")
    print("=" * 60)

    images_path = os.path.join(split_path, "images")
    annotations_path = os.path.join(split_path, "annotations")

    classes = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]

    total_images = 0
    total_annotations = 0

    print("\nImages per class:")
    print("-" * 40)

    for class_name in classes:

        class_image_path = os.path.join(images_path, class_name)

        if os.path.exists(class_image_path):
            image_files = [
                f for f in os.listdir(class_image_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"{class_name:20s}: {len(image_files)}")
            total_images += len(image_files)

        else:
            print(f"{class_name:20s}: FOLDER NOT FOUND")

    print("-" * 40)
    print(f"{'TOTAL IMAGES':20s}: {total_images}")

    # --------------------------------------------------------
    # Count XML annotations
    # --------------------------------------------------------

    print("\nAnnotations:")
    print("-" * 40)

    if os.path.exists(annotations_path):

        xml_files = [
            f for f in os.listdir(annotations_path)
            if f.lower().endswith(".xml")
        ]

        total_annotations = len(xml_files)

    print(f"{'XML files':20s}: {total_annotations}")

    # --------------------------------------------------------
    # Inspect first image
    # --------------------------------------------------------

    print("\nImage properties:")
    print("-" * 40)

    first_image_found = False

    for class_name in classes:

        class_image_path = os.path.join(images_path, class_name)

        if not os.path.exists(class_image_path):
            continue

        image_files = [
            f for f in os.listdir(class_image_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) > 0:

            first_image = os.path.join(
                class_image_path,
                image_files[0]
            )

            with Image.open(first_image) as img:

                print(f"Example image : {image_files[0]}")
                print(f"Dimensions    : {img.size}")
                print(f"Image mode    : {img.mode}")

            first_image_found = True
            break

    if not first_image_found:
        print("No images found.")

    # --------------------------------------------------------
    # Inspect first XML annotation
    # --------------------------------------------------------

    print("\nExample annotation:")
    print("-" * 40)

    if os.path.exists(annotations_path):

        xml_files = [
            f for f in os.listdir(annotations_path)
            if f.lower().endswith(".xml")
        ]

        if len(xml_files) > 0:

            first_xml = os.path.join(
                annotations_path,
                xml_files[0]
            )

            tree = ET.parse(first_xml)
            root = tree.getroot()

            filename = root.find("filename").text

            object_element = root.find("object")

            class_name = object_element.find("name").text

            bbox = object_element.find("bndbox")

            xmin = bbox.find("xmin").text
            ymin = bbox.find("ymin").text
            xmax = bbox.find("xmax").text
            ymax = bbox.find("ymax").text

            print(f"XML file      : {xml_files[0]}")
            print(f"Image         : {filename}")
            print(f"Class         : {class_name}")
            print(f"Bounding box  : ({xmin}, {ymin}) → ({xmax}, {ymax})")


# ============================================================
# RUN INSPECTION
# ============================================================

inspect_split("Training", TRAIN_PATH)
inspect_split("Validation", VAL_PATH)

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETE")
print("=" * 60)