import os
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, "train", "annotations")

OUTPUT_DIR = r"C:\Users\Admin\Downloads\NEU_FYP\results\figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Number of examples to generate
NUM_EXAMPLES = 6

# Fixed seed so the same images are selected every time
RANDOM_SEED = 42


# ============================================================
# CLASS NAMES
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
# FIND MULTI-DEFECT IMAGES
# ============================================================

def find_multi_defect_images():

    candidates = []

    for xml_file in os.listdir(TRAIN_ANNOTATIONS):

        if not xml_file.lower().endswith(".xml"):
            continue

        xml_path = os.path.join(
            TRAIN_ANNOTATIONS,
            xml_file
        )

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

        except Exception:
            continue

        objects = root.findall("object")

        # We specifically want images with
        # two or more annotated defects.
        if len(objects) >= 2:

            filename_element = root.find("filename")

            if filename_element is None:
                continue

            image_filename = filename_element.text

            image_path = os.path.join(
                TRAIN_IMAGES,
                image_filename
            )

            # If images are stored inside class folders,
            # search for the actual file.
            if not os.path.exists(image_path):

                image_path = None

                for class_name in CLASS_NAMES:

                    possible_path = os.path.join(
                        TRAIN_IMAGES,
                        class_name,
                        image_filename
                    )

                    if os.path.exists(possible_path):
                        image_path = possible_path
                        break

            if image_path is not None:

                candidates.append(
                    (xml_path, image_path)
                )

    return candidates


# ============================================================
# READ ANNOTATION
# ============================================================

def read_annotation(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for obj in root.findall("object"):

        name_element = obj.find("name")
        bbox = obj.find("bndbox")

        if (
            name_element is None
            or bbox is None
        ):
            continue

        class_name = name_element.text.strip()

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        annotations.append(
            {
                "class": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            }
        )

    return annotations


# ============================================================
# VISUALIZE ONE IMAGE
# ============================================================

def visualize_image(
    xml_path,
    image_path,
    example_number
):

    annotations = read_annotation(xml_path)

    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(image)

    for index, annotation in enumerate(
        annotations,
        start=1
    ):

        xmin = annotation["xmin"]
        ymin = annotation["ymin"]
        xmax = annotation["xmax"]
        ymax = annotation["ymax"]

        class_name = annotation["class"]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        rectangle = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=2
        )

        ax.add_patch(rectangle)

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = f"{index}: {class_name}"

        ax.text(
            xmin,
            max(ymin - 3, 5),
            label,
            fontsize=9,
            fontweight="bold",
            backgroundcolor="white"
        )

    ax.set_title(
        f"NEU-DET Localization Annotation\n"
        f"{os.path.basename(image_path)} | "
        f"{len(annotations)} defects"
    )

    ax.axis("off")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"localization_example_{example_number}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEU-DET LOCALIZATION ANNOTATION VISUALIZATION")
    print("=" * 70)

    candidates = find_multi_defect_images()

    print(
        f"\nMulti-defect images found : "
        f"{len(candidates)}"
    )

    if len(candidates) == 0:

        print(
            "\nNo multi-defect images were found."
        )

        raise SystemExit

    # --------------------------------------------------------
    # Random selection
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    number_to_select = min(
        NUM_EXAMPLES,
        len(candidates)
    )

    selected = random.sample(
        candidates,
        number_to_select
    )

    print(
        f"Examples selected         : "
        f"{number_to_select}"
    )

    print("\nGenerating visualizations...\n")

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    for example_number, (
        xml_path,
        image_path
    ) in enumerate(
        selected,
        start=1
    ):

        annotations = read_annotation(
            xml_path
        )

        output_path = visualize_image(
            xml_path,
            image_path,
            example_number
        )

        print(
            f"Example {example_number}: "
            f"{os.path.basename(image_path)}"
        )

        print(
            f"  Defects : {len(annotations)}"
        )

        for annotation in annotations:

            print(
                f"  - {annotation['class']:<20} "
                f"Box: "
                f"({annotation['xmin']}, "
                f"{annotation['ymin']}, "
                f"{annotation['xmax']}, "
                f"{annotation['ymax']})"
            )

        print(
            f"  Saved   : {output_path}"
        )
        print()

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nFigures saved to:"
    )

    print(OUTPUT_DIR)