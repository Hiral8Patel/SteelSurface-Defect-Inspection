import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "train", "annotations")

RESULTS_PATH = r"C:\Users\Admin\Downloads\NEU_FYP\results"

FIGURES_PATH = os.path.join(RESULTS_PATH, "figures")
DATASET_ANALYSIS_PATH = os.path.join(
    RESULTS_PATH,
    "dataset_analysis"
)


# Create output folders if they don't exist
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(DATASET_ANALYSIS_PATH, exist_ok=True)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_first_image(class_name):

    class_path = os.path.join(
        TRAIN_IMAGES,
        class_name
    )

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files.sort()

    return os.path.join(
        class_path,
        image_files[0]
    )


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

class_counts = []

for class_name in CLASSES:

    class_path = os.path.join(
        TRAIN_IMAGES,
        class_name
    )

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts.append(len(image_files))


plt.figure(figsize=(10, 6))

bars = plt.bar(CLASSES, class_counts)

plt.title("NEU-DET Training Dataset - Class Distribution")
plt.xlabel("Defect Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=30)

# Display values on top of bars
for bar, count in zip(bars, class_counts):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3,
        str(count),
        ha="center"
    )

plt.tight_layout()

distribution_path = os.path.join(
    FIGURES_PATH,
    "class_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {distribution_path}")


# ============================================================
# 2. SIX SAMPLE IMAGES
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()

for index, class_name in enumerate(CLASSES):

    image_path = get_first_image(class_name)

    image = Image.open(image_path)

    axes[index].imshow(image)

    axes[index].set_title(
        class_name.replace("_", " ").title()
    )

    axes[index].axis("off")


plt.suptitle(
    "Sample Images from Each NEU-DET Defect Class",
    fontsize=16
)

plt.tight_layout()

samples_path = os.path.join(
    FIGURES_PATH,
    "sample_images.png"
)

plt.savefig(
    samples_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {samples_path}")


# ============================================================
# 3. BOUNDING BOX VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()


for index, class_name in enumerate(CLASSES):

    image_path = get_first_image(class_name)

    image_filename = os.path.basename(image_path)

    xml_filename = os.path.splitext(image_filename)[0] + ".xml"

    xml_path = os.path.join(
        TRAIN_ANNOTATIONS,
        xml_filename
    )

    image = Image.open(image_path).convert("RGB")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_element = root.find("object")

    class_from_xml = object_element.find("name").text

    bbox = object_element.find("bndbox")

    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [xmin, ymin, xmax, ymax],
        outline="red",
        width=2
    )

    axes[index].imshow(image)

    axes[index].set_title(
        f"{class_from_xml.replace('_', ' ').title()}\n"
        f"Box: ({xmin}, {ymin}) - ({xmax}, {ymax})"
    )

    axes[index].axis("off")


plt.suptitle(
    "NEU-DET Defect Bounding-Box Annotations",
    fontsize=16
)

plt.tight_layout()

bbox_path = os.path.join(
    FIGURES_PATH,
    "bounding_boxes.png"
)

plt.savefig(
    bbox_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {bbox_path}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("DATASET VISUALIZATION COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print("1.", distribution_path)
print("2.", samples_path)
print("3.", bbox_path)