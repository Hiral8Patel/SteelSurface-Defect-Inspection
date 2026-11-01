import os
from PIL import Image


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train",
    "images"
)

VAL_PATH = os.path.join(
    DATASET_PATH,
    "validation",
    "images"
)


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

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}


# ============================================================
# LOAD IMAGE INFORMATION
# ============================================================

def collect_images(dataset_path):

    data = []

    for class_name in CLASSES:

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        image_files = [
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        image_files.sort()

        for file in image_files:

            image_path = os.path.join(
                class_path,
                file
            )

            data.append({
                "image_path": image_path,
                "class_name": class_name,
                "class_index": CLASS_TO_INDEX[class_name]
            })

    return data


# ============================================================
# LOAD TRAINING AND VALIDATION DATA
# ============================================================

train_data = collect_images(TRAIN_PATH)
val_data = collect_images(VAL_PATH)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("DATA PREPARATION CHECK")
print("=" * 60)

print("\nClass mapping:")
print("-" * 40)

for class_name, index in CLASS_TO_INDEX.items():
    print(f"{index} -> {class_name}")


print("\nTraining data:")
print("-" * 40)
print(f"Total training images: {len(train_data)}")


print("\nValidation data:")
print("-" * 40)
print(f"Total validation images: {len(val_data)}")


# ============================================================
# VERIFY IMAGE LOADING
# ============================================================

print("\nVerifying images...")
print("-" * 40)

broken_images = []

for item in train_data + val_data:

    try:

        with Image.open(item["image_path"]) as image:
            image.verify()

    except Exception:
        broken_images.append(item["image_path"])


print(f"Broken images: {len(broken_images)}")


if broken_images:

    print("\nBroken image files:")

    for image in broken_images:
        print(image)


# ============================================================
# SHOW SAMPLE RECORDS
# ============================================================

print("\nSample records:")
print("-" * 40)

for item in train_data[:5]:

    print(
        f"Image: {os.path.basename(item['image_path'])} | "
        f"Class: {item['class_name']} | "
        f"Label: {item['class_index']}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)

if len(train_data) == 1440 and len(val_data) == 360:

    print("DATASET SIZE CHECK: PASSED")

else:

    print("DATASET SIZE CHECK: CHECK REQUIRED")

if len(broken_images) == 0:

    print("IMAGE INTEGRITY CHECK: PASSED")

else:

    print("IMAGE INTEGRITY CHECK: FAILED")

print("=" * 60)