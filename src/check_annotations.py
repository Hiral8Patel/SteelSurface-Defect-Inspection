import os

DATASET_PATH = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "train", "annotations")

VAL_IMAGES = os.path.join(DATASET_PATH, "validation", "images")
VAL_ANNOTATIONS = os.path.join(DATASET_PATH, "validation", "annotations")


def get_image_names(images_path):
    image_names = set()

    for root, dirs, files in os.walk(images_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_names.add(os.path.splitext(file)[0])

    return image_names


def get_annotation_names(annotation_path):
    annotation_names = set()

    for file in os.listdir(annotation_path):
        if file.lower().endswith(".xml"):
            annotation_names.add(os.path.splitext(file)[0])

    return annotation_names


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

train_images = get_image_names(TRAIN_IMAGES)
train_annotations = get_annotation_names(TRAIN_ANNOTATIONS)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

val_images = get_image_names(VAL_IMAGES)
val_annotations = get_annotation_names(VAL_ANNOTATIONS)


# ------------------------------------------------------------
# CHECK TRAINING
# ------------------------------------------------------------

train_missing = train_images - train_annotations
train_extra = train_annotations - train_images


# ------------------------------------------------------------
# CHECK VALIDATION
# ------------------------------------------------------------

val_missing = val_images - val_annotations
val_extra = val_annotations - val_images


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("=" * 60)
print("ANNOTATION CONSISTENCY CHECK")
print("=" * 60)

print("\nTRAINING")
print("-" * 40)

print(f"Images       : {len(train_images)}")
print(f"Annotations  : {len(train_annotations)}")

print(f"\nImages WITHOUT annotation:")
print(train_missing if train_missing else "None")

print(f"\nAnnotations WITHOUT image:")
print(train_extra if train_extra else "None")


print("\nVALIDATION")
print("-" * 40)

print(f"Images       : {len(val_images)}")
print(f"Annotations  : {len(val_annotations)}")

print(f"\nImages WITHOUT annotation:")
print(val_missing if val_missing else "None")

print(f"\nAnnotations WITHOUT image:")
print(val_extra if val_extra else "None")


print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)