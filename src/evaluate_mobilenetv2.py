import os
import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Admin\Downloads\archive\NEU-DET"

VAL_PATH = os.path.join(
    DATASET_PATH,
    "validation",
    "images"
)

MODEL_PATH = (
    r"C:\Users\Admin\Downloads\NEU_FYP"
    r"\results\model_results\mobilenetv2_baseline.keras"
)

RESULTS_PATH = (
    r"C:\Users\Admin\Downloads\NEU_FYP"
    r"\results\model_results"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_PATH,
    "mobilenetv2_confusion_matrix.png"
)

REPORT_PATH = os.path.join(
    RESULTS_PATH,
    "mobilenetv2_classification_report.csv"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("MOBILENETV2 MODEL EVALUATION")
print("=" * 60)

print("\nTensorFlow version:", tf.__version__)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained MobileNetV2 model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation images: 360")

# ============================================================
# PREPROCESSING
# ============================================================

preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

validation_dataset = validation_dataset.map(
    lambda images, labels:
        (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    validation_dataset,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# ============================================================
# GET TRUE LABELS
# ============================================================

true_classes = []

for images, labels in validation_dataset:
    true_classes.extend(labels.numpy())

true_classes = np.array(true_classes)

# ============================================================
# CHECK PREDICTION COUNT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION CHECK")
print("=" * 60)

print("True labels      :", len(true_classes))
print("Predictions      :", len(predicted_classes))

# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True
)

report_dataframe = pd.DataFrame(report).transpose()

print(
    report_dataframe.to_string()
)

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dataframe.to_csv(
    REPORT_PATH
)

print(
    "\nClassification report saved to:"
)

print(REPORT_PATH)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title(
    "MobileNetV2 Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

# Add numbers inside cells
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved to:"
)

print(CONFUSION_MATRIX_PATH)

# ============================================================
# PER-CLASS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS SUMMARY")
print("=" * 60)

for class_name in CLASS_NAMES:

    precision = report[class_name]["precision"]
    recall = report[class_name]["recall"]
    f1 = report[class_name]["f1-score"]
    support = report[class_name]["support"]

    print(
        f"\n{class_name}"
    )

    print(
        f"  Precision : {precision * 100:.2f}%"
    )

    print(
        f"  Recall    : {recall * 100:.2f}%"
    )

    print(
        f"  F1-score  : {f1 * 100:.2f}%"
    )

    print(
        f"  Images    : {int(support)}"
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(
    "1.",
    REPORT_PATH
)

print(
    "2.",
    CONFUSION_MATRIX_PATH
)

print("\nMobileNetV2 evaluation is complete.")