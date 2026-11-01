import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# RESNET50 MODEL EVALUATION
# ============================================================

print("=" * 60)
print("RESNET50 MODEL EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\Admin\Downloads\NEU_FYP"

DATASET_ROOT = r"C:\Users\Admin\Downloads\archive\NEU-DET"

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "validation",
    "images"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results",
    "resnet50_baseline.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# ------------------------------------------------------------
# TENSORFLOW INFORMATION
# ------------------------------------------------------------

print(f"\nTensorFlow version: {tf.__version__}")


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

print("\nLoading trained ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ------------------------------------------------------------
# LOAD VALIDATION DATASET
# ------------------------------------------------------------

print("\nLoading validation dataset...")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_dataset.class_names

print(f"Validation images: {len(val_dataset.file_paths)}")

print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"  {index} -> {class_name}")


# ------------------------------------------------------------
# GENERATE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(
    val_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ------------------------------------------------------------
# GET TRUE LABELS
# ------------------------------------------------------------

true_labels = np.concatenate(
    [
        labels.numpy()
        for _, labels in val_dataset
    ]
)


# ------------------------------------------------------------
# PREDICTION CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION CHECK")
print("=" * 60)

print(f"True labels      : {len(true_labels)}")
print(f"Predictions      : {len(predicted_labels)}")


# ------------------------------------------------------------
# OVERALL ACCURACY
# ------------------------------------------------------------

accuracy = np.mean(
    true_labels == predicted_labels
)

print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print(report_df)


# ------------------------------------------------------------
# SAVE CLASSIFICATION REPORT
# ------------------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_classification_report.csv"
)

report_df.to_csv(
    report_path
)

print("\nClassification report saved to:")
print(report_path)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)


# ------------------------------------------------------------
# PLOT CONFUSION MATRIX
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title(
    "ResNet50 Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()


# ------------------------------------------------------------
# SAVE CONFUSION MATRIX
# ------------------------------------------------------------

confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to:")
print(confusion_matrix_path)


# ------------------------------------------------------------
# PER-CLASS SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS SUMMARY")
print("=" * 60)

for class_name in class_names:

    precision = report[class_name]["precision"]
    recall = report[class_name]["recall"]
    f1 = report[class_name]["f1-score"]
    support = int(report[class_name]["support"])

    print(f"\n{class_name}")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print(f"  F1-score  : {f1 * 100:.2f}%")
    print(f"  Images    : {support}")


# ------------------------------------------------------------
# COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(
    f"1. {report_path}"
)

print(
    f"2. {confusion_matrix_path}"
)

print("\nResNet50 evaluation is complete.")