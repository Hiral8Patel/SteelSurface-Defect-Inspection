import json
import os
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

HISTORY_PATH = r"C:\Users\Admin\Downloads\NEU_FYP\results\model_results\resnet50_history.json"
OUTPUT_DIR = r"C:\Users\Admin\Downloads\NEU_FYP\results\model_results"


# --------------------------------------------------
# LOAD TRAINING HISTORY
# --------------------------------------------------

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = range(1, len(history["accuracy"]) + 1)


# --------------------------------------------------
# ACCURACY CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet50 Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_accuracy_curve.png"
)

plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# LOSS CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

loss_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_loss_curve.png"
)

plt.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

best_val_accuracy = max(history["val_accuracy"])
best_val_accuracy_epoch = (
    history["val_accuracy"].index(best_val_accuracy) + 1
)

lowest_val_loss = min(history["val_loss"])
lowest_val_loss_epoch = (
    history["val_loss"].index(lowest_val_loss) + 1
)

training_time = history.get("training_time_seconds", None)

print("=" * 60)
print("RESNET50 TRAINING CURVES")
print("=" * 60)

print(f"Epochs              : {len(history['accuracy'])}")
print(
    f"Best validation acc : "
    f"{best_val_accuracy * 100:.2f}% "
    f"(Epoch {best_val_accuracy_epoch})"
)
print(
    f"Lowest validation loss : "
    f"{lowest_val_loss:.4f} "
    f"(Epoch {lowest_val_loss_epoch})"
)

if training_time is not None:
    print(
        f"Training time       : "
        f"{training_time / 60:.2f} minutes"
    )

print()
print(f"Accuracy curve saved: {accuracy_path}")
print(f"Loss curve saved    : {loss_path}")
print("=" * 60)