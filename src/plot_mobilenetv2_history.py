import json
import os
import matplotlib.pyplot as plt


# ============================================================
# MOBILENETV2 TRAINING HISTORY PLOTS
# ============================================================

print("=" * 60)
print("MOBILENETV2 TRAINING HISTORY")
print("=" * 60)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\Admin\Downloads\NEU_FYP"

HISTORY_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results",
    "mobilenetv2_history.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD TRAINING HISTORY
# ------------------------------------------------------------

print("\nLoading training history...")

with open(HISTORY_PATH, "r") as file:
    history = json.load(file)

print("Training history loaded successfully.")

# Check available keys
print("\nAvailable metrics:")
for key in history.keys():
    print(f"  - {key}")


# ------------------------------------------------------------
# EXTRACT METRICS
# ------------------------------------------------------------

epochs = range(1, len(history["accuracy"]) + 1)

train_accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

train_loss = history["loss"]
val_loss = history["val_loss"]


# ------------------------------------------------------------
# BEST RESULTS
# ------------------------------------------------------------

best_val_accuracy = max(val_accuracy)
best_val_accuracy_epoch = val_accuracy.index(best_val_accuracy) + 1

best_val_loss = min(val_loss)
best_val_loss_epoch = val_loss.index(best_val_loss) + 1

print("\n" + "=" * 60)
print("BEST RESULTS")
print("=" * 60)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}% "
    f"(Epoch {best_val_accuracy_epoch})"
)

print(
    f"Lowest validation loss   : "
    f"{best_val_loss:.4f} "
    f"(Epoch {best_val_loss_epoch})"
)


# ============================================================
# 1. ACCURACY CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(list(epochs))
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv2_accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAccuracy curve saved to:")
print(accuracy_path)


# ============================================================
# 2. LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.title("MobileNetV2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(list(epochs))
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv2_loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nLoss curve saved to:")
print(loss_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("PLOTTING COMPLETED")
print("=" * 60)

print("\nGenerated files:")
print(f"1. {accuracy_path}")
print(f"2. {loss_path}")

print("\nMobileNetV2 training curves are ready.")