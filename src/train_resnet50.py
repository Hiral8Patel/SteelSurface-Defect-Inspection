import os
import json
import time
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
ResNet50 = tf.keras.applications.ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input


# ============================================================
# RESNET50 TRAINING
# ============================================================

print("=" * 60)
print("RESNET50 TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\Admin\Downloads\NEU_FYP"

DATASET_ROOT = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train",
    "images"
)

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "validation",
    "images"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 6
EPOCHS = 15
SEED = 42


# ------------------------------------------------------------
# DISPLAY TENSORFLOW INFORMATION
# ------------------------------------------------------------

print(f"\nTensorFlow version: {tf.__version__}")

print("Checking GPU availability...")

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"GPUs found: {gpus}")
else:
    print("No GPU detected. Training will use CPU.")


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"  {index} -> {class_name}")


# ------------------------------------------------------------
# DATASET PERFORMANCE
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)


# ------------------------------------------------------------
# BUILD RESNET50
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BUILDING RESNET50 MODEL")
print("=" * 60)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers for baseline experiment
base_model.trainable = False


inputs = layers.Input(
    shape=(224, 224, 3),
    name="input_image"
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="predictions"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="ResNet50_Baseline"
)


# ------------------------------------------------------------
# COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# MODEL SUMMARY
# ------------------------------------------------------------

model.summary()


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "resnet50_baseline.keras"
)

history_path = os.path.join(
    MODEL_DIR,
    "resnet50_history.json"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    mode="min",
    verbose=1
)


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING RESNET50 TRAINING")
print("=" * 60)

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

training_time = time.time() - start_time


# ------------------------------------------------------------
# SAVE TRAINING HISTORY
# ------------------------------------------------------------

history_data = history.history

history_data["training_time_seconds"] = training_time

with open(history_path, "w") as file:
    json.dump(
        history_data,
        file,
        indent=4
    )


# ------------------------------------------------------------
# TRAINING SUMMARY
# ------------------------------------------------------------

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_accuracy_epoch = (
    history.history["val_accuracy"].index(
        best_val_accuracy
    ) + 1
)

print("\n" + "=" * 60)
print("RESNET50 TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nBest validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best validation accuracy epoch : "
    f"{best_val_accuracy_epoch}"
)

print(
    f"Training time : "
    f"{training_time / 60:.2f} minutes"
)

print("\nGenerated files:")

print(
    f"1. {model_path}"
)

print(
    f"2. {history_path}"
)

print("\nResNet50 training is complete.")