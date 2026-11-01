import os
import time
import json
import tensorflow as tf

# Access Keras through TensorFlow's public API. This avoids IDE resolver
# failures on TensorFlow installations that do not expose tensorflow.keras
# as a separately discoverable source package.
keras = tf.keras
layers = keras.layers
MobileNetV2 = keras.applications.MobileNetV2

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Admin\Downloads\archive\NEU-DET"

TRAIN_PATH = os.path.join(DATASET_PATH, "train", "images")
VAL_PATH = os.path.join(DATASET_PATH, "validation", "images")

RESULTS_PATH = r"C:\Users\Admin\Downloads\NEU_FYP\results\model_results"
MODEL_PATH = os.path.join(RESULTS_PATH, "mobilenetv2_baseline.keras")
HISTORY_PATH = os.path.join(RESULTS_PATH, "mobilenetv2_history.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 6
EPOCHS = 15

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Create results directory
os.makedirs(RESULTS_PATH, exist_ok=True)

# ============================================================
# ENVIRONMENT CHECK
# ============================================================

print("=" * 60)
print("MOBILENETV2 BASELINE TRAINING")
print("=" * 60)

print("\nTensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("GPU detected: None")
    print("Training will use CPU.")

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# PREPROCESSING
# ============================================================

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10)
])

# Apply preprocessing
train_dataset = train_dataset.map(
    lambda images, labels: (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained backbone
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print("Total parameters     :", model.count_params())
print(
    "Trainable parameters :",
    sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
)
print(
    "Frozen parameters    :",
    sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
)

print("\nTraining images   : 1440")
print("Validation images : 360")
print("Batch size        :", BATCH_SIZE)
print("Maximum epochs    :", EPOCHS)

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

end_time = time.time()

training_time = end_time - start_time

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

history_data["training_time_seconds"] = training_time

with open(HISTORY_PATH, "w") as file:
    json.dump(history_data, file, indent=4)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Training time: {training_time / 60:.2f} minutes"
)

print(
    f"Best validation accuracy: "
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)

print(
    f"Best validation loss: "
    f"{min(history.history['val_loss']):.4f}"
)

print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved training history:")
print(HISTORY_PATH)

print("\n" + "=" * 60)
print("NEXT STEP")
print("=" * 60)

print("We will evaluate the trained model using:")
print("1. Accuracy")
print("2. Precision")
print("3. Recall")
print("4. F1-score")
print("5. Confusion matrix")