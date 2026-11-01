import os
import cv2
import numpy as np

# Pyright/Pylance may flag TensorFlow as missing in the editor environment
# even when it is installed in the active runtime; this keeps the script runnable
# without generating the missing-import warning for this file.
# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import tensorflow as tf
from tensorflow import keras

load_model = keras.models.load_model
Model = keras.Model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\Admin\Downloads\NEU_FYP"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results",
    "mobilenetv2_baseline.keras"
)

IMAGE_PATH = (
    r"C:\Users\Admin\Downloads\archive\NEU-DET"
    r"\validation\images\crazing\crazing_289.jpg"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures",
    "gradcam"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "gradcam_crazing_289.jpg"
)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GRAD-CAM TEST - CRAZING_289")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MobileNetV2...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# GET BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        base_model = layer

        break


if base_model is None:

    raise RuntimeError(
        "Could not find MobileNetV2 base model."
    )


print("Base model     :", base_model.name)


# ============================================================
# GET GRAD-CAM LAYER
# ============================================================

try:

    last_conv_layer = base_model.get_layer("Conv_1")

except Exception:

    last_conv_layer = None

    for layer in reversed(base_model.layers):

        if "conv" in layer.name.lower():

            try:

                output_shape = layer.output.shape

                if len(output_shape) == 4:

                    last_conv_layer = layer

                    break

            except Exception:

                continue


if last_conv_layer is None:

    raise RuntimeError(
        "Could not find suitable Grad-CAM layer."
    )


print(
    "Grad-CAM layer :",
    last_conv_layer.name
)


# ============================================================
# FEATURE MODEL
# ============================================================

feature_model = Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# ============================================================
# LOAD IMAGE
# ============================================================

print(
    "\nProcessing:",
    os.path.basename(IMAGE_PATH)
)

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:

    raise FileNotFoundError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# SAME PREPROCESSING AS TRAINING/EVALUATION
# ============================================================

resized = cv2.resize(
    image_rgb,
    (224, 224)
)

image_array = (
    resized.astype(np.float32) / 255.0
)

input_tensor = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# GRAD-CAM FOR PREDICTED CLASS
# ============================================================

with tf.GradientTape() as tape:

    conv_output, base_output = feature_model(
        input_tensor
    )

    x = base_output

    for layer in model.layers:

        if layer.name == base_model.name:

            continue

        x = layer(x)

    predictions = x

    predicted_index = tf.argmax(
        predictions[0]
    )

    predicted_score = predictions[
        0,
        predicted_index
    ]


grads = tape.gradient(
    predicted_score,
    conv_output
)


# ============================================================
# CALCULATE GRAD-CAM
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(1, 2)
)

conv_output = conv_output[0]

pooled_grads = pooled_grads[0]

heatmap = tf.reduce_sum(
    conv_output * pooled_grads,
    axis=-1
)

heatmap = tf.maximum(
    heatmap,
    0
)

max_value = tf.reduce_max(
    heatmap
)

if float(max_value) > 0:

    heatmap /= max_value


heatmap = heatmap.numpy()


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

predictions_np = predictions[0].numpy()

predicted_index = int(
    predicted_index.numpy()
)

predicted_name = CLASS_NAMES[
    predicted_index
]

predicted_confidence = float(
    predictions_np[predicted_index]
)


print("\nClassification result:")
print("-" * 40)

for name, probability in zip(
    CLASS_NAMES,
    predictions_np
):

    print(
        f"{name:18s}: "
        f"{probability:.4f} "
        f"({probability * 100:.2f}%)"
    )

print("-" * 40)

print(
    f"Predicted class : {predicted_name}"
)

print(
    f"Confidence      : "
    f"{predicted_confidence * 100:.2f}%"
)


# ============================================================
# CREATE HEATMAP
# ============================================================

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_resized = cv2.resize(
    heatmap_uint8,
    (
        image_rgb.shape[1],
        image_rgb.shape[0]
    )
)

heatmap_color = cv2.applyColorMap(
    heatmap_resized,
    cv2.COLORMAP_JET
)


image_bgr_original = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR
)


overlay = cv2.addWeighted(
    image_bgr_original,
    0.55,
    heatmap_color,
    0.45,
    0
)


# ============================================================
# LABEL
# ============================================================

label = (
    f"Predicted: {predicted_name} "
    f"({predicted_confidence * 100:.2f}%)"
)


cv2.putText(
    overlay,
    label,
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    overlay,
    label,
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 0, 0),
    1,
    cv2.LINE_AA
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    overlay
)


print("\nGrad-CAM saved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)