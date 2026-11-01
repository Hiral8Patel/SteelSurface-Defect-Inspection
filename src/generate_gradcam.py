import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Admin\Downloads\NEU_FYP")

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_results"
    / "mobilenetv2_baseline.keras"
)

IMAGE_DIR = (
    Path(r"C:\Users\Admin\Downloads\archive\NEU-DET")
    / "validation"
    / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "gradcam"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MOBILENETV2 GRAD-CAM")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MobileNetV2...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# FIND THE ACTUAL MOBILENETV2 MODEL
# ============================================================

print("\nOuter model layers:")

for i, layer in enumerate(model.layers):
    print(f"  [{i}] {layer.name} ({layer.__class__.__name__})")


# The trained model contains:
#
# Input
#   ↓
# Sequential augmentation
#   ↓
# MobileNetV2
#   ↓
# GlobalAveragePooling
#   ↓
# Dropout
#   ↓
# Dense
#
# We need to find the actual Functional MobileNetV2 model,
# not the outer Sequential augmentation model.

mobile_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        print(
            f"\nFound nested model: "
            f"{layer.name} ({layer.__class__.__name__})"
        )

        # Search inside it for the actual MobileNetV2 Functional model
        if layer.name == "mobilenetv2_1.00_224":
            mobile_model = layer
            break

        for nested_layer in layer.layers:
            if (
                isinstance(nested_layer, tf.keras.Model)
                and "mobilenetv2" in nested_layer.name.lower()
            ):
                mobile_model = nested_layer
                break

        if mobile_model is not None:
            break


if mobile_model is None:

    # Fallback: search recursively through all nested models
    def find_mobilenet_recursive(current_model):

        for layer in current_model.layers:

            if isinstance(layer, tf.keras.Model):

                if "mobilenetv2" in layer.name.lower():
                    return layer

                result = find_mobilenet_recursive(layer)

                if result is not None:
                    return result

        return None


    mobile_model = find_mobilenet_recursive(model)


if mobile_model is None:
    raise RuntimeError(
        "Could not locate the nested MobileNetV2 model."
    )


print("\nActual MobileNetV2 model:")
print("Name :", mobile_model.name)
print("Type :", mobile_model.__class__.__name__)


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

print("\nSearching MobileNetV2 convolutional layers...")

conv_layers = []

for layer in mobile_model.layers:

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.SeparableConv2D,
        ),
    ):
        conv_layers.append(layer)


if not conv_layers:
    raise RuntimeError(
        "No convolutional layers found inside MobileNetV2."
    )


print(f"Convolutional layers found: {len(conv_layers)}")

for layer in conv_layers[-5:]:
    print(
        f"  {layer.name} "
        f"({layer.__class__.__name__})"
    )


last_conv_layer = conv_layers[-1]

print("\nGrad-CAM layer:", last_conv_layer.name)


# ============================================================
# IDENTIFY AUGMENTATION SEQUENTIAL
# ============================================================

augmentation_model = None

for layer in model.layers:

    if (
        isinstance(layer, tf.keras.Sequential)
        and layer is not mobile_model
    ):
        augmentation_model = layer
        break


if augmentation_model is not None:

    print("\nAugmentation model:", augmentation_model.name)

    print("Augmentation layers:")

    for layer in augmentation_model.layers:
        print(
            f"  - {layer.name} "
            f"({layer.__class__.__name__})"
        )

else:

    print("\nNo separate augmentation Sequential found.")


# ============================================================
# FIND CLASSIFIER HEAD
# ============================================================

gap_layer = None
dropout_layer = None
dense_layer = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.GlobalAveragePooling2D
    ):
        gap_layer = layer

    elif isinstance(
        layer,
        tf.keras.layers.Dropout
    ):
        dropout_layer = layer

    elif isinstance(
        layer,
        tf.keras.layers.Dense
    ):
        dense_layer = layer


if gap_layer is None:
    raise RuntimeError(
        "GlobalAveragePooling2D layer not found."
    )

if dense_layer is None:
    raise RuntimeError(
        "Final Dense classifier layer not found."
    )


print("\nClassifier head:")
print("  -", gap_layer.name)

if dropout_layer is not None:
    print("  -", dropout_layer.name)

print("  -", dense_layer.name)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam_heatmap(
    image_array,
    predicted_class_index
):
    """
    Generate Grad-CAM from the actual nested MobileNetV2
    Functional model.

    The classifier uses:
        MobileNetV2 -> GAP -> Dropout -> Dense

    Grad-CAM uses:
        MobileNetV2 -> Conv_1
    """

    # --------------------------------------------------------
    # Build a model that returns:
    # 1. Conv_1 feature maps
    # 2. Final MobileNetV2 feature vector
    # --------------------------------------------------------

    feature_model = tf.keras.models.Model(
        inputs=mobile_model.input,
        outputs=[
            last_conv_layer.output,
            mobile_model.output,
        ],
        name="gradcam_feature_model",
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # Run the actual MobileNetV2 Functional graph
        # ----------------------------------------------------

        conv_output, mobile_output = feature_model(
            image_tensor,
            training=False
        )

        # ----------------------------------------------------
        # Apply the same classifier head
        # ----------------------------------------------------

        x = gap_layer(mobile_output)

        if dropout_layer is not None:
            x = dropout_layer(
                x,
                training=False
            )

        predictions = dense_layer(x)

        # ----------------------------------------------------
        # Score of predicted class
        # ----------------------------------------------------

        class_score = predictions[
            :, predicted_class_index
        ]

    # --------------------------------------------------------
    # Gradients of predicted class with respect to Conv_1
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. "
            "Grad-CAM graph is not connected."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    conv_output = conv_output[0]

    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------------
    # Weight each feature map
    # --------------------------------------------------------

    weighted_features = (
        conv_output
        * pooled_gradients
    )

    # --------------------------------------------------------
    # Sum feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize to 0-1
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()
    """
    Generate Grad-CAM using the actual nested MobileNetV2
    convolutional feature map.

    The forward path is:

        input
          ↓
        augmentation (inference mode)
          ↓
        MobileNetV2
          ↓
        last convolution
          ↓
        GAP
          ↓
        Dropout (inference mode)
          ↓
        Dense
    """

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # Apply augmentation model if present
        # ----------------------------------------------------

        if augmentation_model is not None:

            augmented = augmentation_model(
                image_tensor,
                training=False
            )

        else:

            augmented = image_tensor


        # ----------------------------------------------------
        # Run MobileNetV2 manually
        # ----------------------------------------------------

        # We need both:
        # 1. final MobileNetV2 output
        # 2. activation of last convolutional layer
        #
        # Instead of constructing a disconnected Keras graph,
        # run the MobileNetV2 layers directly.

        x = augmented

        conv_output = None

        for layer in mobile_model.layers:

            x = layer(
                x,
                training=False
            )

            if layer is last_conv_layer:
                conv_output = x


        mobile_output = x


        if conv_output is None:
            raise RuntimeError(
                "Grad-CAM convolution output was not captured."
            )


        # ----------------------------------------------------
        # Classifier head
        # ----------------------------------------------------

        x = gap_layer(mobile_output)

        if dropout_layer is not None:
            x = dropout_layer(
                x,
                training=False
            )

        predictions = dense_layer(x)


        # ----------------------------------------------------
        # Target class score
        # ----------------------------------------------------

        class_score = predictions[
            :, predicted_class_index
        ]


    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_output
    )


    if gradients is None:
        raise RuntimeError(
            "Gradients are None. "
            "Grad-CAM graph was not connected."
        )


    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


    # --------------------------------------------------------
    # Weight feature maps
    # --------------------------------------------------------

    conv_output = conv_output[0]

    pooled_gradients = pooled_gradients[0]

    weighted_features = (
        conv_output
        * pooled_gradients
    )


    # --------------------------------------------------------
    # Combine feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )


    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )


    return heatmap.numpy()


# ============================================================
# CREATE GRAD-CAM VISUALIZATION
# ============================================================

def create_gradcam_visualization(
    original_image,
    heatmap,
    output_path
):

    height, width = original_image.shape[:2]

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # OpenCV heatmap
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Convert original RGB → BGR
    original_bgr = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )

    # Overlay
    overlay = cv2.addWeighted(
        original_bgr,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image_path):

    print("\n" + "-" * 70)
    print(f"Processing: {image_path.name}")
    print("-" * 70)


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )


    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        image_rgb,
        (224, 224)
    )


    # IMPORTANT:
    # The classifier was trained using 0-1 scaling.
    #
    # Do NOT use MobileNetV2 preprocess_input here.
    #
    image_array = (
        resized.astype(np.float32) / 255.0
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Normal classification
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    predicted_class_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_class_index
    ]

    confidence = float(
        predictions[predicted_class_index]
    )


    # --------------------------------------------------------
    # Print probabilities
    # --------------------------------------------------------

    print("\nClassification probabilities:")
    print("-" * 45)

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions
    ):

        print(
            f"{class_name:<18}: "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%)"
        )

    print("-" * 45)

    print(
        f"Prediction : {predicted_class}"
    )

    print(
        f"Confidence : {confidence:.4f}"
    )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print("\nGenerating Grad-CAM...")

    heatmap = make_gradcam_heatmap(
        image_array,
        predicted_class_index
    )


    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_filename = (
        f"gradcam_{image_path.stem}.jpg"
    )

    output_path = (
        OUTPUT_DIR
        / output_filename
    )


    create_gradcam_visualization(
        image_rgb,
        heatmap,
        output_path
    )


    print(
        f"Grad-CAM saved to:"
        f"\n{output_path}"
    )


    return {
        "image": image_path.name,
        "prediction": predicted_class,
        "confidence": confidence,
        "output": str(output_path),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("GENERATING FINAL SIX-CLASS GRAD-CAM EXAMPLES")
    print("=" * 70)

    # One representative image for each defect class.
    #
    # These images are from the NEU-DET validation dataset.
    representative_images = {
        "crazing": "crazing_241.jpg",
        "inclusion": "inclusion_241.jpg",
        "patches": "patches_241.jpg",
        "pitted_surface": "pitted_surface_241.jpg",
        "rolled-in_scale": "rolled-in_scale_241.jpg",
        "scratches": "scratches_241.jpg",
    }

    results = []

    for class_name, filename in representative_images.items():

        image_path = (
            IMAGE_DIR
            / class_name
            / filename
        )

        print("\n")
        print("=" * 70)
        print(f"CLASS: {class_name}")
        print(f"IMAGE: {filename}")
        print("=" * 70)

        if not image_path.exists():

            print(
                f"WARNING: Image not found:\n"
                f"{image_path}"
            )

            continue

        try:

            result = process_image(
                image_path
            )

            results.append(result)

        except Exception as e:

            print(
                f"\nERROR processing {filename}:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL GRAD-CAM SUMMARY")
    print("=" * 70)

    print(
        f"\nSuccessfully processed: "
        f"{len(results)}/{len(representative_images)}"
    )

    print("\nResults:")

    for result in results:

        print(
            f"  {result['image']:<28} "
            f"-> {result['prediction']:<18} "
            f"({result['confidence'] * 100:.2f}%)"
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for result in results:

        print(
            f"  - "
            f"{Path(result['output']).name}"
        )

    print("\n")
    print("=" * 70)
    print("SIX-CLASS GRAD-CAM GENERATION COMPLETED")
    print("=" * 70)