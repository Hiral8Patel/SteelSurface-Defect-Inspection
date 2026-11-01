import os
import json
import cv2
import numpy as np
import tensorflow as tf
load_model = tf.keras.models.load_model
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\Admin\Downloads\NEU_FYP"

CLASSIFICATION_MODEL = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results",
    "mobilenetv2_baseline.keras"
)

DETECTION_MODEL = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_results",
    "yolo11n_baseline",
    "weights",
    "best.pt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures",
    "final_inspection"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("AI-ASSISTED STEEL SURFACE INSPECTION PIPELINE")
print("=" * 70)

print("\nLoading MobileNetV2 classification model...")
classification_model = load_model(CLASSIFICATION_MODEL)

print("Loading YOLO11n detection model...")
detection_model = YOLO(DETECTION_MODEL)

print("Models loaded successfully.")


# ============================================================
# GRAD-CAM SETUP
# ============================================================

print("\nSetting up Grad-CAM...")

BASE_MODEL = classification_model.get_layer("mobilenetv2_1.00_224")
GRAD_CAM_LAYER = BASE_MODEL.get_layer("Conv_1")

feature_model = tf.keras.Model(
    inputs=BASE_MODEL.input,
    outputs=[
        GRAD_CAM_LAYER.output,
        BASE_MODEL.output
    ]
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, (224, 224))

    # Match the preprocessing used during MobileNetV2 training
    image_array = resized.astype(np.float32) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image, image_rgb, image_array


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_image(image_array):

    predictions = classification_model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index])

    all_predictions = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return (
        predicted_class,
        confidence,
        all_predictions
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_defects(image_path):

    results = detection_model.predict(
        source=image_path,
        imgsz=224,
        conf=0.10,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy()

            confidence = float(
                boxes.conf[i].cpu().numpy()
            )

            class_id = int(
                boxes.cls[i].cpu().numpy()
            )

            detections.append({
                "class": CLASS_NAMES[class_id],
                "class_id": class_id,
                "confidence": confidence,
                "box": [
                    int(xyxy[0]),
                    int(xyxy[1]),
                    int(xyxy[2]),
                    int(xyxy[3])
                ]
            })

    return result, detections


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_array, predicted_index):

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_output = feature_model(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        gap_output = classification_model.get_layer(
            "global_average_pooling2d"
        )(base_output)

        dropout_output = classification_model.get_layer(
            "dropout"
        )(gap_output, training=False)

        predictions = classification_model.get_layer(
            "dense"
        )(dropout_output)

        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap /= max_value

    heatmap = heatmap.numpy()

    return heatmap


# ============================================================
# CREATE GRAD-CAM VISUALIZATION
# ============================================================

def create_gradcam_visualization(
    original_rgb,
    heatmap,
    output_path
):

    original = cv2.resize(
        original_rgb,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (224, 224)
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_resized,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        output_path,
        overlay_bgr
    )


# ============================================================
# CREATE DETECTION VISUALIZATION
# ============================================================

def create_detection_visualization(
    image,
    detections,
    output_path
):

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        label = (
            f'{detection["class"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(
        output_path,
        output
    )


# ============================================================
# FINAL PIPELINE
# ============================================================

def run_inspection(image_path):

    print("\n" + "=" * 70)
    print("RUNNING FINAL INSPECTION")
    print("=" * 70)

    print(f"\nInput image:")
    print(image_path)

    # --------------------------------------------------------
    # 1. PREPROCESS
    # --------------------------------------------------------

    image, image_rgb, image_array = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # 2. CLASSIFICATION
    # --------------------------------------------------------

    print("\n[1/3] Running classification...")

    predicted_class, confidence, all_predictions = classify_image(
        image_array
    )

    predicted_index = CLASS_NAMES.index(
        predicted_class
    )

    print(
        f"Predicted defect : {predicted_class}"
    )

    print(
        f"Confidence       : {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 3. YOLO LOCALIZATION
    # --------------------------------------------------------

    print("\n[2/3] Running defect localization...")

    _, detections = detect_defects(
        image_path
    )

    print(
        f"Detections       : {len(detections)}"
    )

    for i, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"  {i}. "
            f"{detection['class']} "
            f"({detection['confidence']:.2f}) "
            f"box={detection['box']}"
        )

    # --------------------------------------------------------
    # 4. GRAD-CAM
    # --------------------------------------------------------

    print("\n[3/3] Generating Grad-CAM...")

    heatmap = generate_gradcam(
        image_array,
        predicted_index
    )

    # --------------------------------------------------------
    # 5. SAVE OUTPUTS
    # --------------------------------------------------------

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    detection_output = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_detection.jpg"
    )

    gradcam_output = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_gradcam.jpg"
    )

    result_json = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_result.json"
    )

    create_detection_visualization(
        image,
        detections,
        detection_output
    )

    create_gradcam_visualization(
        image_rgb,
        heatmap,
        gradcam_output
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    final_result = {

        "input_image": image_path,

        "classification": {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_class_probabilities": all_predictions
        },

        "localization": {
            "number_of_detections": len(detections),
            "detections": detections
        },

        "explainability": {
            "method": "Grad-CAM",
            "target_class": predicted_class,
            "description":
                "Visualization of image regions contributing "
                "to the classification prediction."
        },

        "scope": {
            "system_type":
                "AI-assisted visual inspection",
            "limitations":
                "The system performs visible defect "
                "classification and localization. "
                "It does not determine structural strength, "
                "material safety, or suitability for reuse."
        },

        "outputs": {
            "detection_visualization":
                detection_output,
            "gradcam_visualization":
                gradcam_output
        }
    }

    with open(
        result_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_result,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL INSPECTION RESULT")
    print("=" * 70)

    print(
        f"Classification : {predicted_class}"
    )

    print(
        f"Confidence     : {confidence * 100:.2f}%"
    )

    print(
        f"YOLO detections: {len(detections)}"
    )

    print(
        f"\nDetection image:"
    )

    print(
        detection_output
    )

    print(
        f"\nGrad-CAM image:"
    )

    print(
        gradcam_output
    )

    print(
        f"\nJSON report:"
    )

    print(
        result_json
    )

    print("\nPipeline completed successfully.")

    return final_result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nEnter the path of a steel surface image.")

    image_path = input(
        "\nImage path: "
    ).strip().strip('"')

    if not os.path.exists(image_path):

        print(
            "\nERROR: Image file does not exist."
        )

        raise SystemExit

    run_inspection(
        image_path
    )