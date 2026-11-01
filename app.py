import os
import sys
import json
import tempfile

import streamlit as st
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = r"C:\Users\Admin\Downloads\NEU_FYP"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# Import the integrated pipeline
from final_inspection_pipeline import (
    run_inspection,
    CLASS_NAMES
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Steel Surface Inspection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
        color: #666666;
    }

    /* Plain white text for final inspection summary */
    .result-text {
        color: white;
        font-size: 16px;
        line-height: 1.8;
    }

    /* Plain white text for scope and limitation */
    .scope-text {
        color: white;
        font-size: 14px;
        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'AI-Assisted Steel Surface Inspection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Visual Defect Classification, '
    'Localization and Explainability'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Inspection System")

    st.write(
        "This system combines three AI components:"
    )

    st.markdown(
        """
        **1. MobileNetV2**  
        Defect classification

        **2. YOLO11n**  
        Visible defect localization

        **3. Grad-CAM**  
        Classification explainability
        """
    )

    st.divider()

    st.subheader("Supported Defects")

    for class_name in CLASS_NAMES:
        st.write(
            f"• {class_name.replace('_', ' ').title()}"
        )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Upload Steel Surface Image'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a steel surface image for AI-assisted visual inspection."
)


# ============================================================
# RUN INSPECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    st.write("")

    analyze_button = st.button(
        "🔍 Analyze Image",
        type="primary",
        width="stretch"
    )

    if analyze_button:

        # Save uploaded image temporarily
        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_image_path = temp_file.name

        try:

            with st.spinner(
                "Running AI-assisted inspection..."
            ):

                result = run_inspection(
                    temp_image_path
                )

            st.success(
                "Inspection completed successfully."
            )


            # =================================================
            # CLASSIFICATION RESULT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '1. Defect Classification'
                '</div>',
                unsafe_allow_html=True
            )

            classification = result[
                "classification"
            ]

            predicted_class = classification[
                "predicted_class"
            ]

            confidence = classification[
                "confidence"
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Defect",
                    predicted_class.replace(
                        "_",
                        " "
                    ).title()
                )

            with col2:

                st.metric(
                    "Classification Confidence",
                    f"{confidence * 100:.2f}%"
                )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.subheader(
                "Class Probability Distribution"
            )

            probabilities = classification[
                "all_class_probabilities"
            ]

            probability_data = {
                key.replace("_", " ").title():
                value
                for key, value in probabilities.items()
            }

            st.bar_chart(
                probability_data
            )


            # =================================================
            # LOCALIZATION
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '2. Defect Localization'
                '</div>',
                unsafe_allow_html=True
            )

            localization = result[
                "localization"
            ]

            detections = localization[
                "detections"
            ]

            st.metric(
                "Visible Defect Regions Detected",
                localization[
                    "number_of_detections"
                ]
            )

            if detections:

                st.subheader(
                    "Detected Defect Regions"
                )

                for index, detection in enumerate(
                    detections,
                    start=1
                ):

                    defect_name = detection[
                        "class"
                    ].replace(
                        "_",
                        " "
                    ).title()

                    detection_confidence = (
                        detection[
                            "confidence"
                        ] * 100
                    )

                    box = detection[
                        "box"
                    ]

                    st.write(
                        f"**{index}. "
                        f"{defect_name}** — "
                        f"{detection_confidence:.2f}%"
                    )

                    st.caption(
                        f"Bounding box: {box}"
                    )


            # =================================================
            # DETECTION IMAGE
            # =================================================

            detection_path = result[
                "outputs"
            ][
                "detection_visualization"
            ]

            if os.path.exists(
                detection_path
            ):

                st.image(
                    detection_path,
                    caption=(
                        "YOLO11n Defect Localization"
                    ),
                    width="stretch"
                )


            # =================================================
            # GRAD-CAM
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '3. Explainable AI — Grad-CAM'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Grad-CAM highlights image regions "
                "that contributed to the classification "
                "prediction."
            )

            gradcam_path = result[
                "outputs"
            ][
                "gradcam_visualization"
            ]

            if os.path.exists(
                gradcam_path
            ):

                st.image(
                    gradcam_path,
                    caption=(
                        "Grad-CAM Classification Explanation"
                    ),
                    width="stretch"
                )


            # =================================================
            # INSPECTION SUMMARY
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '4. Inspection Summary'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-text">

                <b>Classification:</b>
                {predicted_class.replace("_", " ").title()}

                <br><br>

                <b>Classification confidence:</b>
                {confidence * 100:.2f}%

                <br><br>

                <b>Localized visible defect regions:</b>
                {len(detections)}

                <br><br>

                <b>Explainability method:</b>
                Grad-CAM

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.markdown(
                """
                <div class="scope-text">

                <b>Scope and limitation</b>

                <br><br>

                This system performs AI-assisted visual
                classification and localization of visible
                steel-surface defects.

                It does <b>not</b> determine structural
                strength, material safety, or suitability
                for reuse.

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # JSON REPORT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '5. Inspection Report'
                '</div>',
                unsafe_allow_html=True
            )

            json_data = json.dumps(
                result,
                indent=4
            )

            st.download_button(
                label="📄 Download Inspection Report (JSON)",
                data=json_data,
                file_name=(
                    "steel_surface_inspection_report.json"
                ),
                mime="application/json",
                width="stretch"
            )


        except Exception as error:

            st.error(
                "An error occurred during inspection."
            )

            st.exception(
                error
            )


        finally:

            # Remove temporary uploaded image
            if os.path.exists(
                temp_image_path
            ):

                os.remove(
                    temp_image_path
                )


else:

    st.info(
        "Upload a steel surface image above "
        "to begin the inspection."
    )