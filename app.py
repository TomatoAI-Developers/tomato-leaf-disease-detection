import io
import os
import time
import threading
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import joblib
import tensorflow as tf

from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model, Model

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERIFIER_DIR = os.path.join(
    BASE_DIR,
    "artifacts", "phase3", "verifier_v2",
    "run_20261107_190658_190660"
)

CLASSICAL_DIR = os.path.join(
    BASE_DIR,
    "artifacts", "phase3", "classical_ml",
    "run_20261107_211856"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Not_Tomato",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Mosaic_Virus",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_CurlVirus"
]

MODEL_FILES = {
    "SVM": "svm.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "KNN": "knn.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Extra Trees": "extra_trees.joblib",
    "Gradient Boosting": "gradient_boosting.joblib"
}

MODEL_LOCK = threading.Lock()


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

print("\nLoading TomatoAI models...")

mobilenet = load_model(
    os.path.join(VERIFIER_DIR, "best_verifier_v2.keras"),
    compile=False
)

feature_extractor = Model(
    inputs=mobilenet.input,
    outputs=mobilenet.get_layer("global_average_pooling2d").output
)

pca = joblib.load(os.path.join(CLASSICAL_DIR, "pca.joblib"))

classical_models = {}

for model_name, filename in MODEL_FILES.items():
    model_path = os.path.join(CLASSICAL_DIR, filename)
    classical_models[model_name] = joblib.load(model_path)
    print("Loaded:", model_name)

print("Loaded: MobileNetV2")
print("All eight models are ready.\n")


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_uploaded_image(file_stream):
    image = Image.open(file_stream)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return array


# --------------------------------------------------
# PREDICTION HELPERS
# --------------------------------------------------

def class_display_name(class_name):
    if class_name == "Not_Tomato":
        return "Not Tomato Leaf"

    if class_name.startswith("Tomato_"):
        class_name = class_name[len("Tomato_"):]

    return class_name.replace("_", " ")


def probability_for_prediction(model, probabilities, prediction):
    probabilities = np.asarray(probabilities)[0]
    classes = np.asarray(model.classes_)

    positions = np.where(classes == prediction)[0]

    if len(positions) == 0:
        return float(np.max(probabilities) * 100)

    return float(probabilities[positions[0]] * 100)


def get_recommendations(class_name):
    if class_name == "Not_Tomato":
        return {
            "home": "Upload a clear image of a tomato leaf.",
            "doctor": "This image was not classified as a tomato leaf.",
            "expert": "Try again with a well-lit, focused tomato leaf image."
        }

    if class_name == "Tomato_Healthy":
        return {
            "home": "Continue regular watering and balanced plant care.",
            "doctor": "No disease was identified by the primary model.",
            "expert": "Monitor the plant regularly for any new symptoms."
        }

    recommendations = {
        "home": "Remove visibly affected leaves when appropriate and avoid wetting foliage.",
        "doctor": "Confirm the diagnosis before applying any pesticide or fungicide.",
        "expert": "Consult a local agricultural officer for treatment suited to the crop and region."
    }

    return recommendations


def get_severity(class_name, confidence):
    if class_name in ("Not_Tomato", "Tomato_Healthy"):
        return "Not Applicable"

    if confidence >= 85:
        return "High"
    elif confidence >= 65:
        return "Moderate"
    return "Low"


def get_progression(class_name):
    if class_name == "Not_Tomato":
        return "Not Applicable"
    if class_name == "Tomato_Healthy":
        return "No disease detected"
    return "Requires monitoring"


# --------------------------------------------------
# MAIN PREDICTION FUNCTION
# --------------------------------------------------

def run_all_models(image_array):

    # Keep TensorFlow inference serialized.
    with MODEL_LOCK:

        # MobileNetV2 prediction
        start = time.perf_counter()

        mobilenet_output = mobilenet(
            tf.convert_to_tensor(image_array),
            training=False
        ).numpy()[0]

        mobilenet_ms = (time.perf_counter() - start) * 1000

        mobilenet_probabilities = np.asarray(mobilenet_output, dtype=np.float64)

        # Normalize defensively if model outputs are not normalized.
        total = mobilenet_probabilities.sum()
        if total > 0:
            mobilenet_probabilities /= total

        mobilenet_index = int(np.argmax(mobilenet_probabilities))
        mobilenet_class = CLASS_NAMES[mobilenet_index]
        mobilenet_confidence = float(
            mobilenet_probabilities[mobilenet_index] * 100
        )

        # Shared feature extraction for classical models.
        feature_start = time.perf_counter()

        features = feature_extractor(
            tf.convert_to_tensor(image_array),
            training=False
        ).numpy()

        feature_extraction_ms = (
            time.perf_counter() - feature_start
        ) * 1000

        reduced_features = pca.transform(features)

        # Build comparison results.
        model_results = [
            {
                "model": "MobileNetV2",
                "prediction": class_display_name(mobilenet_class),
                "confidence": round(mobilenet_confidence, 2),
                "inference_ms": round(mobilenet_ms, 3)
            }
        ]

        for model_name, classifier in classical_models.items():

            classifier_start = time.perf_counter()

            prediction = classifier.predict(reduced_features)[0]

            classifier_ms = (
                time.perf_counter() - classifier_start
            ) * 1000

            if hasattr(classifier, "predict_proba"):
                probabilities = classifier.predict_proba(reduced_features)
                confidence = probability_for_prediction(
                    classifier, probabilities, prediction
                )
            else:
                confidence = None

            prediction_index = int(prediction)
            prediction_class = CLASS_NAMES[prediction_index]

            # Total includes shared feature extraction, PCA and classifier.
            total_inference_ms = (
                feature_extraction_ms + classifier_ms
            )

            model_results.append({
                "model": model_name,
                "prediction": class_display_name(prediction_class),
                "confidence": (
                    round(confidence, 2)
                    if confidence is not None else None
                ),
                "inference_ms": round(total_inference_ms, 3),
                "classifier_ms": round(classifier_ms, 3),
                "feature_extraction_ms": round(feature_extraction_ms, 3)
            })

    # --------------------------------------------------
    # TOMATO IMAGE VERIFICATION: 8-MODEL VOTING GATE
    # --------------------------------------------------

    non_tomato_votes = sum(
        1 for item in model_results
        if item.get("prediction") == "Not Tomato Leaf"
    )

    # If at least 5 of the 8 models reject the image,
    # do not display a tomato disease diagnosis.
    if non_tomato_votes >= 5:
        mobilenet_class = "Not_Tomato"
        mobilenet_confidence = float(
            mobilenet_probabilities[0] * 100
        )

    return (
        mobilenet_class,
        mobilenet_probabilities,
        mobilenet_confidence,
        model_results
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Please select an image."}), 400

    try:
        image_array = preprocess_uploaded_image(uploaded_file.stream)

        (
            predicted_class,
            probabilities,
            confidence,
            model_results
        ) = run_all_models(image_array)

        top_indices = np.argsort(probabilities)[::-1][:3]

        top3 = [
            {
                "name": class_display_name(CLASS_NAMES[int(index)]),
                "confidence": round(float(probabilities[index] * 100), 2)
            }
            for index in top_indices
        ]

        tomato_confidence = (
            1.0 - float(probabilities[0])
        ) * 100

        result = {
            "disease": class_display_name(predicted_class),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "severity": get_severity(predicted_class, confidence),
            "progression": get_progression(predicted_class),
            "tomato_conf": round(tomato_confidence, 2),
            "top3": top3,
            "recommendations": get_recommendations(predicted_class),
            "model_results": model_results,
            "non_tomato_votes": sum(
                1 for item in model_results
                if item.get("prediction") == "Not Tomato Leaf"
            ),
            "verification_rule": "At least 5 of 8 models must identify a non-tomato image.",
            "primary_model": "Eight-model verification gate with MobileNetV2 diagnosis",
            "feature_extraction_ms": round(
                model_results[1]["feature_extraction_ms"], 3
            ),
            "duplicate_warning": (
                "The test dataset contains cross-split duplicate images. "
                "Reported test metrics should not be interpreted as "
                "independent, leakage-free generalization."
            ),
            "pdf_enabled": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(result)

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Unable to read this image. Please upload a valid image."
        }), 400

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed: " + str(error)
        }), 500


# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------

import io
from datetime import datetime
from xml.sax.saxutils import escape

from flask import request, jsonify, send_file

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)


@app.route("/report", methods=["POST"])
def generate_report():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid report data"}), 400

    buffer = io.BytesIO()

    NAVY = HexColor("#14243A")
    GREEN = HexColor("#19B968")
    DARK_GREEN = HexColor("#205C43")
    BLUE = HexColor("#397BFF")
    ORANGE = HexColor("#F59E0B")
    RED = HexColor("#F04444")
    GREY = HexColor("#65748B")
    LIGHT = HexColor("#F4F8F6")
    BORDER = HexColor("#DCE7E1")

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TomatoTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=23,
        leading=28, textColor=NAVY,
        alignment=TA_CENTER, spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        "TomatoSubtitle", fontName="Helvetica",
        fontSize=8, leading=12, textColor=GREY,
        alignment=TA_CENTER
    )

    section_style = ParagraphStyle(
        "TomatoSection", fontName="Helvetica-Bold",
        fontSize=14, leading=19, textColor=NAVY,
        spaceBefore=13, spaceAfter=8
    )

    small_style = ParagraphStyle(
        "TomatoSmall", fontName="Helvetica",
        fontSize=8, leading=11, textColor=GREY
    )

    body_style = ParagraphStyle(
        "TomatoBody", fontName="Helvetica",
        fontSize=8.5, leading=13, textColor=NAVY
    )

    table_style = ParagraphStyle(
        "TomatoTable", fontName="Helvetica",
        fontSize=7.5, leading=10, textColor=NAVY
    )

    def safe(value):
        return escape(str(value if value is not None else "N/A"))

    story = []

    # HEADER
    story.append(Paragraph("TomatoAI Diagnostic", title_style))
    story.append(Paragraph(
        "PRECISION AGRICULTURE  •  DEEP LEAF INTELLIGENCE PLATFORM",
        subtitle_style
    ))
    story.append(Spacer(1, 9))
    story.append(HRFlowable(
        width="100%", thickness=1, color=BORDER, spaceAfter=10
    ))

    story.append(Paragraph(
        "Generated: " + safe(data.get(
            "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )),
        small_style
    ))
    story.append(Spacer(1, 10))

    # METRICS
    confidence = float(data.get("confidence", 0) or 0)
    severity = str(data.get("severity", "N/A"))
    progression = str(data.get("progression", "N/A"))
    verified = float(data.get("tomato_conf", 0) or 0)

    severity_color = (
        RED if severity.lower() == "high"
        else ORANGE if severity.lower() == "moderate"
        else GREEN
    )

    metrics = [
        ("CONFIDENCE", f"{confidence:.2f}%", GREEN),
        ("SEVERITY", severity, severity_color),
        ("PROGRESSION", progression, ORANGE),
        ("TOMATO VERIFIED", f"{verified:.2f}%", BLUE)
    ]

    metric_cells = []

    for heading, value, accent in metrics:
        metric_cells.append([
            Paragraph(
                safe(heading),
                ParagraphStyle(
                    "heading_" + heading.replace(" ", ""),
                    parent=small_style, alignment=TA_CENTER
                )
            ),
            Spacer(1, 6),
            Paragraph(
                safe(value),
                ParagraphStyle(
                    "value_" + heading.replace(" ", ""),
                    fontName="Helvetica-Bold",
                    fontSize=11 if len(value) > 13 else 16,
                    leading=19, alignment=TA_CENTER,
                    textColor=accent
                )
            )
        ])

    metrics_table = Table(
        [metric_cells],
        colWidths=[43 * mm] * 4
    )

    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4)
    ]))

    story.append(metrics_table)

    # PRIMARY DIAGNOSIS
    story.append(Paragraph("Primary Diagnosis", section_style))

    diagnosis = safe(data.get("disease", "Not available"))
    primary_model = safe(data.get("primary_model", "MobileNetV2"))

    diagnosis_table = Table([[
        Paragraph(
            "<b>" + diagnosis + "</b>",
            ParagraphStyle(
                "DiagnosisName", fontName="Helvetica-Bold",
                fontSize=15, leading=20, textColor=DARK_GREEN
            )
        ),
        Paragraph(
            "<b>PRIMARY MODEL</b><br/>" + primary_model,
            body_style
        )
    ]], colWidths=[105 * mm, 67 * mm])

    diagnosis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#ECF7F0")),
        ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#C8E6D1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12)
    ]))

    story.append(diagnosis_table)

    # PROBABILITY BARS
    story.append(Paragraph(
        "Disease Probability Distribution", section_style
    ))

    top3 = data.get("top3", [])
    bar_rows = []

    for index, item in enumerate(top3[:3]):
        percentage = max(
            0, min(100, float(item.get("confidence", 0) or 0))
        )

        bar_color = [GREEN, BLUE, ORANGE][index]

        filled_width = max(0.5, 90 * percentage / 100)
        remaining_width = max(0.5, 90 - filled_width)

        bar = Table(
            [["", ""]],
            colWidths=[filled_width * mm, remaining_width * mm],
            rowHeights=[5 * mm]
        )

        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), bar_color),
            ("BACKGROUND", (1, 0), (1, 0), HexColor("#E9EEF0")),
            ("BOX", (0, 0), (-1, -1), 0, colors.white)
        ]))

        bar_rows.append([
            Paragraph(safe(item.get("name", "Unknown")), table_style),
            bar,
            Paragraph(
                f"<b>{percentage:.2f}%</b>", table_style
            )
        ])

    if bar_rows:
        probability_table = Table(
            bar_rows,
            colWidths=[43 * mm, 95 * mm, 30 * mm]
        )

        probability_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER)
        ]))

        story.append(probability_table)
    else:
        story.append(Paragraph(
            "Probability distribution unavailable.", body_style
        ))

    # Include model comparison only when user selects YES.
    if data.get("include_comparison", True):
        # LIVE MODEL COMPARISON
        story.append(Paragraph(
            "Live Model Comparison", section_style
        ))
        story.append(Paragraph(
            "Actual predictions from the eight trained models.",
            small_style
        ))
        story.append(Spacer(1, 7))

        model_rows = [[
            "MODEL", "PREDICTION", "CONFIDENCE", "INFERENCE (ms)"
        ]]

        for item in data.get("model_results", []):
            model_rows.append([
                Paragraph(safe(item.get("model")), table_style),
                Paragraph(safe(item.get("prediction")), table_style),
                (
                    f"{float(item['confidence']):.2f}%"
                    if item.get("confidence") is not None else "N/A"
                ),
                str(item.get("inference_ms", "N/A"))
            ])

        comparison = Table(
            model_rows,
            repeatRows=1,
            colWidths=[43 * mm, 51 * mm, 35 * mm, 38 * mm]
        )

        comparison.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK_GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white, LIGHT
            ]),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(comparison)
        story.append(Spacer(1, 6))

        story.append(Paragraph(
            "Timing note: Classical model inference includes shared feature "
            "extraction and PCA. MobileNetV2 timing measures its prediction "
            "call. Confidence is not the same as model accuracy.",
            small_style
        ))


    # RECOMMENDATIONS
    story.append(Paragraph(
        "Actionable Insights & Recommendations", section_style
    ))

    recommendations = data.get("recommendations", {})

    recommendation_cards = [
        ("HOME PROTOCOL", recommendations.get("home", "Not available"), GREEN),
        ("MEDICAL GUIDANCE", recommendations.get("doctor", "Not available"), RED),
        ("EXPERT GUIDANCE", recommendations.get("expert", "Not available"), BLUE)
    ]

    for heading, content, accent in recommendation_cards:
        card = Table([[
            Paragraph(
                "<b>" + safe(heading) + "</b><br/><br/>" + safe(content),
                body_style
            )
        ]], colWidths=[172 * mm])

        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
            ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
        ]))

        story.append(KeepTogether([card, Spacer(1, 7)]))

    # FOOTER AND DISCLAIMER
    story.append(Spacer(1, 9))
    story.append(HRFlowable(
        width="100%", thickness=0.8, color=BORDER, spaceAfter=9
    ))

    story.append(Paragraph(
        "<i>Disclaimer: This AI-generated report is for informational "
        "purposes. Final verification by an agricultural expert is "
        "recommended before applying treatments.</i>",
        small_style
    ))

    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>© TomatoAI — Advanced Precision Agriculture Intelligence</b>",
        ParagraphStyle(
            "TomatoFooter", fontName="Helvetica-Bold",
            fontSize=8, alignment=TA_CENTER, textColor=NAVY
        )
    ))

    warning = data.get("duplicate_warning", "")

    if warning:
        story.append(Spacer(1, 7))
        story.append(Paragraph(safe(warning), small_style))

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="TomatoAI_Diagnostic_Report.pdf"
    )
# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":
    print("TomatoAI server starting...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )

