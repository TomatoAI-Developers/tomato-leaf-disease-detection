from pathlib import Path

path = Path("app.py")
code = path.read_text(encoding="utf-8")

old = '''    return (
        mobilenet_class,
        mobilenet_probabilities,
        mobilenet_confidence,
        model_results
    )
'''

new = '''    # --------------------------------------------------
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
'''

if old not in code:
    raise SystemExit(
        "ERROR: Expected prediction return block not found. "
        "No changes made."
    )

code = code.replace(old, new, 1)

# Add transparent voting information to the API response.
old_result = '''            "model_results": model_results,
            "primary_model": "MobileNetV2",
'''

new_result = '''            "model_results": model_results,
            "non_tomato_votes": sum(
                1 for item in model_results
                if item.get("prediction") == "Not Tomato Leaf"
            ),
            "verification_rule": "At least 5 of 8 models must identify a non-tomato image.",
            "primary_model": "Eight-model verification gate with MobileNetV2 diagnosis",
'''

if old_result not in code:
    raise SystemExit(
        "ERROR: Prediction response section not found. "
        "No changes saved."
    )

code = code.replace(old_result, new_result, 1)

path.write_text(code, encoding="utf-8")

print("SUCCESS: Eight-model non-tomato voting gate installed.")
