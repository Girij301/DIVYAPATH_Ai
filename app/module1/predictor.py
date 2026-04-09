import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] / "models"

MODEL_PATH = BASE_DIR / "student_model_v2.pkl"
GRADE_ENCODER_PATH = BASE_DIR / "grade_encoder_v2.pkl"
EXTRA_ENCODER_PATH = BASE_DIR / "extra_encoder_v2.pkl"
PARENT_ENCODER_PATH = BASE_DIR / "parent_encoder_v2.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    grade_enc = joblib.load(GRADE_ENCODER_PATH)
    extra_enc = joblib.load(EXTRA_ENCODER_PATH)
    parent_enc = joblib.load(PARENT_ENCODER_PATH)
    return model, grade_enc, extra_enc, parent_enc


def predict_grade(
    model,
    grade_enc,
    extra_enc,
    parent_enc,
    study_hours,
    attendance,
    previous_grade,
    extra,
    parent_edu,
):
    # Encode categorical inputs
    try:
        extra_val = extra_enc.transform([extra])[0]
    except ValueError:
        extra_val = 0  # default to 0
    try:
        parent_val = parent_enc.transform([parent_edu])[0]
    except ValueError:
        parent_val = 0  # default to 0

    # Prepare input
    input_data = np.array(
        [[study_hours, attendance, previous_grade, extra_val, parent_val]]
    )

    # Predict
    pred = model.predict(input_data)

    # Decode grade (A/B/C/D)
    try:
        grade = grade_enc.inverse_transform(pred)[0]
    except ValueError:
        grade = "C"  # default grade
    return grade
