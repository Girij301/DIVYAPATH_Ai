import numpy as np
import cv2
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "emotion_model.h5"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_model = None
_tf = None


def _load_model():
    global _model, _tf
    if _model is not None:
        return _model

    try:
        import tensorflow as tf
    except ImportError as ex:
        raise ImportError(
            "TensorFlow is required for emotion detection. "
            "Install a compatible TensorFlow build or run on a machine with AVX/AVX2 support."
        ) from ex

    _tf = tf
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes into model-ready tensor.
    """
    # Bytes → numpy
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode with OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))

    # Normalize
    normalized = resized / 255.0

    # Shape: (1, 48, 48, 1)
    final_img = normalized.reshape(1, 48, 48, 1).astype("float32")

    return final_img


def detect_emotion(image_bytes):
    """
    Returns: (emotion, confidence)
    """
    model = _load_model()
    processed = preprocess_image(image_bytes)

    preds = model.predict(processed, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    emotion = EMOTIONS[idx]
    return emotion, confidence