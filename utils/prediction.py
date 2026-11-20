"""
Inference and prediction pipeline with input validation, out-of-distribution
uncertainty detection, multi-model execution, and model caching.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

import config

# Global cache for loaded models
_LOADED_MODELS = {}


def load_model_cached(model_type="mobilenet"):
    """
    Loads and caches Keras models in memory to avoid repeated disk I/O.
    model_type: 'mobilenet' or 'cnn'
    """
    model_type = model_type.lower()
    if model_type in _LOADED_MODELS:
        return _LOADED_MODELS[model_type]

    if model_type in ["mobilenet", "mobilenetv2"]:
        path = config.MOBILENET_MODEL_PATH
        model_name = "MobileNetV2"
        key = "mobilenet"
    elif model_type == "cnn":
        path = config.CNN_MODEL_PATH
        model_name = "Custom CNN"
        key = "cnn"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'mobilenet' or 'cnn'.")

    if not path.exists():
        raise FileNotFoundError(
            f"The {model_name} model was not found at {path}. "
            "Please train the model first by running the training script."
        )

    model = tf.keras.models.load_model(str(path))
    _LOADED_MODELS[key] = model
    return model


def preprocess_image_for_inference(image_input):
    """
    Validates, converts, and prepares an image for model inference.
    Supports PIL Image, file path string, or BytesIO buffer.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        # File buffer / BytesIO / UploadedFile
        img = Image.open(image_input)

    # Convert to RGB (handles RGBA PNGs, Grayscale, CMYK)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Basic input validation: Check minimum resolution and variance
    width, height = img.size
    if width < 32 or height < 32:
        raise ValueError(f"Image is too small ({width}x{height} pixels). Please upload an image with at least 32x32 resolution.")

    # Resize to model input shape (224, 224)
    img_resized = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)

    # Check for blank / solid color images
    std_dev = np.std(img_array)
    if std_dev < 3.0:
        raise ValueError("Image appears to be blank or a single solid color with no distinct visual features.")

    # Expand dims to batch shape: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch


def predict_waste(image_input, model_type="mobilenet", confidence_threshold=0.45):
    """
    Predicts waste class for an image with uncertainty detection.

    Returns:
        dict containing:
        - predicted_class: string
        - confidence: float (0.0 to 1.0)
        - probabilities: dict of {class_name: float}
        - is_uncertain: bool
        - warning_message: str or None
        - entropy: float
        - category_info: dict of metadata
        - original_image: PIL Image
        - model_used: string
    """
    orig_img, img_batch = preprocess_image_for_inference(image_input)
    model = load_model_cached(model_type)

    # Inference
    preds = model.predict(img_batch, verbose=0)[0]
    preds = [float(p) for p in preds]

    # Map to classes in fixed order
    prob_dict = {cls: float(preds[i]) for i, cls in enumerate(config.CLASSES)}

    # Top prediction
    top_idx = int(np.argmax(preds))
    top_class = config.CLASSES[top_idx]
    confidence = float(preds[top_idx])

    # Normalized Shannon entropy across the 6 classes: H(P) / log(K) in [0, 1]
    epsilon = 1e-12
    entropy = float(-sum(p * np.log(p + epsilon) for p in preds) / np.log(len(config.CLASSES)))

    is_uncertain = False
    warning_message = None

    if confidence < confidence_threshold or entropy > 0.85:
        is_uncertain = True
        warning_message = (
            f"Low confidence ({confidence:.1%}) or high entropy ({entropy:.2f}). "
            "Note: Softmax assigns probabilities constrained to sum to 1.0 across the 6 closed-set classes; "
            "it does not guarantee that the image belongs to the waste domain. "
            "Please confirm with manual visual verification."
        )

    return {
        "predicted_class": top_class,
        "predicted_index": top_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "is_uncertain": is_uncertain,
        "warning_message": warning_message,
        "entropy": entropy,
        "category_info": config.CATEGORY_INFO.get(top_class, {}),
        "original_image": orig_img,
        "model_used": "MobileNetV2 (Transfer Learning)" if model_type == "mobilenet" else "Custom CNN",
    }


def predict_both_models(image_input, confidence_threshold=0.45):
    """
    Runs both Custom CNN and MobileNetV2 on the same input image and assesses
    inter-model consensus/agreement.
    """
    orig_img, img_batch = preprocess_image_for_inference(image_input)

    cnn_res = predict_waste(orig_img, model_type="cnn", confidence_threshold=confidence_threshold)
    mob_res = predict_waste(orig_img, model_type="mobilenet", confidence_threshold=confidence_threshold)

    agree = (cnn_res["predicted_class"] == mob_res["predicted_class"])

    return {
        "original_image": orig_img,
        "cnn": cnn_res,
        "mobilenet": mob_res,
        "agree": agree,
        "consensus_class": cnn_res["predicted_class"] if agree else None,
    }
