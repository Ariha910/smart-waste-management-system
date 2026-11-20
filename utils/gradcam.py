"""
Grad-CAM (Gradient-weighted Class Activation Mapping) module.
Dynamically computes gradient-based visual explanations for Custom CNN and MobileNetV2 models.
"""

import os
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

import config


def get_last_conv_layer_name(model):
    """
    Dynamically finds the last convolutional layer in a Keras Model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    # Fallback search by layer name
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("Could not find a convolutional layer in the provided model.")


def compute_gradcam(model, img_batch, class_index=None, model_type="cnn"):
    """
    Generates a Grad-CAM heatmap for an input image batch.

    Args:
        model: Loaded tf.keras.Model.
        img_batch: Preprocessed numpy array of shape (1, 224, 224, 3) with raw RGB values [0, 255].
        class_index: Target class index (0-5) to explain. If None, uses model's top predicted class.
        model_type: 'cnn' or 'mobilenet'.

    Returns:
        tuple: (heatmap, target_class_index, top_class_index, class_probabilities)
    """
    model_type = model_type.lower()

    if model_type == "cnn":
        # In Custom CNN, rescaling (1/255) is inside the model
        last_conv_layer_name = "conv4_1"
        try:
            target_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            last_conv_layer_name = get_last_conv_layer_name(model)
            target_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch, training=False)
            probs = predictions[0].numpy()
            top_class = int(np.argmax(probs))
            target_class = top_class if class_index is None else int(class_index)
            loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    elif model_type in ["mobilenet", "mobilenetv2"]:
        # MobileNetV2 uses a nested Functional base model
        base_layer = None
        for l in model.layers:
            if "mobilenetv2" in l.name.lower() or (hasattr(l, "layers") and len(l.layers) > 10):
                base_layer = l
                break

        if base_layer is None:
            raise ValueError("MobileNetV2 base model layer not found in the loaded model.")

        # Last conv activation in MobileNetV2 is 'out_relu'
        try:
            target_sublayer = base_layer.get_layer("out_relu")
        except ValueError:
            target_sublayer = base_layer.layers[-1]

        sub_grad_model = tf.keras.models.Model(
            inputs=base_layer.inputs,
            outputs=[target_sublayer.output, base_layer.output],
        )

        gap_layer = model.get_layer("gap")
        bn_gap = model.get_layer("bn_gap")
        d256 = model.get_layer("dense_256")
        bn_d = model.get_layer("bn_dense")
        d128 = model.get_layer("dense_128")
        out_layer = model.get_layer("classification_output")

        # MobileNetV2 requires [-1, 1] normalization
        x_norm = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.copy())

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = sub_grad_model(x_norm, training=False)
            tape.watch(conv_outputs)
            h = gap_layer(conv_outputs)
            h = bn_gap(h, training=False)
            h = d256(h)
            h = bn_d(h, training=False)
            h = d128(h)
            predictions = out_layer(h)
            probs = predictions[0].numpy()
            top_class = int(np.argmax(probs))
            target_class = top_class if class_index is None else int(class_index)
            loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Rectified Linear Unit (ReLU) to retain features that positively influence the target class
    cam = np.maximum(cam.numpy(), 0)
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam)

    return cam, target_class, top_class, probs


def overlay_gradcam(original_pil_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Blends a 2D Grad-CAM heatmap with the original image.

    Args:
        original_pil_img: PIL Image in RGB format.
        heatmap: 2D numpy array of shape (H, W) in [0.0, 1.0].
        alpha: Float between 0.0 and 1.0 controlling heatmap opacity.
        colormap: OpenCV colormap enum (default: cv2.COLORMAP_JET).

    Returns:
        tuple: (heatmap_colored_pil, overlay_pil)
    """
    orig_np = np.array(original_pil_img)
    if orig_np.ndim == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w = orig_np.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap (OpenCV uses BGR)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Create overlay
    overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)

    return Image.fromarray(heatmap_colored_rgb), Image.fromarray(overlay)
