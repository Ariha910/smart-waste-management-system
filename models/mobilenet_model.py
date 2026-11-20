"""
MobileNetV2 Transfer Learning model architecture for waste classification.
"""

import os
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_mobilenet_model(
    input_shape=(224, 224, 3),
    num_classes=6,
    learning_rate=0.0005,
    dropout_rate=0.35,
):
    """
    Build a Transfer Learning model using MobileNetV2 pre-trained on ImageNet.

    Phase 1:
    - Base MobileNetV2 layers are frozen.
    - Custom classification head is trained on TrashNet features.

    Architecture:
    - Input (224x224x3)
    - MobileNetV2 Preprocessing [-1, 1]
    - Base MobileNetV2 (include_top=False, weights='imagenet')
    - GlobalAveragePooling2D
    - BatchNormalization
    - Dense (256, relu, L2)
    - Dropout (0.35)
    - Dense (128, relu)
    - Dropout (0.2)
    - Dense (6, softmax)
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # MobileNetV2 requires input values in [-1, 1] range
    x = preprocess_input(inputs)

    # Load pre-trained MobileNetV2 feature extractor
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze all base layers initially

    # Pass inputs through base model in inference mode
    x = base_model(x, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_gap")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256",
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_rate * 0.6, name="head_dropout_2")(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="classification_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Waste")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_mobilenet_top_layers(
    model,
    num_unfreeze_layers=35,
    fine_tune_lr=0.00005,
):
    """
    Unfreeze top N layers of the MobileNetV2 base model for fine-tuning.
    Keeps BatchNormalization layers in inference mode to avoid destroying learned statistics.
    Re-compiles the model with a smaller learning rate.
    """
    # Locate the MobileNetV2 base model within the graph
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find MobileNetV2 base model layer inside the model.")

    base_model.trainable = True

    # Freeze all layers except the last num_unfreeze_layers
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_unfreeze_layers)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # Keep BatchNorm frozen for stable fine-tuning
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    # Re-compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
