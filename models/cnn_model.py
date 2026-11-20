"""
Custom Convolutional Neural Network (CNN) architecture for waste classification.
"""

import os
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_custom_cnn(
    input_shape=(224, 224, 3),
    num_classes=6,
    learning_rate=0.001,
    dropout_rate=0.3,
):
    """
    Build and compile a custom deep CNN for 6-class waste image classification.

    Architecture:
    - Input Layer & Normalization Rescaling (1./255)
    - Block 1: Conv2D (32, 3x3) -> BatchNorm -> ReLU -> Conv2D (32, 3x3) -> ReLU -> MaxPool (2x2)
    - Block 2: Conv2D (64, 3x3) -> BatchNorm -> ReLU -> Conv2D (64, 3x3) -> ReLU -> MaxPool (2x2)
    - Block 3: Conv2D (128, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Block 4: Conv2D (256, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Global Average Pooling -> Dense (256) -> BatchNorm -> Dropout -> Dense (128) -> Dropout
    - Output: Dense (6, softmax)
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Rescale pixel values from [0, 255] to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Conv Block 1
    x = layers.Conv2D(
        32, (3, 3), padding="same", kernel_initializer="he_normal", name="conv1_1"
    )(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(
        32, (3, 3), padding="same", kernel_initializer="he_normal", name="conv1_2"
    )(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Conv Block 2
    x = layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal", name="conv2_1"
    )(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(
        64, (3, 3), padding="same", kernel_initializer="he_normal", name="conv2_2"
    )(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Conv Block 3
    x = layers.Conv2D(
        128, (3, 3), padding="same", kernel_initializer="he_normal", name="conv3_1"
    )(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Conv Block 4
    x = layers.Conv2D(
        256, (3, 3), padding="same", kernel_initializer="he_normal", name="conv4_1"
    )(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256",
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_rate / 2, name="dense_dropout_2")(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="classification_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_Waste_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
