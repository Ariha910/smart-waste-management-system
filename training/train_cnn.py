"""
Training script for Custom CNN on TrashNet.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from models.cnn_model import build_custom_cnn
from training.preprocess import (
    compute_balanced_class_weights,
    create_train_val_test_split,
    ensure_dataset_exists,
    explore_dataset,
    get_data_generators,
)
from utils.visualization import plot_and_save_training_curves


def train_cnn(epochs=config.CNN_EPOCHS, batch_size=config.BATCH_SIZE):
    """
    Orchestrates the entire training lifecycle for Custom CNN:
    1. Validates and prepares dataset splits.
    2. Generates balanced class weights.
    3. Builds and compiles the Custom CNN architecture.
    4. Applies Early Stopping, Learning Rate Scheduling, and Model Checkpointing.
    5. Saves the best model weights and training curves.
    """
    print("\n" + "=" * 60)
    print("      TRAINING MODEL A: CUSTOM CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 60)

    # 1. Dataset verification & split
    ensure_dataset_exists()
    explore_dataset()
    create_train_val_test_split()

    # 2. Data generators
    train_gen, val_gen, _ = get_data_generators(batch_size=batch_size)
    class_weights = compute_balanced_class_weights()

    # 3. Model construction
    model = build_custom_cnn(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.CNN_LR,
    )
    model.summary()

    # 4. Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=str(config.CNN_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

    # 5. Model training
    print(f"\nStarting CNN training for up to {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final model if not already saved by checkpoint
    if not config.CNN_MODEL_PATH.exists():
        model.save(str(config.CNN_MODEL_PATH))

    # 6. Save training history metrics
    history_dict = {
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "lr": [float(x) for x in history.history.get("learning_rate", history.history.get("lr", []))],
    }
    history_file = config.METRICS_DIR / "cnn_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"Saved CNN training history to: {history_file}")

    # 7. Generate and save training/validation plots
    plot_path = config.GRAPHS_DIR / "cnn_training_curves.png"
    plot_and_save_training_curves(history_dict, "Custom CNN", plot_path)
    print(f"Saved CNN training curve plot to: {plot_path}")

    return model, history_dict


if __name__ == "__main__":
    train_cnn()
