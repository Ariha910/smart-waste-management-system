"""
Training script for MobileNetV2 Transfer Learning and Fine-Tuning on TrashNet.
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
from models.mobilenet_model import build_mobilenet_model, unfreeze_mobilenet_top_layers
from training.preprocess import (
    compute_balanced_class_weights,
    create_train_val_test_split,
    ensure_dataset_exists,
    explore_dataset,
    get_data_generators,
)
from utils.visualization import plot_and_save_training_curves


def train_mobilenet(
    base_epochs=config.MOBILENET_BASE_EPOCHS,
    fine_epochs=config.MOBILENET_FINE_EPOCHS,
    batch_size=config.BATCH_SIZE,
):
    """
    Two-stage training workflow for MobileNetV2:
    - Phase 1: Feature Extraction. Base layers are frozen, training only the custom classification head.
    - Phase 2: Fine-Tuning. Unfreezes upper convolutional blocks of MobileNetV2 with a small learning rate.
    """
    print("\n" + "=" * 60)
    print("      TRAINING MODEL B: MOBILENETV2 (TRANSFER LEARNING)")
    print("=" * 60)

    # 1. Dataset verification & split
    ensure_dataset_exists()
    create_train_val_test_split()

    # 2. Data generators & class weights
    train_gen, val_gen, _ = get_data_generators(batch_size=batch_size)
    class_weights = compute_balanced_class_weights()

    # 3. Model construction - Phase 1: Feature Extraction
    print("\n[Phase 1] Building MobileNetV2 with frozen base layers...")
    model, base_model = build_mobilenet_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.MOBILENET_BASE_LR,
    )
    model.summary()

    # Callbacks for Phase 1
    checkpoint_phase1 = ModelCheckpoint(
        filepath=str(config.MOBILENET_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping_phase1 = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_phase1 = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    print(f"\n[Phase 1] Training classification head for up to {base_epochs} epochs...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=base_epochs,
        callbacks=[checkpoint_phase1, early_stopping_phase1, reduce_lr_phase1],
        class_weight=class_weights,
        verbose=1,
    )

    # 4. Phase 2: Fine-Tuning
    print("\n[Phase 2] Unfreezing top 35 layers for fine-tuning...")
    model = unfreeze_mobilenet_top_layers(
        model,
        num_unfreeze_layers=35,
        fine_tune_lr=config.MOBILENET_FINE_LR,
    )

    checkpoint_phase2 = ModelCheckpoint(
        filepath=str(config.MOBILENET_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping_phase2 = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_phase2 = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    initial_epoch = len(history1.history.get("accuracy", []))
    total_epochs = initial_epoch + fine_epochs

    print(f"\n[Phase 2] Fine-tuning from epoch {initial_epoch} to {total_epochs}...")
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_phase2, early_stopping_phase2, reduce_lr_phase2],
        class_weight=class_weights,
        verbose=1,
    )

    # 5. Consolidate training history
    combined_history = {
        "accuracy": [float(x) for x in history1.history.get("accuracy", []) + history2.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history1.history.get("val_accuracy", []) + history2.history.get("val_accuracy", [])],
        "loss": [float(x) for x in history1.history.get("loss", []) + history2.history.get("loss", [])],
        "val_loss": [float(x) for x in history1.history.get("val_loss", []) + history2.history.get("val_loss", [])],
    }

    # Save final model if checkpoint didn't trigger on last step
    if not config.MOBILENET_MODEL_PATH.exists():
        model.save(str(config.MOBILENET_MODEL_PATH))

    history_file = config.METRICS_DIR / "mobilenet_training_history.json"
    with open(history_file, "w") as f:
        json.dump(combined_history, f, indent=4)
    print(f"Saved MobileNetV2 training history to: {history_file}")

    # 6. Save plots
    plot_path = config.GRAPHS_DIR / "mobilenet_training_curves.png"
    plot_and_save_training_curves(combined_history, "MobileNetV2 (Transfer Learning)", plot_path)
    print(f"Saved MobileNetV2 training curve plot to: {plot_path}")

    return model, combined_history


if __name__ == "__main__":
    train_mobilenet()
