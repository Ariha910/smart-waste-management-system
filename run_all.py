"""
End-to-End Pipeline Orchestrator for Smart Waste Classification.
Executes preprocessing, CNN training, MobileNetV2 training, evaluation, and sample testing.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from training.preprocess import (
    create_train_val_test_split,
    ensure_dataset_exists,
    explore_dataset,
)
from training.train_cnn import train_cnn
from training.train_mobilenet import train_mobilenet
from training.evaluate import run_evaluation
from sample_predict import test_sample_predictions
from utils.visualization import plot_and_save_category_distribution


def run_pipeline():
    start_time = time.time()
    print("=" * 75)
    print("   ECOSORT AI: SMART WASTE CLASSIFICATION AUTOMATED PIPELINE")
    print("=" * 75)

    # 1. Dataset Verification & Exploration
    print("\n[Step 1/5] Verifying TrashNet dataset...")
    ensure_dataset_exists()
    df_dist = explore_dataset()
    plot_and_save_category_distribution(df_dist, config.GRAPHS_DIR / "category_distribution.png")

    # 2. Stratified Splitting
    print("\n[Step 2/5] Creating stratified 70:15:15 split...")
    create_train_val_test_split()

    # 3. Train Model A: Custom CNN (if not already trained)
    print("\n[Step 3/5] Checking / Training Custom CNN...")
    if config.CNN_MODEL_PATH.exists() and (config.METRICS_DIR / "cnn_training_history.json").exists():
        print(f"Custom CNN model already exists at {config.CNN_MODEL_PATH}. Skipping retraining.")
    else:
        print("Training Custom CNN...")
        train_cnn(epochs=config.CNN_EPOCHS)

    # 4. Train Model B: MobileNetV2 (if not already trained)
    print("\n[Step 4/5] Checking / Training MobileNetV2...")
    if config.MOBILENET_MODEL_PATH.exists() and (config.METRICS_DIR / "mobilenet_training_history.json").exists():
        print(f"MobileNetV2 model already exists at {config.MOBILENET_MODEL_PATH}. Skipping retraining.")
    else:
        print("Training MobileNetV2 Transfer Learning model...")
        train_mobilenet(
            base_epochs=config.MOBILENET_BASE_EPOCHS,
            fine_epochs=config.MOBILENET_FINE_EPOCHS,
        )

    # 5. Independent Test Set Evaluation & Comparison
    print("\n[Step 5/5] Running Independent Test Set Benchmarking...")
    run_evaluation()

    # 6. Sample Prediction Demonstration
    print("\nRunning sample predictions on test set samples...")
    test_sample_predictions()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"   PIPELINE COMPLETE IN {elapsed / 60:.1f} MINUTES!")
    print("   Start the web dashboard with: streamlit run app.py")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()

