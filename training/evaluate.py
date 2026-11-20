"""
Evaluation script to benchmark Custom CNN vs MobileNetV2 on the independent test dataset.
Generates metrics, confusion matrices, classification reports, and comparison tables.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
from training.preprocess import (
    create_train_val_test_split,
    ensure_dataset_exists,
    get_data_generators,
)
from utils.visualization import (
    plot_and_save_comparison_chart,
    plot_and_save_confusion_matrix,
)


def evaluate_single_model(model_path, test_generator, model_name, cm_filename):
    """
    Evaluates a saved Keras model on the provided test generator.
    Computes true confusion matrices and metrics from model predictions.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")

    print(f"\nEvaluating {model_name} from: {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    # Reset generator to ensure deterministic ordering
    test_generator.reset()
    y_true = test_generator.classes
    num_samples = len(y_true)

    # Predict probabilities
    print(f"Generating predictions for {num_samples} test samples...")
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Accuracy, Precision, Recall, F1
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Classification report
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Save Confusion Matrix Plots
    cm_path_norm = config.CM_DIR / f"{cm_filename}_normalized.png"
    cm_path_raw = config.CM_DIR / f"{cm_filename}_raw.png"
    plot_and_save_confusion_matrix(cm, config.CLASSES, model_name, cm_path_norm, normalize=True)
    plot_and_save_confusion_matrix(cm, config.CLASSES, model_name, cm_path_raw, normalize=False)

    total_params = int(model.count_params())

    results = {
        "model_name": model_name,
        "test_samples": num_samples,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_parameters": total_params,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "cm_normalized_plot": str(cm_path_norm),
        "cm_raw_plot": str(cm_path_raw),
    }

    report_save_path = config.METRICS_DIR / f"{cm_filename}_report.json"
    with open(report_save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation report to: {report_save_path}")

    return results


def run_evaluation():
    """
    Evaluates both Custom CNN and MobileNetV2 on the same independent test set,
    generates side-by-side comparison tables and comparative bar charts.
    """
    print("\n" + "=" * 60)
    print("      INDEPENDENT TEST SET BENCHMARK & COMPARISON")
    print("=" * 60)

    # Ensure dataset split exists
    ensure_dataset_exists()
    create_train_val_test_split()

    _, _, test_gen = get_data_generators(batch_size=config.BATCH_SIZE)

    cnn_results = None
    mobilenet_results = None

    if config.CNN_MODEL_PATH.exists():
        cnn_results = evaluate_single_model(
            config.CNN_MODEL_PATH,
            test_gen,
            "Custom CNN",
            "cnn",
        )
    else:
        print(f"Warning: CNN model not found at {config.CNN_MODEL_PATH}. Skipping CNN evaluation.")

    if config.MOBILENET_MODEL_PATH.exists():
        mobilenet_results = evaluate_single_model(
            config.MOBILENET_MODEL_PATH,
            test_gen,
            "MobileNetV2 (Transfer Learning)",
            "mobilenet",
        )
    else:
        print(f"Warning: MobileNetV2 model not found at {config.MOBILENET_MODEL_PATH}. Skipping MobileNet evaluation.")

    # Build comparison summary
    comparison_rows = []
    if cnn_results:
        comparison_rows.append({
            "Model": "Custom CNN",
            "Accuracy": cnn_results["accuracy"],
            "Macro Precision": cnn_results["macro_precision"],
            "Macro Recall": cnn_results["macro_recall"],
            "Macro F1-Score": cnn_results["macro_f1"],
            "Weighted F1-Score": cnn_results["weighted_f1"],
            "Total Parameters": f"{cnn_results['total_parameters']:,}",
        })

    if mobilenet_results:
        comparison_rows.append({
            "Model": "MobileNetV2 (Transfer Learning)",
            "Accuracy": mobilenet_results["accuracy"],
            "Macro Precision": mobilenet_results["macro_precision"],
            "Macro Recall": mobilenet_results["macro_recall"],
            "Macro F1-Score": mobilenet_results["macro_f1"],
            "Weighted F1-Score": mobilenet_results["weighted_f1"],
            "Total Parameters": f"{mobilenet_results['total_parameters']:,}",
        })

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        print("\n" + "=" * 60)
        print("                FINAL MODEL COMPARISON TABLE")
        print("=" * 60)
        print(comp_df.to_string(index=False))

        # Save to JSON and CSV
        comp_df.to_json(config.COMPARISON_TABLE_PATH, orient="records", indent=4)
        comp_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
        print(f"\nSaved comparison table to: {config.COMPARISON_TABLE_PATH}")

        # If both models evaluated, produce comparison bar chart
        if cnn_results and mobilenet_results:
            chart_path = config.GRAPHS_DIR / "model_comparison_chart.png"
            plot_and_save_comparison_chart(comp_df, chart_path)
            print(f"Saved comparison chart to: {chart_path}")

    return cnn_results, mobilenet_results


if __name__ == "__main__":
    run_evaluation()
