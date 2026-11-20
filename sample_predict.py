"""
Sample Prediction Script
Runs inference on representative test samples and prints classification results,
confidence scores, and class probability distributions.
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from utils.prediction import predict_waste


def test_sample_predictions():
    """
    Evaluates representative sample images across all six waste categories.
    """
    sample_dir = config.BASE_DIR / "data" / "sample_test_images"
    if not sample_dir.exists():
        print(f"Sample directory not found: {sample_dir}")
        return

    sample_files = sorted(list(sample_dir.glob("*.*")))
    if not sample_files:
        print("No sample images found.")
        return

    # Determine which model is available to test
    available_models = []
    if config.MOBILENET_MODEL_PATH.exists():
        available_models.append("mobilenet")
    if config.CNN_MODEL_PATH.exists():
        available_models.append("cnn")

    if not available_models:
        print("No trained models found in saved_models/. Please wait for training to complete.")
        return

    for model_key in available_models:
        model_display = "MobileNetV2 (Transfer Learning)" if model_key == "mobilenet" else "Custom CNN"
        print("\n" + "=" * 70)
        print(f"   SAMPLE INFERENCE BENCHMARK - {model_display.upper()}")
        print("=" * 70)

        correct_count = 0
        total_count = len(sample_files)

        for img_path in sample_files:
            # Extract true class from filename, e.g. sample_glass.jpg -> glass
            true_label = img_path.stem.replace("sample_", "").lower()

            try:
                res = predict_waste(img_path, model_type=model_key)
                pred_label = res["predicted_class"]
                conf = res["confidence"]
                is_correct = (pred_label == true_label)
                if is_correct:
                    correct_count += 1

                status_icon = "✅ MATCH" if is_correct else "❌ MISMATCH"
                print(f"\nImage: {img_path.name}")
                print(f"  ├─ Ground Truth:   {true_label.capitalize()}")
                print(f"  ├─ Model Result:   {pred_label.capitalize()} ({status_icon})")
                print(f"  ├─ Confidence:     {conf:.2%}")
                print(f"  └─ Top 3 Probabilities:")
                sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
                for c, p in sorted_probs:
                    print(f"       • {c.capitalize():<10}: {p:.2%}")

            except Exception as e:
                print(f"Error predicting {img_path.name}: {e}")

        print("\n" + "-" * 70)
        print(f"Sample Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.1%})")
        print("-" * 70)


if __name__ == "__main__":
    test_sample_predictions()

