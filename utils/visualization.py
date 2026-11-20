"""
Visualization utilities for model training curves, confusion matrices,
evaluation comparisons, and prediction probabilities.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import config

# Use modern matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_and_save_training_curves(history_dict, model_name, save_path):
    """
    Plots training & validation accuracy and loss side-by-side and saves to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2e7d32", linewidth=2.5, marker="o", markersize=4)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#00897b", linewidth=2.5, linestyle="--", marker="s", markersize=4)
    ax1.set_title(f"{model_name} - Accuracy Progression", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0, 1.02])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Loss Plot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#c62828", linewidth=2.5, marker="o", markersize=4)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#fb8c00", linewidth=2.5, linestyle="--", marker="s", markersize=4)
    ax2.set_title(f"{model_name} - Loss Progression", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_and_save_confusion_matrix(cm, class_names, model_name, save_path, normalize=True):
    """
    Renders and saves a stylish confusion matrix heatmap.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
        title_suffix = "(Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = "(Raw Counts)"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Greens",
        cbar=True,
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        linewidths=1,
        linecolor="#e0e0e0",
        square=True,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title(f"{model_name} - Confusion Matrix {title_suffix}", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("True Ground Truth Category", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Model Predicted Category", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_and_save_comparison_chart(comparison_df, save_path):
    """
    Plots a multi-metric bar chart comparing Custom CNN vs MobileNetV2.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    cnn_vals = [
        comparison_df.loc[comparison_df["Model"] == "Custom CNN", m].values[0]
        for m in metrics
    ]
    mobilenet_vals = [
        comparison_df.loc[comparison_df["Model"] == "MobileNetV2 (Transfer Learning)", m].values[0]
        for m in metrics
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    rects1 = ax.bar(x - width / 2, cnn_vals, width, label="Custom CNN", color="#43a047", edgecolor="#2e7d32")
    rects2 = ax.bar(x + width / 2, mobilenet_vals, width, label="MobileNetV2", color="#00897b", edgecolor="#004d40")

    ax.set_title("Model Architecture Performance Comparison on Test Set", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="semibold")
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.legend(frameon=True, loc="upper right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_and_save_category_distribution(dist_df, save_path):
    """
    Visualizes the raw class counts and proportions in TrashNet.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    palette = ["#8d6e63", "#0288d1", "#78909c", "#43a047", "#fb8c00", "#e53935"]

    bars = ax.bar(dist_df["Category"].str.capitalize(), dist_df["Count"], color=palette, edgecolor="#333", linewidth=0.8)
    ax.set_title("TrashNet Dataset Class Distribution (Total: 2,527 Images)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)} ({h / dist_df['Count'].sum() * 100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

