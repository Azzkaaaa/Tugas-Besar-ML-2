import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
})


def plot_loss_curves(history_dict, title="", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    epochs = range(1, len(history_dict["loss"]) + 1)
    ax.plot(epochs, history_dict["loss"], "o-", label="Training Loss", markersize=3)
    ax.plot(epochs, history_dict["val_loss"], "s-", label="Validation Loss", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_all_loss_curves(all_results, save_path=None):
    n = len(all_results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, result in enumerate(all_results):
        ax = axes[i]
        h = result.get("history")
        if h is None:
            ax.text(0.5, 0.5, "No history", ha="center", va="center")
            ax.set_title(result["name"], fontsize=8)
            continue

        epochs = range(1, len(h["loss"]) + 1)
        ax.plot(epochs, h["loss"], "o-", label="Train", markersize=2, linewidth=1)
        ax.plot(epochs, h["val_loss"], "s-", label="Val", markersize=2, linewidth=1)
        ax.set_title(result["name"], fontsize=7)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Training vs Validation Loss — Semua Variasi", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()


def _group_results(all_results, param_key):
    """
    param_key: "conv_layers", "filters", "kernel_sizes", "pooling".
    returns {param_value_str: [f1_scores]}
    """
    groups = {}
    for r in all_results:
        cfg = r["config"]

        if param_key == "filters":
            val = "-".join(map(str, cfg["filters"][:2]))  # base filters saja
        elif param_key == "kernel_sizes":
            val = str(cfg["kernel_sizes"][0])
        else:
            val = str(cfg[param_key])

        groups.setdefault(val, []).append(r["macro_f1"])
    return groups


def plot_hyperparameter_effect(all_results, param_key, title=None, save_path=None):
    """
    param_key : "conv_layers" | "filters" | "kernel_sizes" | "pooling"
    """
    LABELS = {
        "conv_layers": "Jumlah Layer Konvolusi",
        "filters": "Banyak Filter per Layer",
        "kernel_sizes": "Ukuran Filter (Kernel)",
        "pooling": "Jenis Pooling",
    }

    groups = _group_results(all_results, param_key)

    fig, ax = plt.subplots(figsize=(6, 4))

    x_labels = sorted(groups.keys())
    means = [np.mean(groups[k]) for k in x_labels]
    stds = [np.std(groups[k]) for k in x_labels]

    bars = ax.bar(x_labels, means, yerr=stds, capsize=5, alpha=0.85,
                  color=["#4e79a7", "#e15759", "#76b7b2", "#f28e2b"][:len(x_labels)],
                  edgecolor="white", linewidth=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel(LABELS.get(param_key, param_key))
    ax.set_ylabel("Macro F1-Score")
    ax.set_title(title or f"Pengaruh {LABELS.get(param_key, param_key)} terhadap F1")
    ax.grid(axis="y", alpha=0.3)

    ymin = max(0, min(means) - 0.1)
    ax.set_ylim(ymin, min(1.0, max(means) + 0.08))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_all_hyperparameter_effects(all_results, save_dir=None):
    PARAMS = ["conv_layers", "filters", "kernel_sizes", "pooling"]
    LABELS = {
        "conv_layers": "Jumlah Layer Konvolusi",
        "filters": "Banyak Filter per Layer",
        "kernel_sizes": "Ukuran Filter (Kernel)",
        "pooling": "Jenis Pooling",
    }
    COLORS = [
        ["#4e79a7", "#e15759", "#76b7b2"],
        ["#4e79a7", "#e15759", "#76b7b2"],
        ["#4e79a7", "#e15759", "#76b7b2"],
        ["#4e79a7", "#e15759", "#76b7b2"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, param_key, colors in zip(axes.flatten(), PARAMS, COLORS):
        groups = _group_results(all_results, param_key)
        x_labels = sorted(groups.keys())
        means = [np.mean(groups[k]) for k in x_labels]
        stds = [np.std(groups[k]) for k in x_labels]

        bars = ax.bar(x_labels, means, yerr=stds, capsize=5, alpha=0.85,
                      color=colors[:len(x_labels)], edgecolor="white", linewidth=0.5)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xlabel(LABELS[param_key], fontsize=9)
        ax.set_ylabel("Macro F1", fontsize=9)
        ax.set_title(f"Pengaruh {LABELS[param_key]}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        ymin = max(0, min(means) - 0.1)
        ax.set_ylim(ymin, min(1.0, max(means) + 0.08))

    fig.suptitle("Pengaruh Hyperparameter terhadap Macro F1-Score", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "hyperparameter_effects.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_f1_ranking(all_results, save_path=None):
    """Horizontal bar chart: ranking semua model berdasarkan F1."""
    sorted_results = sorted(all_results, key=lambda x: x["macro_f1"])

    names = [r["name"] for r in sorted_results]
    f1s = [r["macro_f1"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", linewidth=0.5)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{f1:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Macro F1-Score")
    ax.set_title("Ranking Model CNN — Macro F1-Score (Test Set)")
    ax.set_xlim(0, min(1.0, max(f1s) + 0.08))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()



def plot_shared_vs_nonshared(
    shared_metrics,
    nonshared_metrics,
    shared_history=None,
    nonshared_history=None,
    save_dir=None,
):
    """
    F1, jumlah parameter, dan loss curves.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = ["Shared\n(Conv2D)", "Non-Shared\n(LocallyConnected2D)"]
    colors = ["#4e79a7", "#e15759"]

    ax = axes[0]
    f1s = [shared_metrics["macro_f1"], nonshared_metrics["macro_f1"]]
    bars = ax.bar(labels, f1s, color=colors, alpha=0.85, edgecolor="white")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Perbandingan F1-Score")
    ax.grid(axis="y", alpha=0.3)
    ymin = max(0, min(f1s) - 0.1)
    ax.set_ylim(ymin, min(1.0, max(f1s) + 0.08))

    ax = axes[1]
    params = [shared_metrics["num_params"], nonshared_metrics["num_params"]]
    bars = ax.bar(labels, params, color=colors, alpha=0.85, edgecolor="white")
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{p:,}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Jumlah Parameter")
    ax.set_title("Perbandingan Jumlah Parameter")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    if shared_history and nonshared_history:
        e1 = range(1, len(shared_history["loss"]) + 1)
        e2 = range(1, len(nonshared_history["loss"]) + 1)
        ax.plot(e1, shared_history["val_loss"], "o-", color=colors[0],
                label="Shared (Val)", markersize=3)
        ax.plot(e2, nonshared_history["val_loss"], "s-", color=colors[1],
                label="Non-Shared (Val)", markersize=3)
        ax.plot(e1, shared_history["loss"], "o--", color=colors[0],
                label="Shared (Train)", markersize=2, alpha=0.5)
        ax.plot(e2, nonshared_history["loss"], "s--", color=colors[1],
                label="Non-Shared (Train)", markersize=2, alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "History tidak tersedia", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Perbandingan Loss Curves")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Shared vs Non-Shared Parameter", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "shared_vs_nonshared.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()

def plot_keras_vs_scratch(keras_f1, scratch_f1, model_name="", save_path=None):
    """Bar chart perbandingan Keras vs from scratch forward pass."""
    fig, ax = plt.subplots(figsize=(5, 4))

    labels = ["Keras", "From Scratch"]
    f1s = [keras_f1, scratch_f1]
    colors = ["#4e79a7", "#59a14f"]

    bars = ax.bar(labels, f1s, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.4f}", ha="center", fontweight="bold")

    ax.set_ylabel("Macro F1-Score")
    ax.set_title(f"Keras vs From Scratch — {model_name}" if model_name else "Keras vs From Scratch")
    ax.grid(axis="y", alpha=0.3)
    ymin = max(0, min(f1s) - 0.05)
    ax.set_ylim(ymin, min(1.0, max(f1s) + 0.05))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_confusion_matrix(cm, class_names=None, title="", save_path=None):
    """Plot confusion matrix sebagai heatmap."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = cm.shape[0]
    labels = class_names if class_names else [str(i) for i in range(n)]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()

def plot_loss_by_hyperparameter(all_results, param_key, save_dir=None):
    """
    Plot loss curves dikelompokkan berdasarkan hyperparameter.
    Satu subplot per nilai hyperparameter, di masing-masing subplot
    """
    LABELS = {
        "conv_layers": "Jumlah Layer Konvolusi",
        "filters": "Banyak Filter",
        "kernel_sizes": "Ukuran Kernel",
        "pooling": "Jenis Pooling",
    }

    # Group results
    groups = {}
    for r in all_results:
        cfg = r["config"]
        if param_key == "filters":
            val = "-".join(map(str, cfg["filters"][:2]))
        elif param_key == "kernel_sizes":
            val = str(cfg["kernel_sizes"][0])
        else:
            val = str(cfg[param_key])
        groups.setdefault(val, []).append(r)

    group_keys = sorted(groups.keys())
    n_groups = len(group_keys)

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 4))
    if n_groups == 1:
        axes = [axes]

    cmap = plt.cm.tab10

    for ax, key in zip(axes, group_keys):
        members = groups[key]
        for idx, r in enumerate(members):
            h = r.get("history")
            if h is None:
                continue
            epochs = range(1, len(h["val_loss"]) + 1)
            color = cmap(idx % 10)
            short_name = r["name"].replace(f"_{param_key}", "")
            ax.plot(epochs, h["val_loss"], "-", color=color,
                    label=r["name"][:30], linewidth=1.2)

        ax.set_title(f"{LABELS.get(param_key, param_key)} = {key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend(fontsize=5, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Validation Loss per {LABELS.get(param_key, param_key)}", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f"loss_by_{param_key}.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()
