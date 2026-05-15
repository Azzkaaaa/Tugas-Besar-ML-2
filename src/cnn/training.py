import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.cnn.keras_models import build_cnn_from_config, build_local_cnn_from_config
from src.cnn.keras_layers import KerasLocallyConnected2D

def prepare_datasets(
    train_dir,
    test_dir,
    img_size=(150, 150),
    batch_size=32,
    val_split=0.2,
    seed=42,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    norm = layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (norm(x), y)).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (norm(x), y)).prefetch(autotune)
    test_ds = test_ds.map(lambda x, y: (norm(x), y)).prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def train_single_model(
    config,
    train_ds,
    val_ds,
    epochs=10,
    models_dir="models/cnn",
    results_dir="results/cnn",
    img_size=(150, 150),
    num_classes=6,
    force=False,
):
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    name = config["name"]
    model_path = models_dir / f"{name}.keras"
    weights_path = models_dir / f"{name}.weights.h5"
    history_path = results_dir / f"{name}_history.json"

    if model_path.exists() and history_path.exists() and not force:
        print(f"[SKIP] {name} — model & history sudah ada. Gunakan force=True untuk retrain.")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"KerasLocallyConnected2D": KerasLocallyConnected2D},
        )
        with open(history_path, "r") as f:
            history_dict = json.load(f)
        return model, history_dict

    print(f"\n{'='*60}")
    print(f"[TRAIN] {name}")
    print(f"  Conv layers : {config['conv_layers']}")
    print(f"  Filters     : {config['filters']}")
    print(f"  Kernel sizes: {config['kernel_sizes']}")
    print(f"  Pooling     : {config['pooling']}")
    print(f"  Epochs      : {epochs}")
    print(f"{'='*60}")

    input_shape = img_size + (3,)
    model = build_cnn_from_config(config, input_shape=input_shape, num_classes=num_classes)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    model.save(str(model_path))
    model.save_weights(str(weights_path))

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"[SAVED] Model  → {model_path}")
    print(f"[SAVED] History → {history_path}")

    return model, history_dict

def evaluate_model(model, test_ds, class_names=None):
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "num_params": int(model.count_params()),
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_evaluation(result, config, results_dir="results/cnn"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    name = config["name"]

    np.save(str(results_dir / f"{name}_y_true.npy"), result["y_true"])
    np.save(str(results_dir / f"{name}_y_pred.npy"), result["y_pred"])

    metrics = {
        "name": name,
        "config": config,
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "num_params": result["num_params"],
        "classification_report": result["classification_report"],
        "confusion_matrix": result["confusion_matrix"],
    }

    metrics_path = results_dir / f"{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SAVED] Metrics → {metrics_path}")
    return metrics

def run_all_experiments(
    configs,
    train_ds,
    val_ds,
    test_ds,
    class_names=None,
    epochs=10,
    models_dir="models/cnn",
    results_dir="results/cnn",
    img_size=(150, 150),
    num_classes=6,
    force=False,
):
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i}/{len(configs)}: {config['name']}")
        print(f"{'#'*60}")

        model, history_dict = train_single_model(
            config,
            train_ds,
            val_ds,
            epochs=epochs,
            models_dir=models_dir,
            results_dir=results_dir,
            img_size=img_size,
            num_classes=num_classes,
            force=force,
        )

        result = evaluate_model(model, test_ds, class_names=class_names)

        metrics = save_evaluation(result, config, results_dir=results_dir)
        metrics["history"] = history_dict
        all_results.append(metrics)

        print(f"  Macro F1 : {result['macro_f1']:.4f}")
        print(f"  Accuracy : {result['accuracy']:.4f}")
        print(f"  Params   : {result['num_params']:,}")

        del model
        tf.keras.backend.clear_session()

    _save_summary(all_results, results_dir)

    return all_results


def _save_summary(all_results, results_dir):
    results_dir = Path(results_dir)

    summary = []
    for r in all_results:
        summary.append({
            "name": r["name"],
            "config": r["config"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "num_params": r["num_params"],
        })

    summary_path = results_dir / "all_experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] Summary → {summary_path}")

    ranked = sorted(summary, key=lambda x: x["macro_f1"], reverse=True)
    print("\n" + "=" * 70)
    print("RANKING (by Macro F1)")
    print("=" * 70)
    for i, r in enumerate(ranked, 1):
        print(f"  {i:2d}. {r['name']:<55s} F1={r['macro_f1']:.4f}")
    print("=" * 70)


def find_best_config(all_results):
    best = max(all_results, key=lambda x: x["macro_f1"])
    print(f"Best config: {best['name']} (F1={best['macro_f1']:.4f})")
    return best["config"]


def train_locally_connected(
    config,
    train_ds,
    val_ds,
    epochs=10,
    models_dir="models/cnn",
    results_dir="results/cnn",
    img_size=(150, 150),
    num_classes=6,
    force=False,
):
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    name = config["name"].replace("conv", "local")
    model_path = models_dir / f"{name}.keras"
    history_path = results_dir / f"{name}_history.json"

    if model_path.exists() and history_path.exists() and not force:
        print(f"[SKIP] {name} — sudah ada.")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"KerasLocallyConnected2D": KerasLocallyConnected2D},
        )
        with open(history_path, "r") as f:
            history_dict = json.load(f)
        return model, history_dict

    print(f"\n{'='*60}")
    print(f"[TRAIN] LocallyConnected: {name}")
    print(f"{'='*60}")

    input_shape = img_size + (3,)
    model = build_local_cnn_from_config(config, input_shape=input_shape, num_classes=num_classes)
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save(str(model_path))

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"[SAVED] {model_path}")
    return model, history_dict

def load_all_results(configs, results_dir="results/cnn"):
    results_dir = Path(results_dir)
    all_results = []

    for config in configs:
        name = config["name"]
        metrics_path = results_dir / f"{name}_metrics.json"
        history_path = results_dir / f"{name}_history.json"

        if not metrics_path.exists():
            print(f"[WARN] {name}: metrics belum ada, skip.")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if history_path.exists():
            with open(history_path, "r") as f:
                metrics["history"] = json.load(f)
        else:
            metrics["history"] = None

        all_results.append(metrics)

    return all_results
