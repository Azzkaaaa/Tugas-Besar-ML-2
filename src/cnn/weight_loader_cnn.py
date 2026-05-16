import tensorflow as tf
from pathlib import Path

from src.cnn.scratch_layers_cnn import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    LocallyConnected2D,
)
from src.cnn.scratch_models import ScratchSequential
from src.common.dense import Dense
from src.cnn.keras_layers import KerasLocallyConnected2D


def build_scratch_from_keras(keras_model_or_path, config):
    keras_model = _load_keras(keras_model_or_path)

    scratch_layers = []
    n_conv = config["conv_layers"]
    pooling = config["pooling"]

    for i in range(n_conv):
        conv_w, conv_b = keras_model.get_layer(f"conv{i + 1}").get_weights()
        scratch_layers.append(Conv2D(conv_w, conv_b, padding="same", activation="relu"))

        if pooling == "max":
            scratch_layers.append(MaxPooling2D(pool_size=(2, 2)))
        else:
            scratch_layers.append(AveragePooling2D(pool_size=(2, 2)))

    # transition layer
    scratch_layers.append(_get_transition_layer(keras_model))

    # dense layers
    dense1_w, dense1_b = keras_model.get_layer("dense1").get_weights()
    output_w, output_b = keras_model.get_layer("output").get_weights()
    scratch_layers.append(Dense(dense1_w, dense1_b, activation="relu"))
    scratch_layers.append(Dense(output_w, output_b, activation="softmax"))

    return ScratchSequential(scratch_layers)


def build_scratch_local_from_keras(keras_model_or_path, config):
    keras_model = _load_keras(keras_model_or_path)

    scratch_layers = []
    n_conv = config["conv_layers"]
    kernel_sizes = config["kernel_sizes"]
    pooling = config["pooling"]

    for i in range(n_conv):
        local_w, local_b = keras_model.get_layer(f"local{i + 1}").get_weights()
        scratch_layers.append(
            LocallyConnected2D(
                weights=local_w,
                bias=local_b,
                kernel_size=(kernel_sizes[i], kernel_sizes[i]),
                padding="same",
                activation="relu",
            )
        )

        if pooling == "max":
            scratch_layers.append(MaxPooling2D(pool_size=(2, 2)))
        else:
            scratch_layers.append(AveragePooling2D(pool_size=(2, 2)))

    scratch_layers.append(_get_transition_layer(keras_model))

    dense1_w, dense1_b = keras_model.get_layer("dense1").get_weights()
    output_w, output_b = keras_model.get_layer("output").get_weights()
    scratch_layers.append(Dense(dense1_w, dense1_b, activation="relu"))
    scratch_layers.append(Dense(output_w, output_b, activation="softmax"))

    return ScratchSequential(scratch_layers)


def build_scratch_from_keras_model(keras_model_path):
    config = {
        "conv_layers": 2,
        "kernel_sizes": [3, 3],
        "pooling": "max",
    }
    return build_scratch_from_keras(keras_model_path, config)

def _load_keras(model_or_path):
    """Load Keras model dari path atau return jika sudah model."""
    if isinstance(model_or_path, (str, Path)):
        return tf.keras.models.load_model(
            str(model_or_path),
            custom_objects={"KerasLocallyConnected2D": KerasLocallyConnected2D},
        )
    return model_or_path


def _get_transition_layer(keras_model):
    """Deteksi layer transisi (Flatten / GlobalPooling) dari Keras model."""
    layer_names = {layer.name for layer in keras_model.layers}
    layer_types = {layer.__class__.__name__ for layer in keras_model.layers}

    if "flatten" in layer_names or "Flatten" in layer_types:
        return Flatten()
    if "global_average_pooling2d" in layer_names or "GlobalAveragePooling2D" in layer_types:
        return GlobalAveragePooling2D()
    if "global_max_pooling2d" in layer_names or "GlobalMaxPooling2D" in layer_types:
        return GlobalMaxPooling2D()

    raise ValueError(
        "Layer transisi tidak ditemukan. "
        "Gunakan Flatten, GlobalAveragePooling2D, atau GlobalMaxPooling2D."
    )
