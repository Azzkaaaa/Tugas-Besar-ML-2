import tensorflow as tf
from tensorflow.keras import layers, models

from src.cnn.keras_layers import KerasLocallyConnected2D


def build_baseline_cnn(input_shape=(150, 150, 3), num_classes=6):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_local_cnn(input_shape=(150, 150, 3), num_classes=6):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        KerasLocallyConnected2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="local1"
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        KerasLocallyConnected2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="local2"
        ),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def build_cnn_from_config(config, input_shape=(150, 150, 3), num_classes=6):
    """
    Build model CNN dari config dict.
    config keys: conv_layers, filters, kernel_sizes, pooling.
    """
    layer_list = [layers.Input(shape=input_shape)]

    n_conv = config["conv_layers"]
    filt = config["filters"]
    ks = config["kernel_sizes"]
    pooling = config["pooling"]

    for i in range(n_conv):
        layer_list.append(
            layers.Conv2D(
                filt[i],
                kernel_size=(ks[i], ks[i]),
                padding="same",
                activation="relu",
                name=f"conv{i + 1}",
            )
        )

        if pooling == "max":
            layer_list.append(layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{i + 1}"))
        else:
            layer_list.append(layers.AveragePooling2D(pool_size=(2, 2), name=f"pool{i + 1}"))

    layer_list.append(layers.Flatten(name="flatten"))
    layer_list.append(layers.Dense(128, activation="relu", name="dense1"))
    layer_list.append(layers.Dense(num_classes, activation="softmax", name="output"))

    model = models.Sequential(layer_list)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_local_cnn_from_config(config, input_shape=(150, 150, 3), num_classes=6):
    """
    Build model CNN dengan LocallyConnected2D (non-shared parameter).
    Arsitektur sama dengan build_cnn_from_config tapi Conv2D → LocallyConnected2D.
    """
    layer_list = [layers.Input(shape=input_shape)]

    n_conv = config["conv_layers"]
    filt = config["filters"]
    ks = config["kernel_sizes"]
    pooling = config["pooling"]

    for i in range(n_conv):
        layer_list.append(
            KerasLocallyConnected2D(
                filters=filt[i],
                kernel_size=(ks[i], ks[i]),
                padding="same",
                activation="relu",
                name=f"local{i + 1}",
            )
        )

        if pooling == "max":
            layer_list.append(layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{i + 1}"))
        else:
            layer_list.append(layers.AveragePooling2D(pool_size=(2, 2), name=f"pool{i + 1}"))

    layer_list.append(layers.Flatten(name="flatten"))
    layer_list.append(layers.Dense(128, activation="relu", name="dense1"))
    layer_list.append(layers.Dense(num_classes, activation="softmax", name="output"))

    model = models.Sequential(layer_list)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
