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

# def build_local_cnn_small(input_shape=(150, 150, 3), num_classes=6):
#     model = models.Sequential([
#         layers.Input(shape=input_shape),

#         KerasLocallyConnected2D(
#             filters=8,
#             kernel_size=(3, 3),
#             padding="same",
#             activation="relu",
#             name="local1"
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

#         KerasLocallyConnected2D(
#             filters=16,
#             kernel_size=(3, 3),
#             padding="same",
#             activation="relu",
#             name="local2"
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

#         layers.Flatten(name="flatten"),
#         layers.Dense(64, activation="relu", name="dense1"),
#         layers.Dense(num_classes, activation="softmax", name="output"),
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     return model

