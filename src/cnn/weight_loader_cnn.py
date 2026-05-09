import tensorflow as tf

from src.cnn.scratch_layers_cnn import Conv2D, MaxPooling2D, Flatten, LocallyConnected2D, AveragePooling2D, GlobalAveragePooling2D
from src.cnn.scratch_models import ScratchSequential
from src.common.dense import Dense
from src.cnn.keras_layers import KerasLocallyConnected2D



def build_scratch_from_keras_model(keras_model_path):
    keras_model = tf.keras.models.load_model(keras_model_path)

    conv1_w, conv1_b = keras_model.get_layer("conv1").get_weights()
    conv2_w, conv2_b = keras_model.get_layer("conv2").get_weights()
    dense1_w, dense1_b = keras_model.get_layer("dense1").get_weights()
    output_w, output_b = keras_model.get_layer("output").get_weights()

    scratch_model = ScratchSequential([
        Conv2D(conv1_w, conv1_b, padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(conv2_w, conv2_b, padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(dense1_w, dense1_b, activation="relu"),
        Dense(output_w, output_b, activation="softmax"),
    ])

    return scratch_model

def build_scratch_local_from_keras_model(keras_model_or_path, config):
    if isinstance(keras_model_or_path, str):
        keras_model = tf.keras.models.load_model(
            keras_model_or_path,
            custom_objects={"KerasLocallyConnected2D": KerasLocallyConnected2D}
        )
    else:
        keras_model = keras_model_or_path

    layers = []

    conv_layers = config["conv_layers"]
    kernel_sizes = config["kernel_sizes"]
    pooling = config["pooling"]

    for i in range(conv_layers):
        local_w, local_b = keras_model.get_layer(f"local{i+1}").get_weights()

        layers.append(
            LocallyConnected2D(
                weights=local_w,
                bias=local_b,
                kernel_size=(kernel_sizes[i], kernel_sizes[i]),
                padding="same",
                activation="relu"
            )
        )

        if pooling == "max":
            layers.append(MaxPooling2D(pool_size=(2, 2)))
        elif pooling == "average":
            layers.append(AveragePooling2D(pool_size=(2, 2)))
        else:
            raise ValueError(f"Pooling tidak dikenal: {pooling}")

    layers.append(GlobalAveragePooling2D())

    dense1_w, dense1_b = keras_model.get_layer("dense1").get_weights()
    output_w, output_b = keras_model.get_layer("output").get_weights()

    layers.append(Dense(dense1_w, dense1_b, activation="relu"))
    layers.append(Dense(output_w, output_b, activation="softmax"))

    return ScratchSequential(layers)