import tensorflow as tf

from src.cnn.scratch_layers_cnn import Conv2D, MaxPooling2D, Flatten
from src.cnn.scratch_models import ScratchSequential
from src.common.dense import Dense


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