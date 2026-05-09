import tensorflow as tf
import math

@tf.keras.utils.register_keras_serializable(package="Custom")
class KerasLocallyConnected2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        activation=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding.lower()
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        _, input_h, input_w, input_channels = input_shape

        if input_h is None or input_w is None or input_channels is None:
            raise ValueError("Input height, width, dan channel harus diketahui.")

        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == "valid":
            output_h = (input_h - kh) // sh + 1
            output_w = (input_w - kw) // sw + 1
        elif self.padding == "same":
            output_h = math.ceil(input_h / sh)
            output_w = math.ceil(input_w / sw)
        else:
            raise ValueError(f"Padding tidak dikenal: {self.padding}")

        self.output_h = output_h
        self.output_w = output_w
        self.num_positions = output_h * output_w
        self.patch_size = kh * kw * input_channels

        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.num_positions, self.patch_size, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.num_positions, self.filters),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        kh, kw = self.kernel_size
        sh, sw = self.strides

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, kh, kw, 1],
            strides=[1, sh, sw, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper(),
        )

        batch_size = tf.shape(inputs)[0]

        patches = tf.reshape(
            patches,
            (batch_size, self.num_positions, self.patch_size)
        )

        output = tf.einsum("npk,pkf->npf", patches, self.kernel)
        output = output + self.bias

        output = tf.reshape(
            output,
            (batch_size, self.output_h, self.output_w, self.filters)
        )

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation_name,
        })
        return config