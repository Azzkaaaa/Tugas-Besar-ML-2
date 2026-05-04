import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.common.activations import apply_activation


class MaxPooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size

    def forward(self, x):
        is_single = False

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
            is_single = True

        ph, pw = self.pool_size
        sh, sw = self.strides

        windows = sliding_window_view(x, window_shape=(ph, pw), axis=(1, 2))
        windows = windows[:, ::sh, ::sw, :, :, :]

        output = np.max(windows, axis=(-2, -1))

        if is_single:
            return output[0]

        return output


class AveragePooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size

    def forward(self, x):
        is_single = False

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
            is_single = True

        ph, pw = self.pool_size
        sh, sw = self.strides

        windows = sliding_window_view(x, window_shape=(ph, pw), axis=(1, 2))
        windows = windows[:, ::sh, ::sw, :, :, :]

        output = np.mean(windows, axis=(-2, -1))

        if is_single:
            return output[0]

        return output


class Flatten:
    def forward(self, x):
        if x.ndim == 3:
            return x.reshape(-1)

        if x.ndim == 4:
            batch_size = x.shape[0]
            return x.reshape(batch_size, -1)

        raise ValueError(f"Flatten hanya menerima input 3D atau 4D, dapat shape {x.shape}")


class Conv2D:
    def __init__(self, weights, bias, strides=(1, 1), padding="valid", activation=None):
        self.weights = weights
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def _pad_input(self, x):
        if self.padding == "valid":
            return x

        if self.padding == "same":
            _, h, w, _ = x.shape
            kh, kw, _, _ = self.weights.shape
            sh, sw = self.strides

            out_h = int(np.ceil(h / sh))
            out_w = int(np.ceil(w / sw))

            pad_h = max((out_h - 1) * sh + kh - h, 0)
            pad_w = max((out_w - 1) * sw + kw - w, 0)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            return np.pad(
                x,
                (
                    (0, 0),
                    (pad_top, pad_bottom),
                    (pad_left, pad_right),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )

        raise ValueError(f"Padding tidak dikenal: {self.padding}")

    def forward(self, x):
        is_single = False

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
            is_single = True

        x_padded = self._pad_input(x)

        kh, kw, _, _ = self.weights.shape
        sh, sw = self.strides

        windows = sliding_window_view(
            x_padded,
            window_shape=(kh, kw),
            axis=(1, 2)
        )

        windows = windows[:, ::sh, ::sw, :, :, :]

        # Dari shape:
        # (N, out_h, out_w, C_in, kh, kw)
        # menjadi:
        # (N, out_h, out_w, kh, kw, C_in)
        windows = np.moveaxis(windows, 3, -1)

        output = np.tensordot(
            windows,
            self.weights,
            axes=([3, 4, 5], [0, 1, 2])
        )

        output = output + self.bias
        output = apply_activation(output, self.activation)

        if is_single:
            return output[0]

        return output


class GlobalAveragePooling2D:
    def forward(self, x):
        if x.ndim == 3:
            return np.mean(x, axis=(0, 1))

        if x.ndim == 4:
            return np.mean(x, axis=(1, 2))

        raise ValueError(f"Input harus 3D atau 4D, dapat shape {x.shape}")


class GlobalMaxPooling2D:
    def forward(self, x):
        if x.ndim == 3:
            return np.max(x, axis=(0, 1))

        if x.ndim == 4:
            return np.max(x, axis=(1, 2))

        raise ValueError(f"Input harus 3D atau 4D, dapat shape {x.shape}")