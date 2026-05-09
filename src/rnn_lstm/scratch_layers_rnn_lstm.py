import numpy as np
from src.common.activations import apply_activation, sigmoid, tanh

class Embedding:
    def __init__(self, weights):
        """
        weights: numpy array dengan shape (vocab_size, embed_dim)
        """
        self.weights = weights

    def forward(self, inputs):
        return self.weights[inputs]


class SimpleRNNCell:
    def __init__(self, kernel, recurrent_kernel, bias, activation="tanh"):
        self.kernel = kernel                      # shape: (input_dim, units)
        self.recurrent_kernel = recurrent_kernel  # shape: (units, units)
        self.bias = bias                          # shape: (units,)
        self.activation = activation

    def forward(self, x_t, h_prev):
        z = np.dot(x_t, self.kernel) + np.dot(h_prev, self.recurrent_kernel) + self.bias
        h_t = apply_activation(z, self.activation)
        return h_t


class SimpleRNN:
    def __init__(self, kernel, recurrent_kernel, bias, return_sequences=False, activation="tanh"):
        self.cell = SimpleRNNCell(kernel, recurrent_kernel, bias, activation)
        self.return_sequences = return_sequences
        self.units = recurrent_kernel.shape[0]

    def forward(self, inputs, initial_state=None):
        # shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = inputs.shape

        if initial_state is None:
            h_t = np.zeros((batch_size, self.units))
        else:
            h_t = initial_state

        outputs = []
        for t in range(seq_len):
            h_t = self.cell.forward(inputs[:, t, :], h_t)
            outputs.append(h_t)

        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return h_t


class LSTMCell:
    def __init__(self, kernel, recurrent_kernel, bias):
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias
        self.units = recurrent_kernel.shape[0]

    def forward(self, x_t, h_prev, c_prev):
        z = np.dot(x_t, self.kernel) + np.dot(h_prev, self.recurrent_kernel) + self.bias

        # urutan: input (i), forget (f), cell (c), output (o)
        i_gate = z[:, :self.units]
        f_gate = z[:, self.units:self.units*2]
        c_gate = z[:, self.units*2:self.units*3]
        o_gate = z[:, self.units*3:]

        i       = sigmoid(i_gate)
        f       = sigmoid(f_gate)
        c_can   = tanh(c_gate)
        o       = sigmoid(o_gate)

        c_t = f * c_prev + i * c_can
        h_t = o * tanh(c_t)

        return h_t, c_t


class LSTM:
    def __init__(self, kernel, recurrent_kernel, bias, return_sequences=False):
        self.cell = LSTMCell(kernel, recurrent_kernel, bias)
        self.return_sequences = return_sequences
        self.units = recurrent_kernel.shape[0]

    def forward(self, inputs, initial_state=None):
        # shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = inputs.shape

        # h_0 dan c_0 diinisialisasi zeros jika tidak ada dari encoder
        if initial_state is None:
            h_t = np.zeros((batch_size, self.units))
            c_t = np.zeros((batch_size, self.units))
        else:
            h_t, c_t = initial_state

        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell.forward(inputs[:, t, :], h_t, c_t)
            outputs.append(h_t)

        if self.return_sequences:
            return np.stack(outputs, axis=1), (h_t, c_t)
        else:
            return h_t, (h_t, c_t)