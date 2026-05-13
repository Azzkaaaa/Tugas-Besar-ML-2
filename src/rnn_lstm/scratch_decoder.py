import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, SimpleRNN, LSTM, Reshape
import os

def build_decoder_model(vocab_size, max_seq_len, embed_dim, rnn_type, num_layers, hidden_size):
    """
    Membangun arsitektur pre-inject decoder sesuai dokumen spesifikasi.
    """
    cnn_input = Input(shape=(2048,), name="cnn_features_input")
    # max_seq_len_input adalah (panjang_target - 1) karena t=-1 diisi oleh CNN
    caption_input = Input(shape=(max_seq_len - 1,), name="caption_sequence_input")

    # (1, embed_dim)
    cnn_projected = Dense(embed_dim, activation=None, name="cnn_projection")(cnn_input)
    cnn_projected = Reshape((1, embed_dim), name="cnn_reshape")(cnn_projected)

    # (max_seq_len - 1, embed_dim)
    caption_embedded = Embedding(input_dim=vocab_size, output_dim=embed_dim, name="caption_embedding")(caption_input)

    # (1 + max_seq_len - 1, embed_dim) = (max_seq_len, embed_dim)
    concat_input = Concatenate(axis=1, name="pre_inject_concat")([cnn_projected, caption_embedded])

    x = concat_input
    for i in range(num_layers):
        if rnn_type == "LSTM":
                                    # v prediksi seluruh timestep
            x = LSTM(hidden_size, return_sequences=True, name=f"lstm_layer_{i+1}")(x)
        else:
            x = SimpleRNN(hidden_size, return_sequences=True, name=f"rnn_layer_{i+1}")(x)

    outputs = Dense(vocab_size, activation="softmax", name="dense_output")(x)

    model = Model(inputs=[cnn_input, caption_input], outputs=outputs, name=f"{rnn_type}_decoder")

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    return model