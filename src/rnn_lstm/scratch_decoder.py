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

def run_training_experiments(X_cnn_train, X_caption_train, y_caption_train, vocab_size, max_seq_len, output_dir):
    """
    X_cnn_train: numpy array bentuk (N, 2048)
    X_caption_train: numpy array bentuk (N, max_seq_len-1)
    y_caption_train: numpy array bentuk (N, max_seq_len)
    """
    embed_dim = 256
    variations = {
        "num_layers": [1, 2, 3],
        "hidden_size": [128, 512]
    }

    os.makedirs(output_dir, exist_ok=True)

    for rnn_type in ["RNN", "LSTM"]:
        for layers in variations["num_layers"]:
            for hidden in variations["hidden_size"]:
                print(f"\n--- Training {rnn_type} | Layers: {layers} | Hidden: {hidden} ---")

                model = build_decoder_model(
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len, # Panjang target akhir
                    embed_dim=embed_dim,
                    rnn_type=rnn_type,
                    num_layers=layers,
                    hidden_size=hidden
                )

                # Training proses
                history = model.fit(
                    x=[X_cnn_train, X_caption_train],
                    y=y_caption_train,
                    batch_size=64,
                    epochs=10, # Sesuaikan epoch sesuai kebutuhan (perhatikan validasi loss)
                    validation_split=0.2 # Gunakan split data validasi Anda
                )

                weight_filename = f"{rnn_type}_L{layers}_H{hidden}_weights.h5"
                save_path = os.path.join(output_dir, weight_filename)
                model.save_weights(save_path)
                print(f"Bobot disimpan ke: {save_path}")