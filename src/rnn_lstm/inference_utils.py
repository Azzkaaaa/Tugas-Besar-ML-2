import time
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.common.dense import Dense
from src.rnn_lstm.scratch_layers_rnn_lstm import Embedding, SimpleRNNCell, LSTMCell

class ScratchDecoderInference:
    def __init__(self, vocab, rnn_type="RNN", num_layers=1, embed_dim=256, hidden_size=128):
        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        vocab_size = len(vocab)

        self.projection = Dense(np.zeros((2048, embed_dim)), np.zeros(embed_dim), activation=None)
        self.embedding = Embedding(np.zeros((vocab_size, embed_dim)))

        self.recurrent_cells = []
        for _ in range(num_layers):
            if rnn_type == "LSTM":
                self.recurrent_cells.append(LSTMCell(
                    np.zeros((embed_dim if _ == 0 else hidden_size, hidden_size * 4)),
                    np.zeros((hidden_size, hidden_size * 4)),
                    np.zeros(hidden_size * 4)
                ))
            else:
                self.recurrent_cells.append(SimpleRNNCell(
                    np.zeros((embed_dim if _ == 0 else hidden_size, hidden_size)),
                    np.zeros((hidden_size, hidden_size)),
                    np.zeros(hidden_size),
                    activation="tanh"
                ))

        self.dense_out = Dense(np.zeros((hidden_size, vocab_size)), np.zeros(vocab_size), activation="softmax")

    def load_weights_from_keras(self, keras_model):
        proj_w, proj_b = keras_model.get_layer("cnn_projection").get_weights()
        self.projection.weights = proj_w
        self.projection.bias = proj_b

        self.embedding.weights = keras_model.get_layer("caption_embedding").get_weights()[0]

        for i in range(self.num_layers):
            layer_name = f"{'lstm' if self.rnn_type == 'LSTM' else 'rnn'}_layer_{i+1}"
            k_weights = keras_model.get_layer(layer_name).get_weights()
            self.recurrent_cells[i].kernel = k_weights[0]
            self.recurrent_cells[i].recurrent_kernel = k_weights[1]
            self.recurrent_cells[i].bias = k_weights[2]

        out_w, out_b = keras_model.get_layer("dense_output").get_weights()
        self.dense_out.weights = out_w
        self.dense_out.bias = out_b

    def generate_caption(self, cnn_feature, max_length=40):
        states = []
        for _ in range(self.num_layers):
            if self.rnn_type == "LSTM":
                states.append((np.zeros((1, self.hidden_size)), np.zeros((1, self.hidden_size))))
            else:
                states.append(np.zeros((1, self.hidden_size)))

        # t = -1 (preinject)
        x_t = self.projection.forward(cnn_feature.reshape(1, -1))

        # forward w/o predicgt
        for i in range(self.num_layers):
            if self.rnn_type == "LSTM":
                h_t, c_t = self.recurrent_cells[i].forward(x_t, states[i][0], states[i][1])
                states[i] = (h_t, c_t)
                x_t = h_t
            else:
                h_t = self.recurrent_cells[i].forward(x_t, states[i])
                states[i] = h_t
                x_t = h_t

        current_word_idx = self.vocab["<start>"]
        caption = []

        for _ in range(max_length):
            x_t = self.embedding.forward(np.array([current_word_idx]))

            for i in range(self.num_layers):
                if self.rnn_type == "LSTM":
                    h_t, c_t = self.recurrent_cells[i].forward(x_t, states[i][0], states[i][1])
                    states[i] = (h_t, c_t)
                    x_t = h_t
                else:
                    h_t = self.recurrent_cells[i].forward(x_t, states[i])
                    states[i] = h_t
                    x_t = h_t

            # pred next word
            probs = self.dense_out.forward(x_t)
            next_word_idx = np.argmax(probs, axis=-1)[0]

            if next_word_idx == self.vocab["<end>"]:
                break

            word = self.rev_vocab.get(next_word_idx, "<unk>")
            caption.append(word)
            current_word_idx = next_word_idx

        return " ".join(caption)

def evaluate_model(model, X_cnn_test, y_captions_test_raw, max_length=40):
    """
    y_captions_test_raw: List of lists containing reference captions (string) per image.
    """
    predictions = []
    references = []

    start_time = time.time()

    for i in range(len(X_cnn_test)):
        # gen prediction
        pred_cap = model.generate_caption(X_cnn_test[i], max_length=max_length)
        predictions.append(pred_cap.split())

        img_refs = []
        for ref in y_captions_test_raw[i]:
            clean_ref = ref.replace("<start>", "").replace("<end>", "").strip().split()
            img_refs.append(clean_ref)
        references.append(img_refs)

    execution_time = time.time() - start_time

    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothie)

    return bleu_score, execution_time, predictions


def generate_caption_keras(keras_model, cnn_feature, vocab, rev_vocab, model_max_seq_len, limit_length):
    seq = np.zeros((1, model_max_seq_len - 1))
    seq[0, 0] = vocab["<start>"]
    caption = []

    stop_len = min(model_max_seq_len, limit_length)

    for i in range(1, stop_len):
        preds = keras_model.predict([cnn_feature.reshape(1, -1), seq], verbose=0)

        # get pred t-i
        next_word_idx = np.argmax(preds[0, i])

        if next_word_idx == vocab["<end>"]:
            break

        word = rev_vocab.get(next_word_idx, "<unk>")
        caption.append(word)

        if i < model_max_seq_len - 1:
            seq[0, i] = next_word_idx

    return " ".join(caption)

def evaluate_keras_model(keras_model, X_cnn_test, y_captions_test_raw, vocab, model_max_seq_len, limit_length=40):
    predictions = []
    references = []
    rev_vocab = {v: k for k, v in vocab.items()}

    start_time = time.time()

    for i in range(len(X_cnn_test)):
        pred_cap = generate_caption_keras(keras_model, X_cnn_test[i], vocab, rev_vocab, model_max_seq_len, limit_length)
        predictions.append(pred_cap.split())

        img_refs = []
        for ref in y_captions_test_raw[i]:
            clean_ref = ref.replace("<start>", "").replace("<end>", "").strip().split()
            img_refs.append(clean_ref)
        references.append(img_refs)

    execution_time = time.time() - start_time
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothie)

    return bleu_score, execution_time