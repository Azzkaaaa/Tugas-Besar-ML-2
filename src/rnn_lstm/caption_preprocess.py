import os
import re
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^a-z\s]+', '', caption)
    caption = re.sub(r'\s+', ' ', caption).strip()

    return f"<start> {caption} <end>"

def build_vocabulary(captions, vocab_threshold=5):
    word_counts = {}
    for cap in captions:
        for word in cap.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    idx = 4

    for word, count in word_counts.items():
        if count >= vocab_threshold and word not in vocab:
            vocab[word] = idx
            idx += 1

    return vocab

def preprocess_captions(raw_captions_dict, vocab, max_len):
    """
    raw_captions_dict: dict {image_id: [caption1, caption2, ...]}
    """
    processed_targets = []
    processed_inputs = []
    image_ids = []

    for img_id, caps in raw_captions_dict.items():
        for cap in caps:
            tokens = [vocab.get(word, vocab["<unk>"]) for word in cap.split()]

            target_seq = tokens
            input_seq = tokens[:-1]

            processed_targets.append(target_seq)
            processed_inputs.append(input_seq)
            image_ids.append(img_id)

    targets_padded = pad_sequences(processed_targets, maxlen=max_len, padding='post', value=vocab["<pad>"])
    inputs_padded = pad_sequences(processed_inputs, maxlen=max_len-1, padding='post', value=vocab["<pad>"])

    return np.array(image_ids), inputs_padded, targets_padded
