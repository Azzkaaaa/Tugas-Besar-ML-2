# Tugas-Besar-ML-2
## Overview
Repository ini dibuat untuk memenuhi Tugas Besar 2 Pembelajaran Mesin 2025/2026
dengan mengimplementasikan modul forward propagation CNN, Simple RNN, dan LSTM from scratch.
Dilakukan juga pipeline image captioning melalui arsitektur encoder-decoder yang menggabungkan CNN dan LSTM, yang merupakan fondasi dari banyak sistem deep learning modern seperti LLM multimodal dan speech recognition.

### Convolutional Neural Network
Bagian CNN dikerjakan dalam konteks task image classification menggunakan dataset Intel Image Classification (~25.000 gambar, 6 kategori: buildings, forest, glacier, mountain, sea, street), dengan split train, validation, dan test yang sudah tersedia.
#### Arsitektur Model dan Implementasi
Model CNN memiliki jenis layer berikut:
- Conv2D layer Implementasi shared parameter dan non-shared parameter
- Pooling layers
- Flatten/Global Pooling layer
- Dense layer
Loss function: Sparse Categorical Crossentropy. Optimizer: Adam.

### Simple Recurrent Neural Network dan Long Short-Term Memory Network
Bagian RNN dan LSTM dikerjakan dalam konteks task image captioning menggunakan dataset Flickr8k (8.092 gambar, 5 caption per gambar, split: 6.000 train / 1.000 validation / 1.000 test). Kedua arsitektur (RNN dan LSTM) dilatih secara terpisah sebagai decoder dan dibandingkan hasilnya.

#### Arsitektur Model dan Implementasi
Arsitektur mengacu pada Show and Tell (Vinyals et al., 2015), menggunakan injection method pre-inject: feature vector CNN di-project melalui Dense layer sebagai input x pada timestep t=-1, sebelum token <start>. Hidden state diinisialisasi dengan zeros (h0) Untuk LSTM, c₀ diinisialisasi zeros. Snippet dari paper (hal. 4):
- Input layer untuk caption sequence yang sudah di-prepend dengan feature vector CNN (shape: (seq_len+1, embed_dim))
- Untuk caption words: Embedding layer: token → dense vector (embed_dim)
- Feature vector CNN di-project via Dense layer ke embed_dim, lalu di-concatenate di depan sequence embedding sebagai x₋₁
- SimpleRNN atau LSTM dengan h₀ = zeros (default Keras)
- Dense output layer: → vocab_size, aktivasi softmax

## Structure
```
./
├── doc/
├── data/
│  ├── raw/
│  │  ├── flickr8k/
│  │  │  └── *.jpg
│  │  └── intel_image_classification/
│  │     ├── seg_pred/
│  │     ├── seg_test/
│  │     └── seg_main/
│  └── captions.txt
├── models/
│  ├── cnn/
│  └── rnn_lstm/
├── results/
│  ├── cnn/
│  └── rnn_lstm/
└── src/
   ├── cnn/
   │  ├── config.py
   │  ├── keras_layers.py
   │  ├── keras_models.py
   │  ├── scratch_layers_cnn.py
   │  ├── scratch_models.py
   │  ├── training.py
   │  ├── visualization.py
   │  └── weight_loader_cnn.py
   ├── common/
   │  ├── activations.py
   │  ├── dense.py
   │  └── image_utils.py
   ├── notebook/
   │  ├── cnn_experiments.ipynb
   │  └── rnn_ltsm_experiments.ipynb
   └── rnn_lstm/
      ├── caption_preprocess.py
      ├── inference_utils.py
      ├── scratch_decoder.py
      └── scratch_layers_rnn_lstm.py
``` 

## How to Run
### 1. Create Environment
```bash
    python -m venv .venv
```

### 2. Activate Environment
```bash
    # windows
    venv\Scripts\activate

    # mac/linux
    source .venv/bin/activate
```

### 3. Install Requirements
```bash
    pip install -r requirements.txt
```
### 4. Jalankan Notebook
File notebook ada di `src/notebook`. Pastikan kernel yang digunakan untuk environment notebook adalah kernel Python dari
virtual environment yang telah diaktifkan sebelumnya.

## Pembagian Kerja Anggota Kelompok
Tabel kontribusi anggota dalam pengerjaan:

| Nama Anggota | NIM | Bagian |
| :--- | :---: |:---------------------------|
| Ahmad Syafiq | 13523135 | Training, Notebook |
| Muhammad Aulia Azka | 13523137 | Setup, CNN |
| Fachriza Ahmad Setiyono | 13523162 | RNN, LSTM |
