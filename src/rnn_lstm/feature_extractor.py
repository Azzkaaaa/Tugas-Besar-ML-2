import os
import numpy as np
from tensorflow.keras.applications import InceptionV3
from src.common.image_utils import load_images


def extract_and_save_features(image_dir, output_dir, batch_size=32):
    """
    Mengekstraksi fitur dari dataset gambar menggunakan InceptionV3 dan menyimpannya ke .npy
    """
    # include_top=False membuang layer klasifikasi akhir
    # pooling='avg' biar outputnya vektor 1D (2048 untuk InceptionV3)
    print("Loading InceptionV3")
    encoder_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    encoder_model.trainable = False

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    print(f"Jumlah gambar: {total_images}")

    for i in range(0, total_images, batch_size):
        batch_files = image_files[i : i + batch_size]
        batch_paths = [os.path.join(image_dir, f) for f in batch_files]

        batch_images = load_images(batch_paths, target_size=(299, 299))

        # inceptionv3 input range: [-1, 1]
        features = encoder_model.predict(batch_images, verbose=0)

        for j, file_name in enumerate(batch_files):
            base_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(output_dir, f"{base_name}.npy")

            np.save(save_path, features[j])

        print(f"Processed {min(i + batch_size, total_images)} / {total_images} images")

    print("Ekstraksi fitur selesai")

# if __name__ == "__main__":
#     extract_and_save_features("data/Images", "results/rnn_lstm", batch_size=64)