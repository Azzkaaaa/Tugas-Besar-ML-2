from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np
from PIL import Image
def load_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (150, 150),
    normalize_range: str = "0_1",
) -> np.ndarray:
    """
    Load satu gambar dari file path menggunakan PIL.Image.open,
    resize ke dimensi target, konversi ke numpy array, dan normalisasi
    pixel values.

    Parameters
    image_path : str | Path
        Path gambar.
    target_size : tuple[int, int]
        Ukuran target (height, width).
    normalize_range : str
        "0_1"  → normalisasi ke [0, 1]
        "-1_1" → normalisasi ke [-1, 1] 

    Returns
    np.ndarray
        Array gambar dengan shape (H, W, 3), dtype float32.
    """
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))

    arr = np.asarray(img, dtype=np.float32) / 255.0

    if normalize_range == "-1_1":
        arr = arr * 2.0 - 1.0

    return arr


def load_images(
    image_paths: List[Union[str, Path]],
    target_size: Tuple[int, int] = (150, 150),
    normalize_range: str = "0_1",
) -> np.ndarray:
    """
    Load dan memproses sekumpulan gambar dari list file path menjadi
    numpy array dengan shape (N, H, W, C).

    Parameters
    image_paths : list[str | Path]
        List path gambar.
    target_size : tuple[int, int]
        Ukuran target (height, width).
    normalize_range : str
        "0_1" atau "-1_1", diteruskan ke load_image().

    Returns
    np.ndarray
        Batch gambar dengan shape (N, H, W, 3), dtype float32.
    """
    images = [
        load_image(p, target_size=target_size, normalize_range=normalize_range)
        for p in image_paths
    ]
    return np.stack(images, axis=0)


def extract_features(
    image_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    model_name: str = "InceptionV3",
    batch_size: int = 32,
    force: bool = False,
) -> dict:
    """
    Menerima list path gambar, menggunakan Keras CNN encoder (frozen)
    untuk mengekstraksi feature vectors, dan menyimpan hasilnya ke disk
    (format .npy) agar tidak perlu diekstraksi ulang.

    Setiap gambar disimpan sebagai file .npy terpisah di output_path,
    dengan nama file = nama gambar asli (tanpa ekstensi) + ".npy".

    Jika file .npy sudah ada untuk suatu gambar, gambar tersebut akan
    di-skip (kecuali force=True).

    Parameters
    ----------
    image_paths : list[str | Path]
        List path gambar yang akan diekstraksi fiturnya.
    output_path : str | Path
        Direktori tempat menyimpan file .npy.
    model_name : str
        Nama pretrained model Keras: "InceptionV3" atau "VGG16".
    batch_size : int
        Jumlah gambar per batch saat forward pass.
    force : bool
        Jika True, ekstraksi ulang meskipun file .npy sudah ada.

    Returns
    -------
    dict
        Mapping {image_filename: path_to_npy} untuk semua gambar yang diproses.
    """
    import tensorflow as tf

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_name == "InceptionV3":
        from tensorflow.keras.applications import InceptionV3
        encoder = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        img_size = (299, 299)
        normalize_range = "-1_1"  # InceptionV3 butuh input [-1, 1]
    elif model_name == "VGG16":
        from tensorflow.keras.applications import VGG16
        encoder = VGG16(weights="imagenet", include_top=False, pooling="avg")
        img_size = (224, 224)
        normalize_range = "0_1"  # VGG16 pakai [0, 1] lalu mean subtraction
    else:
        raise ValueError(
            f"Model '{model_name}' tidak didukung. Pilih 'InceptionV3' atau 'VGG16'."
        )
    encoder.trainable = False

    paths_to_process = []
    for p in image_paths:
        p = Path(p)
        npy_name = p.stem + ".npy"
        if force or not (output_path / npy_name).exists():
            paths_to_process.append(p)

    total = len(paths_to_process)
    skipped = len(image_paths) - total
    if skipped > 0:
        print(f"Skip {skipped} gambar (sudah ada .npy). Proses {total} gambar.")
    else:
        print(f"Memproses {total} gambar...")

    result_map = {}

    for i in range(0, total, batch_size):
        batch_paths = paths_to_process[i : i + batch_size]
        batch_images = load_images(
            batch_paths,
            target_size=img_size,
            normalize_range=normalize_range,
        )

        features = encoder.predict(batch_images, verbose=0)

        for j, img_path in enumerate(batch_paths):
            npy_name = img_path.stem + ".npy"
            save_path = output_path / npy_name
            np.save(str(save_path), features[j])
            result_map[img_path.name] = str(save_path)

        done = min(i + batch_size, total)
        print(f"  Extracted {done}/{total}")

    for p in image_paths:
        p = Path(p)
        if p.name not in result_map:
            npy_name = p.stem + ".npy"
            result_map[p.name] = str(output_path / npy_name)

    print("Ekstraksi fitur selesai.")
    return result_map


def load_feature(npy_path: Union[str, Path]) -> np.ndarray:
    """
    Load satu feature vector dari file .npy.

    Parameters
    ----------
    npy_path : str | Path
        Path ke file .npy hasil extract_features().

    Returns
    -------
    np.ndarray
        Feature vector 1D.
    """
    return np.load(str(npy_path))


def load_features_from_dir(
    feature_dir: Union[str, Path],
    image_names: Optional[List[str]] = None,
) -> dict:
    """
    Load semua (atau subset) feature vectors dari direktori .npy.

    Parameters
    ----------
    feature_dir : str | Path
        Direktori berisi file-file .npy.
    image_names : list[str] | None
        Jika diberikan, hanya load fitur untuk image names ini
        (tanpa ekstensi, misal ["image1", "image2"]).
        Jika None, load semua file .npy di direktori.

    Returns
    -------
    dict
        Mapping {image_name: np.ndarray} feature vectors.
    """
    feature_dir = Path(feature_dir)
    features = {}

    if image_names is not None:
        for name in image_names:
            npy_path = feature_dir / f"{name}.npy"
            if npy_path.exists():
                features[name] = np.load(str(npy_path))
    else:
        for npy_path in sorted(feature_dir.glob("*.npy")):
            features[npy_path.stem] = np.load(str(npy_path))

    return features


def list_image_paths_by_class(
    root_dir: Union[str, Path],
) -> Tuple[List[str], List[int], List[str]]:
    """
    Ambil semua path gambar dan label berdasarkan folder kelas.
    Cocok untuk dataset seperti Intel Image Classification yang menggunakan
    struktur folder: root_dir/class_name/image.jpg

    Parameters
    ----------
    root_dir : str | Path
        Root directory yang berisi sub-folder per kelas.

    Returns
    -------
    image_paths : list[str]
        List path gambar.
    labels : list[int]
        List label integer sesuai urutan class_names.
    class_names : list[str]
        Nama kelas, diurutkan alfabet.
    """
    root_dir = Path(root_dir)

    class_names = sorted([
        folder.name for folder in root_dir.iterdir()
        if folder.is_dir()
    ])

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths = []
    labels = []

    valid_exts = {".jpg", ".jpeg", ".png"}

    for class_name in class_names:
        class_dir = root_dir / class_name

        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in valid_exts:
                image_paths.append(str(image_path))
                labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names