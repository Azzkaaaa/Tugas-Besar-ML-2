from pathlib import Path
from PIL import Image
import numpy as np


def load_image(image_path, target_size=(150, 150)):
    """
    Load 1 gambar dari path, resize, ubah ke RGB, normalisasi ke [0, 1].
    Parameters
    image_path : str or Path
        Path gambar.
    target_size : tuple
        Ukuran target dalam format (height, width).

    Returns
    np.ndarray
        Array gambar dengan shape (H, W, 3), dtype float32.
    """
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_images(image_paths, target_size=(150, 150)):
    """
    Load banyak gambar menjadi batch NumPy.

    Parameters
    image_paths : list[str] or list[Path]
        List path gambar.
    target_size : tuple
        Ukuran target dalam format (height, width).

    Returns
    np.ndarray
        Batch gambar dengan shape (N, H, W, 3).
    """
    images = [load_image(path, target_size) for path in image_paths]
    return np.stack(images, axis=0)


def list_image_paths_by_class(root_dir):
    """
    Ambil semua path gambar dan label berdasarkan folder kelas.
    Returns
    image_paths : list[str]
    labels : list[int]
    class_names : list[str]
    """
    root_dir = Path(root_dir)

    class_names = sorted([
        folder.name for folder in root_dir.iterdir()
        if folder.is_dir()
    ])

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

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