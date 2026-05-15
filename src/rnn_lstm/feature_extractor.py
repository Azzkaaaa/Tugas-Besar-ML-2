"""
Feature Extractor untuk RNN/LSTM Image Captioning.

Wrapper tipis di atas src.common.image_utils.extract_features()
dengan default yang sesuai untuk pipeline Flickr8k captioning.
"""

import os
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
from src.common.image_utils import extract_features, load_features_from_dir


def extract_and_save_features(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "InceptionV3",
    batch_size: int = 32,
    force: bool = False,
) -> dict:
    """
    Mengekstraksi fitur dari semua gambar di suatu direktori menggunakan
    pretrained CNN encoder (frozen) dan menyimpannya ke .npy.

    Parameters
    ----------
    image_dir : str | Path
        Direktori berisi gambar-gambar (.jpg, .jpeg, .png).
    output_dir : str | Path
        Direktori tempat menyimpan file .npy.
    model_name : str
        "InceptionV3" atau "VGG16".
    batch_size : int
        Jumlah gambar per batch.
    force : bool
        Jika True, ekstraksi ulang meskipun .npy sudah ada.

    Returns
    -------
    dict
        Mapping {image_filename: path_to_npy}.
    """
    image_dir = Path(image_dir)

    valid_exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        str(p) for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])

    print(f"Ditemukan {len(image_paths)} gambar di {image_dir}")

    return extract_features(
        image_paths=image_paths,
        output_path=output_dir,
        model_name=model_name,
        batch_size=batch_size,
        force=force,
    )