"""
Dataset preparation, splitting, exploration, and augmentation pipeline for TrashNet.
"""

import os
import shutil
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def ensure_dataset_exists(data_dir=config.DATA_DIR):
    """
    Verifies that the TrashNet dataset is available in data_dir.
    If missing, downloads the dataset archive from verified public sources
    and extracts it into the expected directory structure.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if all class folders exist and contain images
    existing_classes = [c for c in config.CLASSES if (data_dir / c).is_dir()]
    if len(existing_classes) == len(config.CLASSES):
        counts = {c: len(list((data_dir / c).glob("*.*"))) for c in config.CLASSES}
        total = sum(counts.values())
        if total >= 2000:
            print(f"TrashNet dataset already present with {total} images across 6 classes.")
            return data_dir

    # Check if data/dataset-resized exists
    alt_dir = data_dir.parent / "dataset-resized"
    if alt_dir.is_dir():
        alt_classes = [c for c in config.CLASSES if (alt_dir / c).is_dir()]
        if len(alt_classes) == len(config.CLASSES):
            print(f"Found dataset in {alt_dir}. Consolidating to {data_dir}...")
            for c in config.CLASSES:
                target_c = data_dir / c
                target_c.mkdir(parents=True, exist_ok=True)
                for img_file in (alt_dir / c).glob("*.*"):
                    shutil.copy2(img_file, target_c / img_file.name)
            return data_dir

    print("TrashNet dataset not found. Downloading dataset archive...")
    zip_path = data_dir.parent / "trashnet.zip"

    # Multiple redundant mirror sources for TrashNet dataset-resized.zip
    urls = [
        "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
        "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
        "https://storage.googleapis.com/smart-waste-public/dataset-resized.zip",
    ]

    downloaded = False
    for url in urls:
        try:
            print(f"Attempting download from: {url}")
            # Use request with headers
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
                print(f"Downloaded {zip_path.stat().st_size / (1024*1024):.1f} MB successfully.")
                downloaded = True
                break
        except Exception as e:
            print(f"Download failed from {url}: {e}")

    if not downloaded:
        # Check if git lfs or direct clone can provide it
        raise RuntimeError(
            "Could not automatically download TrashNet dataset. "
            "Please place the 6 category folders (glass, paper, cardboard, plastic, metal, trash) "
            f"directly inside: {data_dir}"
        )

    print(f"Extracting {zip_path}...")
    extract_to = data_dir.parent / "extracted_temp"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Find the folder containing the category subdirectories
    source_folder = None
    for root, dirs, _ in os.walk(extract_to):
        if set(config.CLASSES).issubset(set(dirs)):
            source_folder = Path(root)
            break

    if source_folder is None:
        # Search for case-insensitive matches
        for root, dirs, _ in os.walk(extract_to):
            dirs_lower = [d.lower() for d in dirs]
            if set(config.CLASSES).issubset(set(dirs_lower)):
                source_folder = Path(root)
                break

    if source_folder:
        for c in config.CLASSES:
            src = source_folder / c
            if not src.exists():
                # case insensitive lookup
                for d in source_folder.iterdir():
                    if d.name.lower() == c.lower():
                        src = d
                        break
            dst = data_dir / c
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    shutil.copy2(f, dst / f.name)

    # Clean up temp
    if extract_to.exists():
        shutil.rmtree(extract_to, ignore_errors=True)
    if zip_path.exists():
        zip_path.unlink()

    return data_dir


def explore_dataset(data_dir=config.DATA_DIR):
    """
    Inspects and logs the number of images per category.
    Returns a pandas DataFrame of counts and proportions.
    """
    data_dir = Path(data_dir)
    data = []
    for cls in config.CLASSES:
        cls_dir = data_dir / cls
        count = len(list(cls_dir.glob("*.*"))) if cls_dir.is_dir() else 0
        data.append({"Category": cls, "Count": count})

    df = pd.DataFrame(data)
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total * 100).round(2) if total > 0 else 0
    print("\n===== Dataset Category Distribution =====")
    print(df.to_string(index=False))
    print(f"Total Images: {total}\n")
    return df


def create_train_val_test_split(
    data_dir=config.DATA_DIR,
    train_dir=config.TRAIN_DIR,
    val_dir=config.VAL_DIR,
    test_dir=config.TEST_DIR,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    random_state=config.RANDOM_STATE,
    force_rebuild=False,
):
    """
    Performs stratified 70:15:15 split into separate directories.
    Prevents data leakage by ensuring train, val, and test splits are strictly separated.
    """
    data_dir = Path(data_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)

    # Check if split already exists
    if not force_rebuild:
        train_count = sum(len(list((train_dir / c).glob("*.*"))) for c in config.CLASSES if (train_dir / c).exists())
        val_count = sum(len(list((val_dir / c).glob("*.*"))) for c in config.CLASSES if (val_dir / c).exists())
        test_count = sum(len(list((test_dir / c).glob("*.*"))) for c in config.CLASSES if (test_dir / c).exists())

        if train_count > 0 and val_count > 0 and test_count > 0:
            print(f"Existing split detected (Train: {train_count}, Val: {val_count}, Test: {test_count}). Using existing split.")
            return train_dir, val_dir, test_dir

    print("\nCreating stratified Train (70%), Validation (15%), Test (15%) split...")

    # Clear and recreate destination directories
    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for c in config.CLASSES:
            (split_dir / c).mkdir(parents=True, exist_ok=True)

    all_files = []
    all_labels = []

    for cls in config.CLASSES:
        cls_dir = data_dir / cls
        files = [f for f in cls_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        for f in files:
            all_files.append(f)
            all_labels.append(cls)

    if not all_files:
        raise ValueError(f"No image files found in {data_dir}. Please verify dataset loading.")

    # First split: Train vs Temp (Val + Test)
    # 70% Train, 30% Temp
    val_test_ratio = val_ratio + test_ratio
    train_files, temp_files, train_y, temp_y = train_test_split(
        all_files,
        all_labels,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=random_state,
    )

    # Second split: Val vs Test from Temp
    # 15% / 30% = 0.5 of temp for validation, 0.5 for testing
    relative_test_ratio = test_ratio / val_test_ratio
    val_files, test_files, val_y, test_y = train_test_split(
        temp_files,
        temp_y,
        test_size=relative_test_ratio,
        stratify=temp_y,
        random_state=random_state,
    )

    # Copy files to respective split directories
    for f, label in zip(train_files, train_y):
        shutil.copy2(f, train_dir / label / f.name)
    for f, label in zip(val_files, val_y):
        shutil.copy2(f, val_dir / label / f.name)
    for f, label in zip(test_files, test_y):
        shutil.copy2(f, test_dir / label / f.name)

    print(f"Split complete: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return train_dir, val_dir, test_dir


def compute_balanced_class_weights(train_dir=config.TRAIN_DIR):
    """
    Computes balanced class weights for handling class imbalance (e.g. fewer 'trash' samples).
    """
    train_dir = Path(train_dir)
    labels = []
    for idx, cls in enumerate(config.CLASSES):
        cls_dir = train_dir / cls
        count = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
        labels.extend([idx] * count)

    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=np.array(labels),
    )
    class_weight_dict = dict(zip(unique_classes, weights))
    print(f"Calculated Class Weights: {class_weight_dict}")
    return class_weight_dict


def get_data_generators(
    train_dir=config.TRAIN_DIR,
    val_dir=config.VAL_DIR,
    test_dir=config.TEST_DIR,
    img_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
):
    """
    Creates ImageDataGenerators for model training, validation, and testing.

    Training generator applies data augmentation:
    - Random rotation (20 degrees)
    - Random zoom (0.2)
    - Horizontal flipping
    - Width and height shifts (0.1)
    - Shear range (0.1)

    Validation and testing generators use ONLY resizing with NO data augmentation,
    strictly preventing test data leakage.
    Note: Both raw and preprocessed models handle their required pixel normalization:
    - Custom CNN includes tf.keras.layers.Rescaling(1./255)
    - MobileNetV2 includes mobilenet_v2.preprocess_input
    Hence, raw pixel values [0, 255] are passed safely to both architectures.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validation and test generators: deterministic, no augmentation
    eval_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        classes=config.CLASSES,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_STATE,
    )

    val_generator = eval_datagen.flow_from_directory(
        str(val_dir),
        target_size=img_size,
        batch_size=batch_size,
        classes=config.CLASSES,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        classes=config.CLASSES,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator

