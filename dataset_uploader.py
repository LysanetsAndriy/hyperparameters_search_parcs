import numpy as np
import pickle
import os
import json
from google.cloud import storage
import sys
from torchvision.datasets import CIFAR10

def download_cifar10():
    """
    Завантажити оригінальний CIFAR-10 датасет

    CIFAR-10 структура:
    - 10 класів: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Зображення: 32×32 RGB
    - Train: 50,000 зображень (5,000 на клас)
    - Test: 10,000 зображень (1,000 на клас)

    Returns:
        X_train: np.array shape (50000, 32, 32, 3) - зображення
        y_train: np.array shape (50000,) - мітки 0-9
        X_test: np.array shape (10000, 32, 32, 3)
        y_test: np.array shape (10000,)
    """


    print("=" * 60)
    print("STEP 1: Downloading CIFAR-10 dataset")
    print("=" * 60)

    # Завантажити через torchvision (автоматично кешується)
    print("Downloading training set...")
    train_dataset = CIFAR10(root='./data', train=True, download=True)

    print("Downloading test set...")
    test_dataset = CIFAR10(root='./data', train=False, download=True)

    # Конвертувати PIL Images в numpy arrays
    print("Converting to numpy arrays...")

    # Train set
    X_train = np.array([np.array(img) for img, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset])

    # Test set
    X_test = np.array([np.array(img) for img, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset])

    print(f"✓ Train shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")
    print(f"✓ Image dtype: {X_train.dtype} (values 0-255)")
    print(f"✓ Label dtype: {y_train.dtype}")

    return X_train, y_train, X_test, y_test


def create_validation_split(X_train, y_train, val_ratio=0.2, seed=42):
    print("\n" + "=" * 60)
    print("STEP 2: Creating Train/Validation split")
    print("=" * 60)

    # Встановити random seed для reproducibility
    np.random.seed(seed)

    n_samples = len(X_train)
    n_val = int(n_samples * val_ratio)
    n_train_new = n_samples - n_val

    print(f"Original train size: {n_samples}")
    print(f"Validation ratio: {val_ratio} ({val_ratio * 100}%)")
    print(f"New train size: {n_train_new}")
    print(f"Validation size: {n_val}")

    # Стратифікована вибірка (зберігає розподіл класів)
    # Для кожного класу візьмемо 20% в validation

    train_indices = []
    val_indices = []

    for class_id in range(10):
        # Знайти всі індекси цього класу
        class_indices = np.where(y_train == class_id)[0]

        # Перемішати
        np.random.shuffle(class_indices)

        # Розділити
        n_val_class = int(len(class_indices) * val_ratio)

        val_indices.extend(class_indices[:n_val_class])
        train_indices.extend(class_indices[n_val_class:])

        print(f"  Class {class_id}: {len(class_indices)} total → "
              f"{len(class_indices) - n_val_class} train, {n_val_class} val")

    # Конвертувати в numpy arrays і перемішати
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # Розділити дані
    X_train_new = X_train[train_indices]
    y_train_new = y_train[train_indices]

    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(f"\n✓ Final train shape: {X_train_new.shape}")
    print(f"✓ Final validation shape: {X_val.shape}")

    print("\nClass distribution check:")
    for class_id in range(10):
        train_count = np.sum(y_train_new == class_id)
        val_count = np.sum(y_val == class_id)
        print(f"  Class {class_id}: train={train_count}, val={val_count}")

    return X_train_new, y_train_new, X_val, y_val


def save_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Зберегти датасет у compressed numpy format

    Формат .npz:
    - Compressed numpy archive
    - Швидке завантаження
    - Малий розмір (~150 MB для CIFAR-10)

    Зберігаємо як uint8 (0-255) для економії місця
    Worker буде конвертувати в float32 та нормалізувати під час завантаження

    Args:
        X_train, y_train: Train set (40000 зображень)
        X_val, y_val: Validation set (10000 зображень)
        X_test, y_test: Test set (10000 зображень)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Saving dataset to disk")
    print("=" * 60)

    # ImageNet statistics для MobileNetV2 pretrained weights
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print("Normalization statistics (ImageNet):")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")

    # Створити директорію
    os.makedirs('data', exist_ok=True)

    print("\nSaving to data/cifar10.npz...")

    # Зберегти compressed
    np.savez_compressed(
        'data/cifar10.npz',
        # Зображення (uint8 для економії: 1 byte per pixel)
        X_train=X_train.astype(np.uint8),
        X_val=X_val.astype(np.uint8),
        X_test=X_test.astype(np.uint8),
        # Мітки (int64)
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        # Статистика для нормалізації
        mean=mean,
        std=std
    )

    # Розмір файлу
    size_mb = os.path.getsize('data/cifar10.npz') / 1024 / 1024
    print(f"✓ Saved! File size: {size_mb:.1f} MB")

    # Перевірити що можна завантажити
    print("\nVerifying saved file...")
    data = np.load('data/cifar10.npz')
    print(f"✓ Keys in archive: {list(data.keys())}")
    print(f"✓ X_train shape: {data['X_train'].shape}")
    print(f"✓ X_val shape: {data['X_val'].shape}")
    print(f"✓ X_test shape: {data['X_test'].shape}")

    # Створити metadata файл
    print("\nCreating metadata.json...")
    metadata = {
        "name": "cifar10",
        "description": "CIFAR-10 dataset prepared for PARCS Grid Search",
        "num_classes": 10,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        "image_shape": [32, 32, 3],
        "splits": {
            "train": {
                "num_samples": len(X_train),
                "percentage": 80.0
            },
            "validation": {
                "num_samples": len(X_val),
                "percentage": 20.0
            },
            "test": {
                "num_samples": len(X_test),
                "percentage": "original"
            }
        },
        "format": "uint8",
        "value_range": [0, 255],
        "normalization": {
            "method": "imagenet",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "formula": "(pixel / 255.0 - mean) / std"
        },
        "random_seed": 42
    }

    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✓ Metadata saved to data/metadata.json")


# ============================================================
# SECTION 4: Завантаження на Google Cloud Storage
# ============================================================

def upload_to_gcs(bucket_name):
    """
    Завантажити датасет на Google Cloud Storage

    Структура на GCS:
    gs://bucket-name/
      └─ datasets/
          └─ cifar10/
              ├─ data.npz        (датасет)
              └─ metadata.json   (інформація)

    Args:
        bucket_name: str - назва GCS bucket
    """
    print("\n" + "=" * 60)
    print("STEP 4: Uploading to Google Cloud Storage")
    print("=" * 60)

    gcs_path = f"gs://{bucket_name}/datasets/cifar10/"
    print(f"Target path: {gcs_path}")

    # Створити GCS client
    print("\nInitializing GCS client...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload data.npz
    print("\nUploading data.npz...")
    blob_data = bucket.blob('datasets/cifar10/data.npz')
    blob_data.upload_from_filename(
        'data/cifar10.npz',
        content_type='application/octet-stream'
    )
    print(f"✓ Uploaded data.npz ({os.path.getsize('data/cifar10.npz') / 1024 / 1024:.1f} MB)")

    # Upload metadata.json
    print("\nUploading metadata.json...")
    blob_meta = bucket.blob('datasets/cifar10/metadata.json')
    blob_meta.upload_from_filename(
        'data/metadata.json',
        content_type='application/json'
    )
    print(f"✓ Uploaded metadata.json")

    # Зробити публічними (опціонально)
    # Це дозволяє workers завантажувати без authentication
    print("\nMaking files public...")
    blob_data.make_public()
    blob_meta.make_public()
    print("✓ Files are now publicly accessible")

    # Вивести URLs
    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nDataset URL:")
    print(f"  {gcs_path}data.npz")
    print(f"\nPublic URLs:")
    print(f"  Data: {blob_data.public_url}")
    print(f"  Metadata: {blob_meta.public_url}")
    print(f"\nUse in your code:")
    print(f"  gcs_path = '{gcs_path}data.npz'")


def print_dataset_statistics(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Вивести детальну статистику датасету
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print("\n1. Dataset Sizes:")
    print(f"   Train:      {len(X_train):>6,} images ({len(X_train) / 600:.1f}% of total)")
    print(f"   Validation: {len(X_val):>6,} images ({len(X_val) / 600:.1f}% of total)")
    print(f"   Test:       {len(X_test):>6,} images ({len(X_test) / 600:.1f}% of total)")
    print(f"   Total:      {len(X_train) + len(X_val) + len(X_test):>6,} images")

    print("\n2. Image Properties:")
    print(f"   Shape: {X_train.shape[1:]}")
    print(f"   Dtype: {X_train.dtype}")
    print(f"   Value range: [{X_train.min()}, {X_train.max()}]")

    print("\n3. Memory Usage:")
    train_mb = X_train.nbytes / 1024 / 1024
    val_mb = X_val.nbytes / 1024 / 1024
    test_mb = X_test.nbytes / 1024 / 1024
    print(f"   Train:      {train_mb:>6.1f} MB")
    print(f"   Validation: {val_mb:>6.1f} MB")
    print(f"   Test:       {test_mb:>6.1f} MB")
    print(f"   Total:      {train_mb + val_mb + test_mb:>6.1f} MB")

    print("\n4. Class Distribution:")
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    print(f"   {'Class':<12} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"   {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 7}")

    for i, name in enumerate(class_names):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        test_count = np.sum(y_test == i)
        print(f"   {name:<12} {train_count:>7} {val_count:>7} {test_count:>7}")

    print("\n5. Pixel Value Statistics:")
    print(f"   Train mean: {X_train.mean():.2f}")
    print(f"   Train std:  {X_train.std():.2f}")
    print(f"   (These are raw uint8 values, will be normalized to ImageNet stats)")


if __name__ == '__main__':
    # Перевірити аргументи
    if len(sys.argv) < 2:
        print("=" * 60)
        print("CIFAR-10 Dataset Uploader for PARCS Grid Search")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 dataset_uploader.py YOUR_BUCKET_NAME")
        print("\nExample:")
        print("  python3 dataset_uploader.py my-parcs-datasets")
        print("\nSteps:")
        print("  1. Download CIFAR-10")
        print("  2. Create Train/Val/Test split")
        print("  3. Save to compressed .npz format")
        print("  4. Upload to Google Cloud Storage")
        print("\n")
        sys.exit(1)

    bucket_name = sys.argv[1]

    print("=" * 60)
    print("CIFAR-10 DATASET PREPARATION")
    print("=" * 60)
    print(f"Target bucket: gs://{bucket_name}/datasets/cifar10/")
    print("")

    # STEP 1: Download
    X_train_full, y_train_full, X_test, y_test = download_cifar10()

    # STEP 2: Create validation split
    X_train, y_train, X_val, y_val = create_validation_split(
        X_train_full, y_train_full,
        val_ratio=0.2,  # 20% для validation
        seed=42
    )

    # Print statistics
    print_dataset_statistics(X_train, y_train, X_val, y_val, X_test, y_test)

    # STEP 3: Save locally
    save_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    # STEP 4: Upload to GCS
    upload_to_gcs(bucket_name)

    print("\n" + "=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
