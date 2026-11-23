"""Load the course classification splits with the team's fixed class indices."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_ORDER = [
    "crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale",
    "scratches",
]


def classification_samples(data_root, split):
    """Return (PNG path, class index) pairs in stable class and filename order."""
    folder = Path(data_root) / "classification" / split
    names = sorted(p.name for p in folder.iterdir() if p.is_dir())
    if names != CLASS_ORDER:
        raise ValueError(f"{folder}: expected {CLASS_ORDER}, found {names}")
    samples = []
    for label, name in enumerate(CLASS_ORDER):
        paths = sorted((folder / name).glob("*.png"))
        if not paths:
            raise ValueError(f"No PNG images in {folder / name}")
        samples.extend((path, label) for path in paths)
    return samples


def _read_grayscale(path, input_size):
    """Read a square bilinear resize as 8-bit grayscale pixels."""
    with Image.open(path) as image:
        size = (input_size, input_size)
        return np.asarray(image.convert("L").resize(size, Image.Resampling.BILINEAR))


def compute_train_normalization(data_root, input_size, channels=1):
    """Calculate population mean and std on resized train pixels in [0, 1]."""
    if channels not in (1, 3) or input_size < 1:
        raise ValueError("Use 1 or 3 channels and a positive input_size")
    total = squares = 0.0
    count = 0
    for path, _ in classification_samples(data_root, "train"):
        pixels = _read_grayscale(path, input_size).astype(np.float64) / 255.0
        total += pixels.sum()
        squares += np.square(pixels).sum()
        count += pixels.size
    mean = float(total / count)
    std = float(np.sqrt(max(0.0, squares / count - mean**2)))
    return [mean] * channels, [std] * channels


class ClassificationDataset(Dataset):
    """Return (image tensor, class index, class/filename) for one official split."""

    def __init__(self, data_root, split, channels, transform):
        """Keep sample paths only and decode each image on demand."""
        self.samples = classification_samples(data_root, split)
        self.mode = "L" if channels == 1 else "RGB"
        self.transform = transform

    def __len__(self):
        """Return the number of images in this split."""
        return len(self.samples)

    def __getitem__(self, index):
        """Convert the source grayscale image before applying the split transform."""
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert(self.mode))
        return tensor, label, f"{CLASS_ORDER[label]}/{path.name}"


def _seed_worker(worker_id):
    """Seed NumPy and Python from the worker's PyTorch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloaders(data_root, input_size, mean, std, batch_size, num_workers,
                    channels=3, train_transform=None, seed=42):
    """Return train and validation loaders; a full custom transform affects train only."""
    if channels not in (1, 3) or input_size < 1:
        raise ValueError("Use 1 or 3 channels and a positive input_size")
    if len(mean) != channels or len(std) != channels or min(std) <= 0:
        raise ValueError("Supply one mean and one positive std per channel")
    basic = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = ClassificationDataset(
        data_root, "train", channels, basic if train_transform is None else train_transform
    )
    validation = ClassificationDataset(data_root, "validation", channels, basic)
    options = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=_seed_worker)
    return (
        DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(seed), **options),
        DataLoader(validation, shuffle=False,
                   generator=torch.Generator().manual_seed(seed), **options),
    )


def load_numpy(data_root, input_size):
    """Return X_train, y_train, X_val, y_val as flat float32 pixels and int labels."""
    if input_size < 1:
        raise ValueError("input_size must be positive")

    def load_split(split):
        """Read one split without fitting or sharing any preprocessing statistics."""
        samples = classification_samples(data_root, split)
        images = np.empty((len(samples), input_size**2), dtype=np.float32)
        labels = np.empty(len(samples), dtype=np.int64)
        for i, (path, label) in enumerate(samples):
            images[i] = _read_grayscale(path, input_size).reshape(-1)
            labels[i] = label
        images /= 255.0
        return images, labels

    return (*load_split("train"), *load_split("validation"))


def parse_args():
    """Parse options for the Stage 1 train-only normalization command."""
    parser = argparse.ArgumentParser(description=compute_train_normalization.__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--input_size", type=int, required=True)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=1)
    return parser.parse_args()


def main():
    """Print the train-only normalization values for a chosen Stage 1 input size."""
    args = parse_args()
    mean, std = compute_train_normalization(args.data_root, args.input_size, args.channels)
    print(json.dumps({"input_size": args.input_size, "normalization_mean": mean,
                      "normalization_std": std}))


if __name__ == "__main__":
    main()
