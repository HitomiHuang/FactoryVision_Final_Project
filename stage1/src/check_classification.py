"""Check a Stage's loaders and metric wiring with a known-answer classifier."""

import argparse

import torch

from data import CLASS_ORDER, get_dataloaders
from metrics import evaluate
from seed import set_seed


class ConstantClassifier(torch.nn.Module):
    """Predict class zero for every image solely for a known-answer check."""

    def forward(self, images):
        """Return six equal logits, for which argmax picks class zero."""
        return images.new_zeros(len(images), len(CLASS_ORDER))


def parse_args():
    """Parse independent smoke-check settings without model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    """Verify input dimensions and compare every confusion-matrix cell to the dataset."""
    args = parse_args()
    set_seed(42)
    train, val = get_dataloaders(
        args.data_root, args.input_size, [0.0] * args.channels, [1.0] * args.channels,
        32, args.num_workers, channels=args.channels, seed=42,
    )
    images, _, _ = next(iter(train))
    if tuple(images.shape[1:]) != (args.channels, args.input_size, args.input_size):
        raise AssertionError(f"Unexpected input shape: {tuple(images.shape)}")
    result = evaluate(ConstantClassifier(), val, "cpu")
    expected = [[0] * len(CLASS_ORDER) for _ in CLASS_ORDER]
    for _, label in val.dataset.samples:
        expected[label][0] += 1
    paths = [f"{CLASS_ORDER[label]}/{p.name}" for p, label in val.dataset.samples]
    if result["confusion_matrix"] != expected or result["filenames"] != paths:
        raise AssertionError("Counts or validation filenames differ from the source split")
    correct = expected[0][0]
    accuracy = result["metrics"]["accuracy"]
    if accuracy != correct / len(paths):
        raise AssertionError("Accuracy differs from the known-class baseline")
    print(f"[loader check] train={len(train.dataset)} validation={len(paths)} "
          f"batch={tuple(images.shape)} constant_accuracy={correct}/{len(paths)} "
          f"({accuracy:.6f}) cm_total={sum(map(sum, expected))}")


if __name__ == "__main__":
    main()
