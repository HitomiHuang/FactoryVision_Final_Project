"""Measure and plot classification results in the team's submission formats."""

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data import CLASS_ORDER


def resolve_device(device):
    """Resolve auto to CUDA if available, otherwise CPU."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_metrics(y_true, y_pred):
    """Return JSON-ready metrics and a true-by-predicted count matrix in CLASS_ORDER."""
    labels = list(range(len(CLASS_ORDER)))
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Expected equal nonempty label and prediction sequences")
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("Class indices must follow CLASS_ORDER")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@contextmanager
def _evaluation_mode(model, device):
    """Restore each module's mode, including frozen batch-normalization layers."""
    modes = [(module, module.training) for module in model.modules()]
    model.to(device).eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@torch.inference_mode()
def evaluate(model, loader, device):
    """Evaluate a loader of (images, labels, filenames) and keep aligned predictions."""
    device = resolve_device(device)
    y_true, y_pred, filenames = [], [], []
    with _evaluation_mode(model, device):
        for images, labels, names in loader:
            logits = model(images.to(device))
            if logits.shape != (len(labels), len(CLASS_ORDER)):
                raise ValueError("Return one logit per course class for each image")
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            filenames.extend(names)
    result = compute_metrics(y_true, y_pred)
    result.update(y_true=y_true, y_pred=y_pred, filenames=filenames)
    return result


def _save_figure(figure, path):
    """Create the output directory and save the figure at the team resolution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def plot_curves(history, path):
    """Save training and validation loss and accuracy across epochs."""
    figure = Figure(figsize=(9, 4), layout="constrained")
    axes = figure.subplots(1, 2)
    lengths = {len(history[key]) for key in ("train_loss", "val_loss", "train_acc", "val_acc")}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("All four history series must span the same nonzero number of epochs")
    epochs = range(1, len(history["train_loss"]) + 1)
    for ax, key, label in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], label="Training")
        ax.plot(epochs, history[f"val_{key}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
    figure.suptitle(f"{Path(path).stem.removesuffix('_curves')}: Training and Validation")
    _save_figure(figure, path)


def plot_confusion_matrix(cm, path):
    """Save the six-class confusion matrix as integer counts."""
    cm = np.asarray(cm)
    if cm.shape != (len(CLASS_ORDER), len(CLASS_ORDER)):
        raise ValueError("Expected a 6 by 6 confusion matrix")
    figure = Figure(figsize=(7.5, 6), layout="constrained")
    ax = figure.subplots()
    heatmap = ax.imshow(cm, cmap="Blues", vmin=0)
    figure.colorbar(heatmap, ax=ax, label="Images")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set(xlabel="Predicted class", ylabel="True class",
           title=f"{Path(path).stem.removesuffix('_cm')}: Confusion Matrix")
    for (row, col), count in np.ndenumerate(cm):
        color = "white" if count > cm.max() / 2 else "black"
        ax.text(col, row, str(int(count)), ha="center", va="center", color=color)
    _save_figure(figure, path)


def model_stats(model, ckpt_path):
    """Count all and trainable parameters and measure the actual checkpoint in MiB."""
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_mb": Path(ckpt_path).stat().st_size / (1024**2),
    }


@torch.inference_mode()
def measure_latency(model, input_size, device, runs):
    """Return batch-one forward latency in ms after 10 warm-ups, including CUDA sync."""
    if runs < 1:
        raise ValueError("runs must be positive; use 100 for the report")
    device = resolve_device(device)
    if isinstance(input_size, int):
        conv = next((m for m in model.modules() if isinstance(m, torch.nn.Conv2d)), None)
        if conv is None:
            raise ValueError("Supply (channels, height, width) for a model without Conv2d")
        shape = (conv.in_channels, input_size, input_size)
    else:
        shape = tuple(input_size)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a square size or (channels, height, width)")
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    batch = torch.zeros((1, *shape), device=device, dtype=dtype)
    with _evaluation_mode(model, device):
        for _ in range(10):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        for _ in range(runs):
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return (perf_counter() - start) * 1000 / runs
