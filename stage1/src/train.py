"""Train the Stage 1 CNN and select a checkpoint by validation Macro F1."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torchvision import transforms

from data import CLASS_ORDER, compute_train_normalization, get_dataloaders
from metrics import (
    compute_metrics, evaluate, measure_latency, model_stats, plot_confusion_matrix,
    plot_curves, resolve_device,
)
from model import MODEL_NAME, build_model
from seed import set_seed


# Ablation table: exp -> (description, use method A, use method B).
# Method A = data augmentation on train only; method B = Dropout before the classifier.
EXPERIMENTS = {
    "E1": ("Baseline CNN", False, False),
    "E2": ("Baseline + Data Augmentation", True, False),
    "E3": ("Baseline + Dropout", False, True),
    "E4": ("Baseline + Data Augmentation + Dropout", True, True),
}


def parse_args():
    """Parse the experiment id and the training hyperparameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--exp", choices=sorted(EXPERIMENTS), default="E1")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Where to save the best state_dict (default: checkpoints/<exp>.pth).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Used by E3 and E4.")
    parser.add_argument("--flip_p", type=float, default=0.5, help="Used by E2 and E4.")
    parser.add_argument("--rotation", type=float, default=15.0,
                        help="Max rotation in degrees, used by E2 and E4.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    return parser.parse_args()


def build_train_transform(args, mean, std):
    """Return the method A augmentation pipeline (applied to the train split only)."""
    return transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(p=args.flip_p),
        transforms.RandomVerticalFlip(p=args.flip_p),
        transforms.RandomRotation(degrees=args.rotation),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one pass over train and return the mean loss and accuracy."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Return validation loss, accuracy and Macro F1 without updating the model."""
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).tolist())
    scores = compute_metrics(y_true, y_pred)["metrics"]
    return total_loss / len(y_true), scores["accuracy"], scores["macro_f1"]


def main():
    """Train one ablation setting, keep the best epoch and save JSON and plots."""
    args = parse_args()
    description, use_augmentation, use_dropout = EXPERIMENTS[args.exp]
    checkpoint = args.checkpoint or Path("checkpoints") / f"{args.exp}.pth"
    device = resolve_device(args.device)
    set_seed(args.seed)

    # Normalization statistics come from the train split only.
    mean, std = compute_train_normalization(args.data_root, args.input_size, channels=1)
    train_transform = build_train_transform(args, mean, std) if use_augmentation else None
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.input_size, mean, std, args.batch_size, args.num_workers,
        channels=1, train_transform=train_transform, seed=args.seed,
    )

    dropout = args.dropout_p if use_dropout else 0.0
    model = build_model(in_channels=1, num_classes=len(CLASS_ORDER), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1, best_epoch = -1.0, 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)
        print(f"[{args.exp}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f}", flush=True)
        # Keep the epoch with the highest validation Macro F1 (ties keep the earlier epoch).
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), checkpoint)
    train_seconds = perf_counter() - start

    # Reload the saved best weights so the reported numbers come from the checkpoint file.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    result = evaluate(model, val_loader, device)
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    stats = model_stats(model, checkpoint)
    latency_ms = measure_latency(model, args.input_size, device, runs=100)
    latency_device = (torch.cuda.get_device_name(device) if device.type == "cuda"
                      else f"CPU ({platform.processor() or platform.machine()})")

    output = {
        "exp": args.exp,
        "stage": 1,
        "description": description,
        "class_order": CLASS_ORDER,
        "settings": {
            "model_name": MODEL_NAME, "conv_channels": [32, 64, 128, 256],
            "input_size": args.input_size, "channels": 1,
            "normalization_mean": mean, "normalization_std": std,
            "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "Adam",
            "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
            "data_augmentation": ({
                "horizontal_flip_p": args.flip_p, "vertical_flip_p": args.flip_p,
                "rotation_degrees": args.rotation,
            } if use_augmentation else None),
            "dropout_p": dropout if use_dropout else None,
            "train_count": len(train_loader.dataset),
            "validation_count": len(val_loader.dataset),
            "torch": torch.__version__,
        },
        "best_epoch": best_epoch,
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
        **stats,
        "checkpoint_sha256": checkpoint_hash,
        "latency_ms": latency_ms,
        "latency_device": latency_device,
        "train_seconds": train_seconds,
        "history": history,
        "misclassified": [
            {"filename": name, "true": true, "predicted": pred}
            for name, true, pred in zip(result["filenames"], result["y_true"], result["y_pred"])
            if true != pred
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.exp}.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_curves(history, args.out_dir / f"{args.exp}_curves.png")
    plot_confusion_matrix(result["confusion_matrix"], args.out_dir / f"{args.exp}_cm.png")
    scores = " ".join(f"{key}={value:.4f}" for key, value in result["metrics"].items())
    print(f"[{args.exp}] best_epoch={best_epoch} {scores} params={stats['params']} "
          f"latency_ms={latency_ms:.2f} train_seconds={train_seconds:.1f}")


if __name__ == "__main__":
    main()
