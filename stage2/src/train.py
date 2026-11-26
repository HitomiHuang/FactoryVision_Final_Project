"""Train the Stage 2 classifier and select a checkpoint by validation Macro F1."""

import argparse
import json
import platform
from pathlib import Path

import torch
import torchvision
from torch import nn

from data import get_dataloaders
from metrics import (
    compute_metrics, evaluate, measure_latency, model_stats,
    plot_confusion_matrix, plot_curves, resolve_device,
)
from model import ARCHITECTURES, STRATEGIES, build_model, pretrained_weights
from seed import set_seed


def parse_args():
    """Parse the experiment settings shared by every Stage 2 run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--exp", default="E1")
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="resnet18")
    parser.add_argument("--strategy", choices=STRATEGIES, default="feature_extraction")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one pass over a loader; update weights only when an optimizer is given."""
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())
    return total_loss / len(y_true), compute_metrics(y_true, y_pred)["metrics"]


def describe_device(device):
    """Return a readable device name for the results JSON."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def main():
    """Train one architecture/strategy pair and save its results in the team format."""
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    weights = pretrained_weights(args.architecture)
    preprocessing = weights.transforms()
    mean, std = list(preprocessing.mean), list(preprocessing.std)
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.input_size, mean, std,
        args.batch_size, args.num_workers, channels=3, seed=args.seed,
    )

    model = build_model(args.architecture, args.strategy).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    checkpoint = Path("checkpoints") / f"{args.exp}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        # Feature extraction keeps the frozen backbone's ImageNet BatchNorm
        # statistics, so only full fine-tuning switches the network to train mode.
        model.train(args.strategy == "full_finetune")
        train_loss, train_scores = run_epoch(model, train_loader, criterion, device, optimizer)
        model.eval()
        val_loss, val_scores = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_scores["accuracy"])
        history["val_acc"].append(val_scores["accuracy"])
        history["val_f1"].append(val_scores["macro_f1"])
        print(f"[{args.exp}] epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_scores['accuracy']:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_scores['accuracy']:.4f} "
              f"val_f1={val_scores['macro_f1']:.4f}")

        # Keep the epoch with the highest validation Macro F1; among tied epochs,
        # keep the one with the lowest validation loss.
        f1 = val_scores["macro_f1"]
        if f1 > best_f1 or (f1 == best_f1 and val_loss < best_loss):
            best_f1, best_loss, best_epoch = f1, val_loss, epoch
            torch.save(model.state_dict(), checkpoint)

    # Reload the saved checkpoint so the reported numbers come from the file itself.
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    result = evaluate(model, val_loader, device)
    stats = model_stats(model, checkpoint)
    # measure_latency only synchronizes CUDA, so MPS timings would be invalid;
    # time on the CPU instead when training on Apple MPS.
    latency_device = torch.device("cpu") if device.type == "mps" else device
    latency_ms = measure_latency(model, args.input_size, latency_device, runs=100)

    children = [(name, list(module.parameters())) for name, module in model.named_children()]
    trainable_modules = [n for n, ps in children if ps and any(p.requires_grad for p in ps)]
    frozen_modules = [n for n, ps in children if ps and not any(p.requires_grad for p in ps)]

    output = {
        "exp": args.exp,
        "stage": 2,
        "description": f"{args.architecture} + {args.strategy}",
        "settings": {
            "model_name": args.architecture,
            "architecture": args.architecture,
            "strategy": args.strategy,
            "pretrained_source": f"torchvision {weights} ({weights.url})",
            "trainable_modules": trainable_modules,
            "frozen_modules": frozen_modules,
            "input_size": args.input_size,
            "channels": 3,
            "preprocessing": "grayscale to 3-channel RGB, Resize((input_size, input_size)), "
                             "ToTensor, Normalize(ImageNet mean/std)",
            "normalization_mean": mean,
            "normalization_std": std,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "train_device": describe_device(device),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "best_epoch": best_epoch,
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
        **stats,
        "latency_ms": latency_ms,
        "latency_device": describe_device(latency_device),
        "history": history,
        "misclassified": [
            {"filename": name, "true": t, "predicted": p}
            for name, t, p in zip(result["filenames"], result["y_true"], result["y_pred"])
            if t != p
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
          f"trainable={stats['trainable_params']} size={stats['checkpoint_mb']:.2f}MB "
          f"latency={latency_ms:.2f}ms ({output['latency_device']})")


if __name__ == "__main__":
    main()
