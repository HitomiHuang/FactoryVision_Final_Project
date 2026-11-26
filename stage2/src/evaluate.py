"""Load the Stage 2 checkpoint and evaluate on the validation split."""

import argparse
import json
from pathlib import Path

import torch

from data import CLASS_ORDER, get_dataloaders
from metrics import evaluate, model_stats, resolve_device
from model import build_model


def parse_args():
    """Parse the checkpoint, its config and the evaluation device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("stage2_model.pth"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    """Rebuild the model from config.json, load its weights and print validation metrics."""
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["class_order"] != CLASS_ORDER:
        raise ValueError(f"config class_order must be {CLASS_ORDER}")
    device = resolve_device(args.device)

    model = build_model(config["architecture"], config["strategy"], pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    _, val_loader = get_dataloaders(
        args.data_root, config["input_size"], config["normalization_mean"],
        config["normalization_std"], args.batch_size, args.num_workers,
        channels=3, seed=config["seed"],
    )
    result = evaluate(model, val_loader, device)
    stats = model_stats(model, args.checkpoint)

    scores = " ".join(f"{key}={value:.4f}" for key, value in result["metrics"].items())
    print(f"[evaluate] {config['architecture']} + {config['strategy']} "
          f"checkpoint={args.checkpoint} validation={len(result['y_true'])} {scores}")
    print(f"[evaluate] params={stats['params']} trainable={stats['trainable_params']} "
          f"size={stats['checkpoint_mb']:.2f}MB")
    print("[evaluate] confusion matrix (rows = true, columns = predicted):")
    for name, row in zip(CLASS_ORDER, result["confusion_matrix"]):
        print(f"  {name:<16}" + " ".join(f"{count:4d}" for count in row))


if __name__ == "__main__":
    main()
