"""Load the Stage 1 checkpoint and evaluate on the validation split."""

import argparse
import json
from pathlib import Path

import torch

from data import CLASS_ORDER, get_dataloaders
from metrics import evaluate, resolve_device
from model import build_model


def parse_args():
    """Parse the data, checkpoint and config locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("stage1_model.pth"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    """Rebuild the model from config.json, load the weights and print validation metrics."""
    args = parse_args()
    # utf-8-sig also accepts files saved with a BOM (e.g. by Windows PowerShell).
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config["class_order"] != CLASS_ORDER:
        raise ValueError("config.json class_order differs from CLASS_ORDER")
    device = resolve_device(args.device)

    methods = config["improvement_methods"] or {}
    dropout = (methods.get("dropout") or {}).get("p", 0.0)
    model = build_model(in_channels=1, num_classes=len(CLASS_ORDER), dropout=dropout)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    # Validation always uses the basic Resize -> ToTensor -> Normalize pipeline.
    _, val_loader = get_dataloaders(
        args.data_root, config["input_size"], config["normalization_mean"],
        config["normalization_std"], args.batch_size, args.num_workers,
        channels=1, seed=config["seed"],
    )
    result = evaluate(model, val_loader, device)

    print(f"[evaluate] checkpoint={args.checkpoint} model={config['model_name']} "
          f"validation={len(result['y_true'])}")
    print(" ".join(f"{key}={value:.4f}" for key, value in result["metrics"].items()))
    print("confusion_matrix (rows=true, cols=predicted):")
    for name, row in zip(CLASS_ORDER, result["confusion_matrix"]):
        print(f"  {name:<16} {row}")
    wrong = [(name, true, pred) for name, true, pred
             in zip(result["filenames"], result["y_true"], result["y_pred"]) if true != pred]
    print(f"misclassified={len(wrong)}")
    for name, true, pred in wrong:
        print(f"  {name}: true={CLASS_ORDER[true]} predicted={CLASS_ORDER[pred]}")


if __name__ == "__main__":
    main()
