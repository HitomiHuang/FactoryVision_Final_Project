"""Fit a traditional PCA + SVM baseline on the official train split."""

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from data import CLASS_ORDER, classification_samples, load_numpy
from metrics import compute_metrics, plot_confusion_matrix
from seed import set_seed


def parse_gamma(value):
    """Accept an SVM preset or a finite, nonnegative RBF coefficient."""
    if value in ("scale", "auto"):
        return value
    message = "gamma must be 'scale', 'auto', or a finite nonnegative number"
    try:
        gamma = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not np.isfinite(gamma) or gamma < 0:
        raise argparse.ArgumentTypeError(message)
    return gamma


def parse_args():
    """Parse the reproducible baseline settings and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--kernel", choices=("rbf", "linear"), default="rbf")
    parser.add_argument(
        "--gamma", type=parse_gamma, default="scale", metavar="GAMMA",
        help="RBF gamma: scale (default), auto, or a nonnegative number.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/pca_svm.pkl")
    )
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    """Fit on train and save the model, validation scores and count matrix."""
    args = parse_args()
    set_seed(args.seed)
    x_train, y_train, x_val, y_val = load_numpy(
        args.data_root, args.input_size
    )
    # Full float64 SVD is feasible here and gives reproducible predictions.
    x_train = x_train.astype(np.float64)
    x_val = x_val.astype(np.float64)
    model = make_pipeline(
        PCA(n_components=args.components, svd_solver="full"),
        SVC(C=args.c, kernel=args.kernel, gamma=args.gamma),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    # The pipeline processes one already-decoded image at a time on the CPU.
    batch = x_val[:1]
    for _ in range(10):
        model.predict(batch)
    start = perf_counter()
    for _ in range(100):
        model.predict(batch)
    latency_ms = (perf_counter() - start) * 1000 / 100

    result = compute_metrics(y_val, predictions)
    val_samples = classification_samples(args.data_root, "validation")
    result.update(
        exp="pca_svm", stage=1,
        description=f"Traditional baseline: PCA + {args.kernel.upper()} SVM",
        class_order=CLASS_ORDER,
        settings={
            "model_name": "PCA + SVM", "input_size": args.input_size,
            "channels": 1,
            "preprocessing": (
                "bilinear resize, grayscale pixels / 255, flatten"
            ),
            "pca_components": args.components, "svd_solver": "full",
            "fit_dtype": "float64", "whiten": False,
            "kernel": args.kernel, "C": args.c, "gamma": args.gamma,
            "seed": args.seed, "scikit_learn": sklearn.__version__,
            "train_count": len(y_train), "validation_count": len(y_val),
        },
        best_epoch=None, params=None, trainable_params=None,
        checkpoint_mb=args.checkpoint.stat().st_size / (1024**2),
        checkpoint_sha256=checkpoint_hash,
        latency_ms=latency_ms,
        latency_device=f"CPU ({platform.processor() or platform.machine()})",
        history={
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
        },
        misclassified=[
            {"filename": f"{CLASS_ORDER[label]}/{path.name}", "true": label,
             "predicted": int(pred)}
            for (path, label), pred in zip(val_samples, predictions)
            if label != pred
        ],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pca_svm.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_confusion_matrix(
        result["confusion_matrix"], args.out_dir / "pca_svm_cm.png"
    )
    scores = " ".join(
        f"{key}={value:.4f}" for key, value in result["metrics"].items()
    )
    print(f"[pca_svm] train={len(y_train)} validation={len(y_val)} {scores}")


if __name__ == "__main__":
    main()
