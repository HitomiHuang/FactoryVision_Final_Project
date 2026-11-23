"""Validate the course detection labels and copy the official splits for Ultralytics."""

import argparse
import filecmp
import json
import math
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def read_class_names(data_root, config):
    """Check the course class indices against the Stage 3 config supplied by D."""
    names_path = Path(data_root) / "detection" / "class_names.txt"
    names = names_path.read_text(encoding="utf-8-sig").splitlines()
    if names != config["class_order"] or config["stage"] != 3:
        raise ValueError("The dataset class order must match the Stage 3 config")
    return names


def detection_pairs(data_root, split):
    """Pair every image with its YOLO text file and reject missing or extra labels."""
    root = Path(data_root) / "detection" / split
    images = {p.stem: p for p in (root / "images").glob("*.png")}
    labels = {p.stem: p for p in (root / "labels_yolo").glob("*.txt")}
    if not images or images.keys() != labels.keys():
        missing = sorted(images.keys() - labels.keys())[:3]
        extra = sorted(labels.keys() - images.keys())[:3]
        raise ValueError(f"{root}: missing labels {missing}, extra labels {extra}, "
                         f"images={len(images)}")
    return [(images[name], labels[name]) for name in sorted(images)]


def read_yolo_boxes(path, class_count):
    """Parse normalized center-format boxes as (class index, normalized xyxy)."""
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            label, x, y, w, h = line.split()
            label = int(label)
            x, y, w, h = map(float, (x, y, w, h))
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if not (0 <= label < class_count and w > 0 and h > 0):
                raise ValueError
            if not all(math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in box):
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid YOLO detection label") from exc
        boxes.append((label, tuple(min(1.0, max(0.0, v)) for v in box)))
    return boxes


def prepare_dataset(data_root, out_dir, config):
    """Build a portable dataset or verify an existing copy against the originals."""
    source, out = Path(data_root).resolve(), Path(out_dir).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Choose an output directory separate from the original dataset")
    names = read_class_names(source, config)
    lines = ["train: images/train", "val: images/validation", "names:"]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    yaml = "\n".join(lines) + "\n"
    copies = {}
    counts = {}
    for split in ("train", "validation"):
        pairs = detection_pairs(source, split)
        box_count = 0
        for image, label in pairs:
            box_count += len(read_yolo_boxes(label, len(names)))
            copies[Path("images", split, image.name)] = image
            copies[Path("labels", split, label.name)] = label
        counts[split] = {"images": len(pairs), "boxes": box_count}

    if out.exists():
        # Validate previous copies, even on reruns after the source or output changed.
        existing = {p.relative_to(out) for kind in ("images", "labels")
                    for p in (out / kind).glob("*/*") if p.is_file()}
        yaml_path = out / "data.yaml"
        if (existing != copies.keys() or not yaml_path.is_file()
                or yaml_path.read_text(encoding="utf-8") != yaml):
            raise ValueError(f"{out} differs from the expected layout. Use a fresh --out_dir.")
        for relative, original in copies.items():
            if not filecmp.cmp(original, out / relative, shallow=False):
                raise ValueError(f"Content differs at {relative}. Use a fresh --out_dir.")
        return counts

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".factoryvision-") as temp:
        staged = Path(temp) / "dataset"
        for relative, original in copies.items():
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        (staged / "data.yaml").write_text(yaml, encoding="utf-8")
        staged.rename(out)
    return counts


def parse_args():
    """Parse the source, portable destination and Stage 3 config paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("../data/factoryvision_yolo"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    return parser.parse_args()


def main():
    """Prepare or verify both splits, then print the measured image and box counts."""
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    counts = prepare_dataset(args.data_root, args.out_dir, config)
    print(f"Dataset YAML: {args.out_dir / 'data.yaml'}")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
