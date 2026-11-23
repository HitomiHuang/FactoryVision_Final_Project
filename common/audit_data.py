"""Audit source images, split overlap and detection labels without changing the data."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from data import CLASS_ORDER, classification_samples
from prepare_yolo import detection_pairs, read_class_names, read_yolo_boxes


def inspect_images(paths, root):
    """Count image formats and group relative filenames by SHA-256 of file bytes."""
    formats, hashes = Counter(), {}
    for path in paths:
        with Image.open(path) as image:
            formats[f"{image.format} {image.mode} {image.width}x{image.height}"] += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(path.relative_to(root).as_posix())
    return dict(formats), hashes


def audit_dataset(data_root):
    """Summarize the two original splits, matching annotations and identical image bytes."""
    root = Path(data_root)
    names = read_class_names(root, {"stage": 3, "class_order": CLASS_ORDER})
    result = {}
    for task in ("classification", "detection"):
        hashes = {}
        splits = {}
        for split in ("train", "validation"):
            if task == "classification":
                samples = classification_samples(root, split)
                paths = [p for p, _ in samples]
                counts = Counter(names[i] for _, i in samples)
                details = {"images_per_class": dict(counts)}
            else:
                pairs = detection_pairs(root, split)
                paths = [p for p, _ in pairs]
                counts = Counter(names[i] for _, label in pairs
                                 for i, _ in read_yolo_boxes(label, len(names)))
                voc = root / task / split / "annotations_voc"
                if {p.stem for p in voc.glob("*.xml")} != {p.stem for p in paths}:
                    raise ValueError(f"VOC filenames differ from the image list in {split}")
                details = {"boxes": sum(counts.values()), "boxes_per_class": dict(counts)}
            formats, hashes[split] = inspect_images(paths, root)
            splits[split] = dict(images=len(paths), formats=formats, **details)
        overlap = [{"sha256": h, "train": hashes["train"][h],
                    "validation": hashes["validation"][h]}
                   for h in sorted(hashes["train"].keys() & hashes["validation"].keys())]
        result[task] = dict(splits=splits, identical_files_across_splits=overlap)
    return result


def parse_args():
    """Parse the course data root and an optional evidence file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    """Print the audit and optionally save the same JSON for the handoff."""
    args = parse_args()
    report = json.dumps(audit_dataset(args.data_root), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
