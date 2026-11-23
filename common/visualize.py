"""Visualize validation predictions and measure the effect of confidence thresholds."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from prepare_yolo import detection_pairs, read_class_names, read_yolo_boxes
from seed import set_seed


MATCH_IOU = 0.5
LOCALIZATION_IOU = 0.1


def box_iou(a, b):
    """Compute the intersection over union of two xyxy pixel boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - w * h
    return w * h / union if union > 0 else 0.0


def match_detections(truth, predictions, conf):
    """Match by score to unused same-class GT; count a localization miss as FP and FN."""
    remaining = set(range(len(truth)))
    unmatched = []
    ordered = sorted(enumerate(predictions), key=lambda item: -item[1][2])
    tp = 0
    for index, (cls, box, score) in ordered:
        if score < conf:
            continue
        candidates = ((box_iou(box, truth[j][1]), j) for j in sorted(remaining)
                      if truth[j][0] == cls)
        iou, target = max(candidates, default=(0.0, None))
        if iou >= MATCH_IOU:
            remaining.remove(target)
            tp += 1
        else:
            unmatched.append(index)
    counts = (tp, len(unmatched), len(remaining))
    errors = []
    for index in unmatched:
        cls, box, _ = predictions[index]
        candidates = ((box_iou(box, truth[j][1]), j) for j in sorted(remaining)
                      if truth[j][0] == cls)
        iou, target = max(candidates, default=(0.0, None))
        if iou >= LOCALIZATION_IOU:
            errors.append(dict(type="LOC", prediction=index, target=target, iou=iou))
            remaining.remove(target)
        else:
            errors.append(dict(type="FP", prediction=index, target=None))
    errors.extend(dict(type="FN", prediction=None, target=j) for j in sorted(remaining))
    return counts, errors


def collect_predictions(args, names, thresholds):
    """Run one post-NMS pass over validation, keeping original-pixel boxes and scores."""
    from ultralytics import YOLO

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Local trained checkpoint required: {args.checkpoint}")
    model = YOLO(str(args.checkpoint), task="detect")
    if model.task != "detect" or [model.names[i] for i in range(len(model.names))] != names:
        raise ValueError("Expected a detection checkpoint with the six course class names in order")
    if getattr(model.model, "end2end", False):
        raise ValueError("This threshold analysis expects a YOLO detector using NMS")
    pairs = {image.name: (image, label)
             for image, label in detection_pairs(args.data_root, "validation")}
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=str(next(iter(pairs.values()))[0].parent), stream=True, batch=1,
        imgsz=args.input_size, conf=min(thresholds), iou=args.iou,
        device=device, verbose=False, save=False, augment=False,
    )
    samples = []
    for result in results:
        image, label = pairs.pop(Path(result.path).name)
        height, width = result.orig_shape
        scale = (width, height, width, height)
        truth = [(cls, [v * s for v, s in zip(box, scale)])
                 for cls, box in read_yolo_boxes(label, len(names))]
        boxes = result.boxes.cpu()
        predictions = [(int(cls), box, score) for cls, box, score in
                       zip(boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist())]
        samples.append((image, truth, predictions))
    if pairs:
        raise ValueError(f"Missing predictions for {len(pairs)} validation images")
    return sorted(samples, key=lambda sample: sample[0].name)


def draw_comparison(sample, names, conf, exp, output, error=None):
    """Save side-by-side green GT and red scored predictions; mark the selected failure."""
    path, truth, predictions = sample
    figure = Figure(figsize=(9, 4.8), layout="constrained")
    left, right = figure.subplots(1, 2)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    for ax, title in ((left, "Ground Truth"), (right, "Prediction")):
        ax.imshow(pixels)
        ax.set(title=title, xlim=(-0.5, pixels.shape[1] - 0.5),
               ylim=(pixels.shape[0] - 0.5, -0.5))
        ax.axis("off")

    def draw_box(ax, box, text, color, highlight):
        """Draw an outline and a readable label within the image."""
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=2.8 if highlight else 1.5))
        if highlight:
            text = f"{error['type']} | {text}"
        align = "right" if x1 > pixels.shape[1] / 2 else "left"
        ax.text(x2 if align == "right" else x1, max(0, y1), text, ha=align, va="top",
                fontsize=7, color="white", bbox=dict(facecolor=color, edgecolor="none", pad=1))

    for i, (cls, box) in enumerate(truth):
        draw_box(left, box, names[cls], "green", error and error["target"] == i)
    for i, (cls, box, score) in enumerate(predictions):
        if score >= conf:
            draw_box(right, box, f"{names[cls]} {score:.2f}", "red",
                     error and error["prediction"] == i)
    suffix = f" | {error['type']}" if error else ""
    figure.suptitle(f"{exp}: {path.name} | confidence {conf:g}{suffix}")
    figure.savefig(output, dpi=150)


def choose_failures(failures):
    """Cycle through FP, FN and LOC and select up to five different source images."""
    selected, used = [], set()
    while len(selected) < 5:
        previous = len(selected)
        for kind in ("FP", "FN", "LOC"):
            case = next(((s, e) for s, e in failures
                         if e["type"] == kind and s[0].name not in used), None)
            if case is not None:
                selected.append(case)
                used.add(case[0][0].name)
                if len(selected) == 5:
                    break
        if len(selected) == previous:
            break
    return selected


def publish_figures(work, out_dir, report_name):
    """Replace only files owned by this run's manifest, leaving other work intact."""
    old_manifest = out_dir / report_name
    owned = set()
    if old_manifest.is_file():
        old = json.loads(old_manifest.read_text(encoding="utf-8"))
        owned = {item["file"] for item in old["comparisons"] + old["failure_cases"]}
    obsolete = []
    for folder, pattern in (("vis", "val_[0-9][0-9].png"),
                            ("failure", "fail_[0-9][0-9]_*.png")):
        dest = out_dir / folder
        dest.mkdir(exist_ok=True)
        for png in dest.glob(pattern):
            if f"{folder}/{png.name}" not in owned:
                raise FileExistsError(f"{png} belongs to another run. Use a fresh --out_dir.")
            if not (work / folder / png.name).exists():
                obsolete.append(png)
    for folder in ("vis", "failure"):
        for png in (work / folder).glob("*.png"):
            png.replace(out_dir / folder / png.name)
    for png in obsolete:
        png.unlink()
    (work / report_name).replace(old_manifest)


def export_analysis(samples, args, names, thresholds):
    """Write ten comparisons, up to five distinct failure images and full-split P/R counts."""
    totals = {t: np.zeros(3, dtype=int) for t in thresholds}
    failures = []
    for sample in samples:
        for t in thresholds:
            counts, errors = match_detections(sample[1], sample[2], t)
            totals[t] += counts
            if t == args.conf:
                failures.extend((sample, error) for error in errors)
    sweep = []
    for t, row in totals.items():
        tp, fp, fn = map(int, row)
        sweep.append(dict(conf_threshold=t, tp=tp, fp=fp, fn=fn,
                          precision=tp / (tp + fp) if tp + fp else 0.0,
                          recall=tp / (tp + fn) if tp + fn else 0.0))

    report = dict(exp=args.exp, validation_count=len(samples), input_size=args.input_size,
                  checkpoint_file=args.checkpoint.name, class_order=names, bbox_format="xyxy_pixel",
                  conf_threshold=args.conf, nms_iou_threshold=args.iou, match_iou=MATCH_IOU,
                  localization_min_iou=LOCALIZATION_IOU, threshold_sweep=sweep,
                  comparisons=[], failure_cases=[])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"{args.exp}_visualization.json"
    with TemporaryDirectory(dir=args.out_dir) as tmp:
        work = Path(tmp)
        (work / "vis").mkdir()
        (work / "failure").mkdir()
        indices = np.linspace(0, len(samples) - 1, min(10, len(samples)), dtype=int)
        for n, i in enumerate(indices, 1):
            sample = samples[i]
            filename = f"vis/val_{n:02}.png"
            draw_comparison(sample, names, args.conf, args.exp, work / filename)
            report["comparisons"].append(dict(file=filename, source=sample[0].name))
        for n, (sample, error) in enumerate(choose_failures(failures), 1):
            kind = error["type"]
            filename = f"failure/fail_{n:02}_{kind}.png"
            draw_comparison(sample, names, args.conf, args.exp, work / filename, error)
            item = dict(file=filename, source=sample[0].name, type=kind)
            if error["target"] is not None:
                cls, box = sample[1][error["target"]]
                item["ground_truth"] = dict(label=names[cls], xyxy=box)
            if error["prediction"] is not None:
                cls, box, score = sample[2][error["prediction"]]
                item["prediction"] = dict(label=names[cls], xyxy=box, confidence=score)
            if "iou" in error:
                item["iou"] = error["iou"]
            report["failure_cases"].append(item)
        (work / report_name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")
        publish_figures(work, args.out_dir, report_name)
    return report


def parse_args():
    """Read the Stage 3 reporting thresholds from config unless explicitly overridden."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--exp", default="D1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.checkpoint = args.checkpoint or Path(config["checkpoint_file"])
    for name, key in (("input_size", "input_size"), ("conf", "confidence_threshold"),
                      ("iou", "nms_iou_threshold")):
        if getattr(args, name) is None:
            setattr(args, name, config[key])
        if getattr(args, name) is None:
            parser.error(f"Set {key} in the config or pass --{name} for this NMS detector")
    if args.input_size < 1 or not all(0 <= t <= 1 for t in [args.conf, args.iou, *args.thresholds]):
        parser.error("Use a positive input size and thresholds between 0 and 1")
    if not args.exp.isalnum():
        parser.error("Use an alphanumeric experiment ID such as D1")
    return args, config


def main():
    """Export validation figures and confidence analysis from the selected real checkpoint."""
    args, config = parse_args()
    set_seed(args.seed)
    names = read_class_names(args.data_root, config)
    thresholds = sorted(set([args.conf, *args.thresholds]))
    samples = collect_predictions(args, names, thresholds)
    report = export_analysis(samples, args, names, thresholds)
    print(f"[{args.exp}] validation={len(samples)} comparisons={len(report['comparisons'])} "
          f"failure_images={len(report['failure_cases'])} report={args.out_dir}")


if __name__ == "__main__":
    main()
