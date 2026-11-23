"""Package only D's verified starter files for A, B and C."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


COMMON = Path(__file__).resolve().parent
CLASSIFICATION = ["data.py", "metrics.py", "seed.py", "check_classification.py"]
MODULES = {
    1: [*CLASSIFICATION, "pca_svm.py"],
    2: CLASSIFICATION,
    3: ["prepare_yolo.py", "visualize.py", "seed.py"],
}


def stage_files(stage):
    """Collect D's handoff files and reject source drift or a mismatched PCA checkpoint."""
    root = COMMON.parent / f"stage{stage}"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config["stage"] != stage or not re.fullmatch(r"Team[0-9]{2}", config["team_id"]):
        raise ValueError(f"Set the confirmed team ID and stage in {root.name}/config.json")
    files = {
        "README.md": COMMON / "handoff" / f"Stage{stage}.md",
        "SETUP.md": COMMON / "handoff" / "SETUP.md",
        "SUBMISSION_README_TEMPLATE.md": COMMON / "templates" / "README.md",
        "evidence/dataset_audit.json": COMMON / "evidence" / "dataset_audit.json",
        "evidence/environment.json": COMMON / "evidence" / "environment.json",
        "config.json": root / "config.json",
        "requirements.txt": root / "requirements.txt",
    }
    for module in MODULES[stage]:
        dest = root / "src" / module
        if (COMMON / module).read_bytes() != dest.read_bytes():
            raise ValueError(f"Sync {root.name}/src/{module} from common before packing")
        files[f"src/{module}"] = dest
    if stage == 1:
        for name in ("checkpoints/pca_svm.pkl", "results/pca_svm.json", "results/pca_svm_cm.png"):
            files[name] = root / name
        result = json.loads(files["results/pca_svm.json"].read_text(encoding="utf-8"))
        with files["checkpoints/pca_svm.pkl"].open("rb") as checkpoint:
            if hashlib.file_digest(checkpoint, "sha256").hexdigest() != result["checkpoint_sha256"]:
                raise ValueError("PCA checkpoint and metrics must come from the same run")
    return f"{config['team_id']}_Stage{stage}_handoff", files


def parse_args():
    """Parse the local destination for the three team handoff archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=COMMON / "handoff")
    return parser.parse_args()


def main():
    """Build all three archives, check their content and publish them after validation."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out_dir) as tmp:
        archives = []
        for stage in MODULES:
            name, files = stage_files(stage)
            archive = Path(tmp) / f"{name}.zip"
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
                for member, path in files.items():
                    output.write(path, f"{name}/{member}")
            with ZipFile(archive) as output:
                if output.testzip() or len(output.namelist()) != len(files):
                    raise ValueError(f"Incomplete archive: {archive.name}")
            archives.append(archive)
        for archive in archives:
            target = archive.replace(args.out_dir / archive.name)
            print(f"{target.name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
