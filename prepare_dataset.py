#!/usr/bin/env python
"""Build the next numbered COCO version of a dataset from its labelme splits.

    python prepare_dataset.py my_dataset

Every folder under _data/{dataset}/labelme/ is a split: train/ and val/ are
required, test/ (or any other name) is optional, and folders starting with "_"
or "." are skipped. All splits are checked first, and nothing is written if any
check fails. Then each split is converted into _data/{dataset}/coco/{split}_vN/,
where N is one more than the highest version already there, so train_vN, val_vN,
and test_vN always belong together. Each version folder also gets a
manifest.json recording exactly which files went into it, and a copy of
labels.txt.

The new version is not written into settings.yaml; set train.version yourself
when you want to train on it.
"""

import argparse
import base64
import collections
import datetime
import glob
import hashlib
import io
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import PureWindowsPath

import PIL.Image
from tqdm import tqdm

from labelme2coco.labelme2coco import IGNORE_LABEL, IMAGE_FORMATS, convert, load_labels

DATA_ROOT = "_data"
REQUIRED_SPLITS = ("train", "val")  # train.py reads both
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUPPORTED_SHAPES = {"polygon", "rectangle", "circle"}
VERSION_DIR = re.compile(r"^(?P<split>.+)_v(?P<num>\d+)$")
MAX_EXAMPLES = 10  # detail lines printed per problem; manifests keep them all


class Report:
    """Problems found while checking, grouped by kind.

    Errors always stop the build. Warnings stop it only under --strict.
    """

    def __init__(self):
        self.items = {}  # (level, kind) -> [details], in the order found

    def add(self, level, kind, detail):
        self.items.setdefault((level, kind), []).append(detail)

    def count(self, level):
        return sum(1 for lvl, _ in self.items if lvl == level)

    def warnings_for(self, split):
        """Warning details about one split, for its manifest."""
        return [
            f"{kind}: {d}"
            for (lvl, kind), details in self.items.items() if lvl == "warning"
            for d in details if d.startswith(f"{split}/")
        ]

    def print(self):
        for (level, kind), details in self.items.items():
            print(f"\n{level.upper()}: {kind} ({len(details)})")
            for d in details[:MAX_EXAMPLES]:
                print(f"  {d}")
            if len(details) > MAX_EXAMPLES:
                print(f"  ... and {len(details) - MAX_EXAMPLES} more")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def list_splits(labelme_dir):
    return [
        name for name in sorted(os.listdir(labelme_dir))
        if not name.startswith(("_", ".")) and osp.isdir(osp.join(labelme_dir, name))
    ]


def check_shape(shape):
    """Return why the converter cannot turn this shape into a mask, or None."""
    shape_type = shape.get("shape_type") or "polygon"
    points = shape.get("points") or []
    if shape_type not in SUPPORTED_SHAPES:
        return f"shape_type {shape_type!r} (supported: {', '.join(sorted(SUPPORTED_SHAPES))})"
    if shape_type == "polygon" and len(points) < 3:
        return f"polygon has {len(points)} points"
    if shape_type in ("rectangle", "circle") and len(points) != 2:
        return f"{shape_type} needs 2 points, has {len(points)}"
    if shape_type == "circle":
        (x1, y1), (x2, y2) = points
        if ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 < 1:
            return "circle radius under 1 px"
    return None


def scan_split(labelme_dir, split, class_names, report):
    """Check one split folder.

    Returns (a manifest entry per JSON file, image files with no JSON). Paths
    are relative to labelme/.
    """
    split_dir = osp.join(labelme_dir, split)
    known = set(class_names)

    entries = []
    annotated = set()  # image files that a JSON accounts for
    for path in sorted(glob.glob(osp.join(split_dir, "*.json"))):
        rel = f"{split}/{osp.basename(path)}"
        with open(path, "rb") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
            shapes = data["shapes"]
        except (ValueError, KeyError) as e:
            report.add("error", "unreadable labelme JSON", f"{rel}: {e}")
            continue

        # Same lookup as labelme.LabelFile: embedded imageData first, else
        # imagePath relative to the JSON (labelme may write it Windows-style).
        image_rel = PureWindowsPath(data.get("imagePath") or "").as_posix()
        image_path = osp.normpath(osp.join(split_dir, image_rel)) if image_rel else None
        if image_path and osp.isfile(image_path):
            annotated.add(image_path)

        if data.get("imageData"):
            image_bytes = base64.b64decode(data["imageData"])
        elif image_path and osp.isfile(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        else:
            report.add("error", "image missing (no imageData, imagePath not found)",
                       f"{rel} -> {image_rel or '(empty imagePath)'}")
            continue
        try:
            PIL.Image.open(io.BytesIO(image_bytes)).verify()
        except Exception as e:
            report.add("error", "image cannot be read", f"{rel}: {e}")
            continue

        objects = collections.Counter()
        instances = set()
        for i, shape in enumerate(shapes):
            label = shape.get("label")
            if label == IGNORE_LABEL:
                continue
            if label not in known:
                report.add("warning", "label not in labels.txt (shape will be dropped)",
                           f"{rel} shape {i}: {label!r}")
                continue
            if problem := check_shape(shape):
                report.add("error", "shape the converter cannot handle",
                           f"{rel} shape {i} ({label}): {problem}")
                continue
            # Shapes sharing (label, group_id) merge into one object.
            gid = shape.get("group_id")
            instances.add((label, gid) if gid is not None else (label, "shape", i))
        for label, *_ in instances:
            objects[label] += 1

        entries.append(dict(
            json=rel,
            json_sha256=sha256(raw),
            image=image_rel or None,
            image_embedded=bool(data.get("imageData")),
            image_sha256=sha256(image_bytes),
            objects=dict(objects),
        ))

    unannotated = sorted(
        f"{split}/{osp.basename(p)}" for p in glob.glob(osp.join(split_dir, "*"))
        if osp.splitext(p)[1].lower() in IMAGE_EXTS and osp.normpath(p) not in annotated
    )
    for name in unannotated:
        report.add("warning", "image with no labelme JSON (not annotated yet, left out)", name)

    return entries, unannotated


def check_duplicates(entries_by_split, report):
    by_hash = collections.defaultdict(list)  # image hash -> [json paths]
    for entries in entries_by_split.values():
        for e in entries:
            by_hash[e["image_sha256"]].append(e["json"])
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        if len({p.split("/", 1)[0] for p in paths}) > 1:
            report.add("error", "same image in more than one split (leakage)", " = ".join(paths))
        else:
            report.add("warning", "same image twice in one split", " = ".join(paths))


def next_version(coco_dir):
    nums = [
        int(m["num"]) for name in (os.listdir(coco_dir) if osp.isdir(coco_dir) else [])
        if (m := VERSION_DIR.match(name)) and osp.isdir(osp.join(coco_dir, name))
    ]
    return max(nums, default=0) + 1


def git_state():
    """Commit hash and whether tracked files have uncommitted changes.

    Asked of this repo (where the script lives), not the working directory.
    """
    repo = osp.dirname(osp.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=repo, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def print_summary(entries_by_split, class_names):
    width = max(5, *(len(s) for s in entries_by_split))
    cols = [max(len(c), 6) for c in class_names]
    header = f"{'split':<{width}}  {'images':>6}  {'background':>10}  " + "  ".join(
        f"{c:>{w}}" for c, w in zip(class_names, cols))
    print("\n" + header)
    print("-" * len(header))
    for split, entries in entries_by_split.items():
        totals = collections.Counter()
        for e in entries:
            totals.update(e["objects"])
        background = sum(1 for e in entries if not e["objects"])
        print(f"{split:<{width}}  {len(entries):>6}  {background:>10}  " + "  ".join(
            f"{totals[c]:>{w}}" for c, w in zip(class_names, cols)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="dataset folder under _data/, e.g. my_dataset")
    parser.add_argument("--image_format", choices=sorted(IMAGE_FORMATS), default="jpeg",
                        help="png is lossless (matches PNG sources) at ~10x the disk space")
    parser.add_argument("--noviz", action="store_true", help="skip the Visualization/ previews")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings (unannotated images, unknown labels, ...) as errors")
    parser.add_argument("--dry_run", action="store_true",
                        help="run the checks and print the summary without writing anything")
    args = parser.parse_args()

    dataset = args.dataset.strip("/")
    if dataset.startswith(DATA_ROOT + "/"):
        dataset = dataset[len(DATA_ROOT) + 1:]
    dataset_dir = osp.join(DATA_ROOT, dataset)
    labelme_dir = osp.join(dataset_dir, "labelme")
    coco_dir = osp.join(dataset_dir, "coco")
    labels_path = osp.join(dataset_dir, "labels.txt")

    if not osp.isdir(labelme_dir):
        sys.exit(f"No labelme folder: {labelme_dir}")
    if not osp.isfile(labels_path):
        sys.exit(f"No labels file: {labels_path} (one label per line, {IGNORE_LABEL} first)")
    try:
        class_names = load_labels(labels_path)
    except ValueError as e:
        sys.exit(str(e))

    report = Report()
    entries_by_split = {}
    unannotated_by_split = {}
    for split in list_splits(labelme_dir):
        entries, unannotated = scan_split(labelme_dir, split, class_names, report)
        if entries:
            entries_by_split[split] = entries
            unannotated_by_split[split] = unannotated
        # An empty optional folder (e.g. an unused test/) is skipped quietly; a
        # missing or empty train/val is an error below.
    for split in REQUIRED_SPLITS:
        if split not in entries_by_split:
            report.add("error", "required split missing or empty",
                       f"labelme/{split}/ needs at least one annotated image")
    check_duplicates(entries_by_split, report)

    version = f"v{next_version(coco_dir)}"
    report.print()
    if entries_by_split:
        print_summary(entries_by_split, class_names)

    errors, warnings = report.count("error"), report.count("warning")
    if errors or (args.strict and warnings):
        reason = f"{errors} error type(s)" if errors else f"{warnings} warning type(s) with --strict"
        sys.exit(f"\nNot building: {reason}. Nothing was written.")
    if args.dry_run:
        print(f"\nDry run: checks passed; would build {version} "
              f"({', '.join(f'{s}_{version}' for s in entries_by_split)}).")
        return

    # Build every split in a hidden staging folder beside the final ones, then
    # move them into place only once all have converted, so a failure never
    # leaves a half-built version that would bump the next version number.
    os.makedirs(coco_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".building_{version}_", dir=coco_dir)
    commit, dirty = git_state()
    created = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        for split, entries in entries_by_split.items():
            out_dir = osp.join(staging, f"{split}_{version}")
            stats = convert(
                [osp.join(labelme_dir, e["json"]) for e in entries],
                out_dir, class_names,
                image_format=args.image_format, noviz=args.noviz,
                progress=lambda it, s=split: tqdm(it, desc=f"{s}_{version}", unit="img"),
            )
            shutil.copyfile(labels_path, osp.join(out_dir, "labels.txt"))
            manifest = dict(
                dataset=dataset,
                split=split,
                version=version,
                created=created,
                git_commit=commit,
                git_dirty=dirty,
                image_format=args.image_format,
                categories=class_names,
                images=stats["images"],
                background_images=stats["background_images"],
                annotations=stats["annotations"],
                objects=dict(stats["objects"]),
                unannotated_images=unannotated_by_split[split],
                warnings=report.warnings_for(split),
                files=entries,
            )
            with open(osp.join(out_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
        for split in entries_by_split:
            os.rename(osp.join(staging, f"{split}_{version}"),
                      osp.join(coco_dir, f"{split}_{version}"))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\nBuilt {version}:")
    for split in entries_by_split:
        print(f"  {osp.join(coco_dir, f'{split}_{version}')}")
    print("\nTo train on it, set in settings.yaml:")
    print(f"  dataset_name: {dataset}")
    print(f"  version: {version}")
    print(f"  num_classes: {len(class_names)}")
    if extra := [s for s in entries_by_split if s not in REQUIRED_SPLITS]:
        print(f"(train.py uses train and val only; {', '.join(extra)} is for your own evaluation.)")


if __name__ == "__main__":
    main()
