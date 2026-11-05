#!/usr/bin/env python
"""Convert a folder of labelme JSON files into one COCO instance-segmentation set.

Normally run through prepare_dataset.py, which validates every split and builds
a numbered dataset version in one go. This CLI is still here for one-off
conversions of a single folder.
"""

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


IGNORE_LABEL = "__ignore__"

# --image_format value -> (image folder, file extension). The annotations file
# is annotations.json either way.
IMAGE_FORMATS = {
    "jpeg": ("JPEGImages", ".jpg"),
    "png": ("PNGImages", ".png"),  # lossless, matching PNG sources
}


def load_labels(path):
    """Return the class names in labels.txt, in category-ID order.

    The first non-blank line must be __ignore__ (it gets no category); each
    line after it is one class, with IDs counting up from 0.
    """
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    if not names or names[0] != IGNORE_LABEL:
        raise ValueError(f"{path}: the first line must be {IGNORE_LABEL}")
    names = names[1:]
    if not names:
        raise ValueError(f"{path}: no classes listed after {IGNORE_LABEL}")
    if dupes := sorted(n for n, c in collections.Counter(names).items() if c > 1):
        raise ValueError(f"{path}: duplicate label(s): {', '.join(dupes)}")
    return names


def convert(label_files, output_dir, class_names, image_format="jpeg",
            noviz=False, progress=None):
    """Write label_files as a COCO dataset into output_dir, which must not exist.

    Image IDs follow the order of label_files. Shapes whose label is not in
    class_names are dropped. `progress`, if given, wraps the file iteration
    (e.g. a tqdm factory) in place of the per-file log line.

    Returns counts: images, annotations, background_images, objects per label.
    """
    image_dir, image_ext = IMAGE_FORMATS[image_format]

    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, image_dir))
    if not noviz:
        os.makedirs(osp.join(output_dir, "Visualization"))

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for class_id, class_name in enumerate(class_names):
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    stats = dict(images=0, annotations=0, background_images=0,
                 objects=collections.Counter())

    out_ann_file = osp.join(output_dir, "annotations.json")
    files = progress(label_files) if progress else label_files
    for image_id, filename in enumerate(files):
        if not progress:
            print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, image_dir, base + image_ext)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                # x: tolerance of the gap between the arc and the line segment
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        kept = 0
        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )
            kept += 1
            stats["objects"][cls_name] += 1

        stats["images"] += 1
        stats["annotations"] += kept
        if kept == 0:
            stats["background_images"] += 1

        if not noviz:
            viz = img
            instances = [
                (class_name_to_id[cnm], cnm, msk)
                for (cnm, gid), msk in masks.items()
                if cnm in class_name_to_id
            ]
            if instances:
                labels, captions, viz_masks = zip(*instances)
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=viz_masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            # Previews stay JPEG whatever image_format is; they are only for
            # eyeballing the conversion, not for training.
            out_viz_file = osp.join(output_dir, "Visualization", base + ".jpg")
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)

    return stats


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", help="input annotated directory", required=True)
    parser.add_argument("--output_dir", help="output dataset directory", required=True)
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--image_format",
        help="output image format",
        choices=sorted(IMAGE_FORMATS),
        default="jpeg",
    )
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    try:
        class_names = load_labels(args.labels)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print("Creating dataset:", args.output_dir)
    # Sorted so image IDs are reproducible across re-conversions.
    label_files = sorted(glob.glob(osp.join(args.input_dir, "*.json")))
    convert(label_files, args.output_dir, class_names,
            image_format=args.image_format, noviz=args.noviz)


if __name__ == "__main__":
    main()
