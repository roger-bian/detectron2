### Normal use: `prepare_dataset.py`

You do not usually run this converter yourself. `prepare_dataset.py` at the
repo root checks every split of a dataset and calls it to build the next
numbered version:

```
python prepare_dataset.py {dataset_name}
```

See "Prepare a dataset" in the top-level README.

### Converting a single folder

For a one-off conversion of one folder of labelme JSON outside the
`_data/{dataset_name}/` layout:

```
# It generates:
#   - {output_folder}/JPEGImages   (or PNGImages, see --image_format)
#   - {output_folder}/Visualization
#   - {output_folder}/annotations.json

python ./labelme2coco/labelme2coco.py \
  --input_dir {input_folder} \
  --output_dir {output_folder} \
  --labels {labels.txt} \
  --image_format jpeg
```

`--labels` is a label list with `__ignore__` first, one label per line.
`--output_dir` must not exist yet. Unlike `prepare_dataset.py`, this does no
checks: shapes whose label is missing from the label list are dropped silently.

`--image_format jpeg` (the default) writes `JPEGImages/*.jpg`;
`--image_format png` writes `PNGImages/*.png`, which is lossless and so keeps
PNG sources pixel-for-pixel, at roughly 10x the disk space. The annotations
file is `annotations.json` either way, and `Visualization/` is always JPEG.
