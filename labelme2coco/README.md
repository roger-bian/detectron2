### Make sure labelme in installed
```
pip install labelme
```

### Update `labels.txt`
Update `labels.txt` to include all labels.
Leave the first `__ignore__` label as-is.


### Convert command
```
# It generates:
#   - data_dataset_coco/JPEGImages   (or PNGImages, see --image_format)
#   - data_dataset_coco/Visualization
#   - data_dataset_coco/annotations.json


python ./labelme2coco/labelme2coco.py \
  --input_dir {input_folder} \
  --output_dir {output_folder} \
  --labels labelme2coco/labels.txt \
  --image_format jpeg
```

`--image_format jpeg` (the default) writes `JPEGImages/*.jpg`;
`--image_format png` writes `PNGImages/*.png`, which is lossless and so keeps
PNG sources pixel-for-pixel, at roughly 10x the disk space. The annotations
file is `annotations.json` either way, and `Visualization/` is always JPEG.

#### Example
```
python ./labelme2coco/labelme2coco.py \
  --input_dir ./_data/screw/labelme/train \
  --output_dir ./_data/screw/coco/train_v1 \
  --labels labelme2coco/labels.txt
```
