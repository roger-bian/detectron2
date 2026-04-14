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
#   - data_dataset_coco/JPEGImages
#   - data_dataset_coco/Visualization
#   - data_dataset_coco/annotations.json


python ./labelme2coco/labelme2coco.py \
  {input_folder} \
  {output_folder} \
  --labels labelme2coco/labels.txt
```

#### Example
```
python ./labelme2coco/labelme2coco.py \
  ./_data/screw/labelme/train \
  ./_data/screw/coco/train_v1 \
  --labels labelme2coco/labels.txt
```
