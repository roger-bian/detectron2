### Folder structure
```
_data/
└── {dataset_name}/
    ├── labels.txt      ## <---- label list, "__ignore__" first
    ├── labelme/        ## <---- what you edit: images + labelme JSON
    │   ├── train/
    │   ├── val/
    │   └── test/       ## optional
    └── coco/           ## <---- built by prepare_dataset.py, never edited by hand
        ├── train_v1/
        │   ├── JPEGImages/
        │   ├── Visualization/
        │   ├── annotations.json
        │   ├── manifest.json
        │   └── labels.txt
        ├── val_v1/
        └── test_v1/
```

Start a new dataset by copying `_{dataset_name}_example/`, which has
`labels.txt` and empty `labelme/{train,val,test}/` folders.

### Build a version
```
python prepare_dataset.py {dataset_name}
```

Checks every split, then writes the next version (`train_vN`, `val_vN`,
`test_vN`) under `coco/`. See "Prepare a dataset" in the top-level README.

### Usage in training
In the `train:` section of `settings.yaml`:
```
dataset_name: {dataset_name}
version: v{N}
```
