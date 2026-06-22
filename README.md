# RSCD_ReMamba

Official implementation package for **ReMamba: Reliability-Enhanced Mamba Change Detection**.

This release contains the source code and 200-epoch pretrained checkpoints needed for reproduction. Experiment logs, prediction results, visualization files, shell scripts, paper drafts and cache files are not included.

## Contents

```text
./
|-- model/                         # ReMambaNet and core modules
|-- checkpoints_remamba/           # pretrained checkpoints for six datasets
|   |-- BCDD/200/netCD_epoch_best.pth
|   |-- CLCD/200/netCD_epoch_best.pth
|   |-- LEV/200/netCD_epoch_best.pth
|   |-- DSIFN/200/netCD_epoch_best.pth
|   |-- GoogleBuilding/200/netCD_epoch_best.pth
|   `-- SECOND/200/netCD_epoch_best.pth
|-- run.py                         # training entry
|-- predict.py                     # evaluation / inference entry
|-- data_utils.py
|-- losses.py
|-- measure.py
`-- requirements.txt
```

## Environment

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version if the default `pip` package is not suitable for your machine.

## Dataset Layout

Datasets are not included. Put each dataset under one root directory with this layout:

```text
DATA_ROOT/
`-- BCDD/
    |-- train/A/
    |-- train/B/
    |-- train/label/
    |-- val/A/
    |-- val/B/
    |-- val/label/
    |-- test/A/
    |-- test/B/
    `-- test/label/
```

The supported dataset names are:

```text
BCDD CLCD LEV DSIFN GoogleBuilding SECOND
```

## Evaluate Pretrained Checkpoints

Single dataset:

```bash
python predict.py \
  --dataset BCDD \
  --gpu_id 0 \
  --save_dir ./checkpoints_remamba \
  --data_root /path/to/DATA_ROOT
```

By default, `predict.py` only prints metrics. To save binary prediction masks and `metrics.csv`, add `--save_outputs`.

You can also pass explicit test folders:

```bash
python predict.py \
  --dataset BCDD \
  --gpu_id 0 \
  --save_dir ./checkpoints_remamba \
  --hr1_dir /path/to/DATA_ROOT/BCDD/test/A \
  --hr2_dir /path/to/DATA_ROOT/BCDD/test/B \
  --label_dir /path/to/DATA_ROOT/BCDD/test/label
```

## Train

Single dataset:

```bash
python run.py \
  --dataset BCDD \
  --gpu_id 0 \
  --epochs 200 \
  --batch_size 4 \
  --save_dir ./checkpoints_remamba \
  --hr1_train /path/to/DATA_ROOT \
  --hr2_train /path/to/DATA_ROOT \
  --lab_train /path/to/DATA_ROOT \
  --hr1_val /path/to/DATA_ROOT \
  --hr2_val /path/to/DATA_ROOT \
  --lab_val /path/to/DATA_ROOT
```

Training and inference use `--scan_backend fast` by default. `--scan_backend torch` is a direct recurrent fallback and is much slower on large 512x512 crops.

## Model Scale

```bash
python measure.py \
  --gpu_id 0 \
  --checkpoint ./checkpoints_remamba/BCDD/200/netCD_epoch_best.pth \
  --image_size 512 \
  --batch_size 1
```
