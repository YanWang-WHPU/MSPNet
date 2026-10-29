# MSPNet

**Multi-Source Privileged Learning for Hyperspectral Rice Field Segmentation**

MSPNet segments a 32-band hyperspectral image into background, rice, and lotus.
Training uses two additional sources of information:

- **Prototype-Adaptive Spectral Prior (PASP):** field spectral prototypes are calibrated with class centroids computed from the training HSI. Two similarity maps are concatenated with the 32 HSI bands.
- **Coverage-Aware Boundary Guidance (CABG):** an auxiliary head learns soft UAV-derived boundaries only where valid UAV coverage is available.

Inference needs only an HSI patch. The calibrated prototypes and normalization
statistics are stored in the checkpoint; field spectra and UAV images are not
required at deployment.

This repository contains source code and configuration only. Datasets, measured
spectra, UAV targets, trained weights, experiment outputs, and manuscript files
are not distributed. Publication details and citation information will be added
by the authors.

## Installation

Use Python 3.10 or newer (Python 3.12 is suitable). Install a PyTorch build
appropriate for your hardware, then the remaining dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Core release checks were run with Python 3.10, PyTorch 2.8.0, NumPy 2.2.6,
SciPy 1.15.3, and Pillow 12.2.0. The requirement ranges are not an exhaustive
compatibility matrix. Optional raster preprocessing scripts require a licensed
ArcGIS Pro environment providing `arcpy`; training and prediction do not.

## Data preparation

See [docs/data.md](docs/data.md) for array shapes, label values, split manifests,
UAV coverage masks, and the prototype JSON format. Supply your own data locally.
The `data/`, `outputs/`, and `results/` directories are ignored by Git.

Prepared data should use this layout:

```text
data/hfz3/
  images/                         # NPZ: image with shape (32, 128, 128)
  masks_triclass/                  # PNG: 0 background, 1 rice, 2 lotus; 255 ignore
  teacher_boundaries_soft/         # PNG: soft targets encoded as 0..255
  teacher_valid/                   # PNG: 0 outside coverage, 255 inside
  threeclass_prototypes.json       # two 32-band field prototype vectors
  splits_spatial_uavtrain/
    train.csv
    val.csv
    test.csv
```

## Training

Run commands from the repository root:

```bash
python train.py
```

[configs/mspnet.json](configs/mspnet.json) supplies the released 128-pixel
configuration: base width 24, 100 epochs, batch size 8, AdamW with learning rate
0.001, augmentation factor 2, square-root inverse class weighting, foreground
Dice weight 0.5, prototype ratio 0.2, and boundary weight 0.05. Only the training
split is used for normalization, class weights, and HSI prototype estimation.
The best validation mIoU selects the checkpoint; the test split is evaluated
after selection.

CLI options override JSON values. For example:

```bash
python train.py --data-root /path/to/hfz3 --split-dir /path/to/hfz3/splits_spatial_uavtrain --out-dir outputs/mspnet --epochs 100 --batch-size 8
python train.py --config configs/mspnet.json --lr 0.0005
```

Training writes `best_model.pt`, `norm_stats.json`, `training_log.csv`,
`summary.json`, and test prediction masks to the output directory.
Training, prediction, and MSPNet checkpoint evaluation accept `--device cpu`
or `--device cuda`; the default selects CUDA when available.

## HSI-only prediction

```bash
python predict.py --checkpoint outputs/mspnet/best_model.pt --image /path/to/patch.npz --output outputs/prediction.png
```

The NPZ must contain the **unnormalized** `image` array in `(32, H, W)` order,
using the same band order and radiometric scale as training. The command loads
normalization statistics and prototypes from the checkpoint and writes a PNG
of class IDs. Nonmultiples of 16 are padded and cropped back to the original
shape. This is patch inference; large-area mosaicking is not provided here.
Load only trusted checkpoints, since the historical training checkpoint format
includes Python/NumPy objects.

To evaluate a saved checkpoint on the test manifest and compute boundary scores:

```bash
python scripts/evaluate_hfz3_checkpoint.py --data-root data/hfz3 --split-dir data/hfz3/splits_spatial_uavtrain --checkpoint outputs/mspnet/best_model.pt --out-dir outputs/evaluation
```

This evaluator additionally needs `metadata.json` with
`"raster_shape": [32, full_height, full_width]` and global `row`/`col` offsets
in the manifest. Metrics use fractions in `[0, 1]`; mF1 is the unweighted mean
of the three class F1 scores.

## Ablations and baselines

See [docs/experiments.md](docs/experiments.md) for PASP/CABG switches and
[baselines/README.md](baselines/README.md) for optional third-party models,
pinned source revisions, and the retained UNetFormer padding patch.

## Code layout

```text
models/mspnet.py                   # shared model, checkpoint-compatible names
train.py                          # configured MSPNet training entry point
predict.py                        # HSI-only patch prediction
configs/mspnet.json                # default training options; no measured data
scripts/train_hsi_unet.py          # original training pipeline and ablations
scripts/prepare_*.py               # optional local preprocessing
scripts/create_balanced_block_splits.py
scripts/train_*baselines.py        # baseline adapters and training
scripts/evaluate_*checkpoint.py    # checkpoint evaluation
scripts/download_baselines.py      # fetch upstream source; no bundled datasets
tests/test_mspnet.py               # synthetic checks; no research data needed
```

## Validation

```bash
python -m unittest discover -s tests -v
```

The checks cover model outputs, HSI-only checkpoint prediction, zero auxiliary
gradients outside UAV coverage, portable manifest paths, and a complete one-epoch
training/evaluation run on temporary synthetic inputs. They validate the code
pipeline, not the published accuracy. The original dataset and weights are
required to reproduce scientific results.
