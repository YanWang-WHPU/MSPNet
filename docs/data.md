# Local data contract

No research data, split records, measured prototype values, or trained weights
are included. The paths and structures below describe inputs to create locally.

## Hyperspectral patches and segmentation labels

Each image is an NPZ file containing an `image` array in band-first order
`(32, 128, 128)`. Keep the training band order and radiometric scale consistent
for all splits and inference. Training casts images to float32 and estimates
per-band normalization statistics from the first 200 training records at most,
using pixels with at least one nonzero band. The original variance floor is 1.0.
Do not pre-normalize patches before passing them to the released pipeline.

Masks are single-channel uint8 PNGs with these values:

| Value | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Rice |
| 2 | Lotus |
| 255 | Ignored segmentation pixel |

The model requires spatial dimensions divisible by 16 during training. The
released configuration uses 128-pixel patches. `predict.py` pads other sizes.

## Split manifests

`train.csv`, `val.csv`, and `test.csv` have at least the columns `image,mask_tri`.
Files use UTF-8 (a BOM is accepted). Paths may be absolute or relative to the
directory containing the CSV. For example, a record in
`data/hfz3/splits_spatial_uavtrain/train.csv` can refer to
`../images/patch.npz` and `../masks_triclass/patch.png`.

Spatial splitting and mosaic evaluation also use:

| Column | Meaning |
| --- | --- |
| `row`, `col` | Top-left pixel offsets on the full HSI grid |
| `background_pixels`, `rice_pixels`, `lotus_pixels` | Per-patch label counts |
| `scene_id` | Optional source identifier retained in generated records |

To generate balanced 512-pixel block splits, provide these columns in
`data/hfz3/splits/all.csv` and run:

```bash
python scripts/create_balanced_block_splits.py --data-root data/hfz3 --out-name splits_spatial_uavtrain --block-size 512 --seed 2026 --force-train-valid-dir data/hfz3/teacher_valid
```

The intended input is one raster with aligned, nonoverlapping 128-pixel patches.
Blocks containing valid UAV supervision are assigned to training; validation
and test contain separate blocks. This command searches for balanced partitions
using label counts. It does not ship or recover the original experimental split.
For an exact comparison, retain the original manifests and their order. The
splitter is not a general overlap remover for arbitrary sliding-window patches
or multiple independent scenes.

## Field prototypes for PASP

Create `threeclass_prototypes.json` locally. It must contain an object
`prototypes` with keys `"1"` (rice) and `"2"` (lotus). Each value is a finite,
nonconstant numeric vector of length 32, aligned with the HSI bands. These are
input spectral values, not similarity maps. The code centers each vector and
L2-normalizes it before blending with training-image prototypes.

Conceptual structure (the placeholders below are not executable JSON):

```text
{
  "prototypes": {
    "1": [32 rice spectral values],
    "2": [32 lotus spectral values]
  }
}
```

Prepare these vectors by resampling your field measurements to the HSI bands.
No measured spectra or calibrated vectors are included in this repository.
With `ground_prototype_ratio=0.2`, PASP uses 20% field prototype and 80%
training-image class centroid, followed by L2 normalization. Centroids are
computed from centered, normalized pixel spectra in the training split only.

## UAV targets for CABG

For each training image `images/NAME.npz`, supply:

- `teacher_boundaries_soft/NAME.png`: a single-channel uint8 target; divide by
  255 to obtain values in `[0,1]`.
- `teacher_valid/NAME.png`: a binary uint8 mask, 255 for valid UAV coverage and
  0 outside coverage. Create an all-zero mask and target for uncovered patches.

Targets must be registered to the same HSI patch grid. Validation and test do
not require UAV files. Coverage masks describe valid observations, including
valid background; they must not be inferred simply from foreground labels.

If binary boundary targets already exist in `teacher_boundaries/`, create the
soft targets with:

```bash
python scripts/prepare_soft_uav_boundaries.py --data-root data/hfz3 --sigma 1.5
```

This applies `exp(-0.5 * (distance / sigma)^2)` and masks it by UAV coverage.
The auxiliary loss combines coverage-masked BCE and soft Dice. Uncovered pixels
contribute zero auxiliary gradient.

## Optional ArcGIS preprocessing

These two scripts require `arcpy` from an ArcGIS Pro environment. It is not a
pip dependency of MSPNet. They are preserved adapters for the local research
workflow, and were not runtime-tested as part of the release checks.

```bash
python scripts/prepare_hfz3_fullsupervised.py --hsi /path/to/hsi.tif --label /path/to/labels.shp --label-field gridcode_1 --out-root data/hfz3
python scripts/prepare_hfz3_auxiliary_targets.py --data-root data/hfz3 --teacher-work /path/to/aligned_uav_labels --prototype-json /path/to/threeclass_prototypes.json
python scripts/prepare_soft_uav_boundaries.py --data-root data/hfz3
```

The first command requires a new or empty output directory. Source polygon
values 1/2 map to rice, 3 to lotus, and all other values to background.
The second command expects `*_teacher_ohs.tif` rasters already projected,
resampled with nearest-neighbor interpolation, and aligned to the HSI grid.
Their values are 0 background, 1/2 rice, 3 lotus, and NoData/255 outside valid
coverage. Check CRS, pixel grid, and NoData before running the adapter. It writes
binary boundaries, validity masks, and teacher labels, and copies the supplied
prototype JSON into the dataset directory.

The public release begins with prepared HSI/UAV labels and field vectors; it
does not include the separate UAV teacher training pipeline, raw spectral
instrument conversion, or the underlying annotations.
