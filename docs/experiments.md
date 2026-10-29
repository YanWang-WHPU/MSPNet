# Training variants

The default entry point `python train.py` uses the complete MSPNet configuration
in `configs/mspnet.json`. The shared backbone is in `models/mspnet.py`; the alias
`HSIUNet` and parameter names are retained for historical checkpoints.

For explicit ablations, use `scripts/train_hsi_unet.py`. Its defaults are
intentionally generic, so specify the common experiment settings:

```bash
python scripts/train_hsi_unet.py --data-root data/hfz3 --split-dir data/hfz3/splits_spatial_uavtrain --label-mode triclass --out-dir outputs/unet --epochs 100 --batch-size 8 --augment-factor 2 --class-weighting sqrt_inverse --dice-weight 0.5 --seed 2026
```

Append the following switches to that command, choosing a distinct output
directory for every run:

| Variant | Additional options |
| --- | --- |
| HSI-only U-Net | None |
| Fixed field prototypes | `--prototype-input --ground-prototype-ratio 1.0` |
| Adaptive prototypes (PASP) | `--prototype-input --ground-prototype-ratio 0.2` |
| Hard UAV boundary guidance | `--boundary-weight 0.05 --teacher-boundary-dir teacher_boundaries` |
| Soft UAV boundary guidance (CABG) | `--boundary-weight 0.05 --teacher-boundary-dir teacher_boundaries_soft` |
| Complete MSPNet | Combine adaptive prototype and soft boundary switches |

Use `--prototype-json PATH` to override the default
`DATA_ROOT/threeclass_prototypes.json`. Boundary variants also need the
`teacher_valid/` masks. Combining prototype and boundary flags produces the
corresponding joint ablations. The wrapper `train.py` enables PASP by default;
use the explicit training script for HSI-only ablations.

All variants should share the same split manifests and seed. Normalization and
class weights are estimated on the training data. The optimizer is AdamW with
weight decay `1e-4`; the scheduler is cosine annealing over the requested number
of epochs. The default criterion combines class-weighted CE with foreground
Dice. Advanced legacy options remain available through `--help`; they are not
enabled by the default release configuration.

Reported metrics are fractions. `mIoU` and `mF1` average over background, rice,
and lotus; `fg_mIoU` averages only rice and lotus. Confusion matrices use target
classes as rows and predicted classes as columns. Classes absent from both
prediction and target have score zero under the original metric convention.

Seeding preserves the original training behavior, including cuDNN benchmarking;
results may vary between hardware and software environments. The release checks
use synthetic data and do not rerun the full 100-epoch research experiments.
