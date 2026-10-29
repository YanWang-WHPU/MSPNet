# Optional comparison methods

Only project-specific adapters and one local padding patch are tracked here.
Upstream repositories, their data, pretrained weights, and generated outputs
are excluded. Upstream code retains its own licensing terms.

Install optional Python dependencies and fetch the pinned source revisions:

```bash
python -m pip install -r requirements-baselines.txt
python scripts/download_baselines.py
```

The downloader does not overwrite existing directories. It fetches source only
and checks out the revisions recorded in the local experiment checkouts:

| Directory | Upstream | Revision |
| --- | --- | --- |
| `DeepLabV3Plus-Pytorch` | [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) | `4e1087de98bc49d55b9239ae92810ef7368660db` |
| `SegFormer` | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | `65fa8cfa9b52b6ee7e8897a98705abf8570f9e32` |
| `CMX` | [huaaaliu/RGBX_Semantic_Segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) | `e251d860aebc2f583a6c4919877e6bebe7f1aff3` |
| `UM2Former` | [ZhaohuiXue/UM2Former](https://github.com/ZhaohuiXue/UM2Former) | `c4611a354fac1dde5c721f38e08f1457ddf65fc9` |

The SegFormer adapter also loads the CMX MLP decoder. The UNetFormer adapter
loads the implementation under UM2Former. The retained patch makes attention
padding fall back to replication when reflection padding is invalid for a small
feature map. The downloader applies it after checking out UM2Former.

Train the HSI-only comparison methods with:

```bash
python scripts/train_hfz3_supervised_baselines.py --model unet --data-root data/hfz3 --split-dir data/hfz3/splits_spatial_uavtrain --out-dir outputs/baseline_unet --epochs 100 --batch-size 8 --augment-factor 2 --seed 2026
```

Supported names are `unet`, `deeplabv3plus`, `pspnet`, `segformer_b0`, and
`unetformer`. U-Net uses the shared project backbone; PSPNet uses a timm encoder.
Keep the same splits for all comparisons. Check upstream installation guidance
if your torchvision/timm versions do not support a historical implementation.
The full third-party model training runs were not repeated for this release.
Forward passes for all five HSI-only model adapters were checked with temporary
synthetic `(1, 32, 128, 128)` inputs against the local upstream checkouts.

`scripts/train_open_source_baselines.py` additionally preserves legacy paired
HSI/teacher-label experiments (`um2former`, `cmx_b0` included). Those experiments
need `teacher_triclass/` labels for every split and use a different input protocol
from MSPNet's HSI-only deployment. They are not the default comparison command.

Baseline checkpoints contain a plain state dictionary, unlike the complete
MSPNet training checkpoint. To evaluate the external baseline checkpoints, use
`scripts/evaluate_hfz3_supervised_baseline_checkpoint.py` with `--model`,
`--split-dir`, `--checkpoint`, `--norm-stats`, and `--out-dir`.
