from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_hsi_unet import (
    SegmentationLoss,
    class_weights,
    compute_stats,
    confusion_matrix,
    label_path,
    metrics,
    read_rows,
    seed_everything,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]


class MultisourceDataset(Dataset):
    def __init__(self, csv_path: Path, mean: np.ndarray, std: np.ndarray, train: bool, augment_factor: int):
        self.rows = read_rows(csv_path)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.train = train
        self.augment_factor = augment_factor if train else 1

    def __len__(self) -> int:
        return len(self.rows) * self.augment_factor

    def __getitem__(self, index: int):
        aug_id, row_id = divmod(index, len(self.rows))
        row = self.rows[row_id]
        stem = Path(row["image"]).stem
        image = np.load(row["image"])["image"].astype(np.float32)
        mask = np.asarray(Image.open(label_path(row, "mask_tri")), dtype=np.int64)
        teacher_path = Path(row["image"]).parent.parent / "teacher_triclass" / f"{stem}.png"
        teacher = np.asarray(Image.open(teacher_path), dtype=np.int64)
        image = (image - self.mean) / self.std

        if self.train:
            if aug_id % 4 == 1:
                image, mask, teacher = image[:, :, ::-1], mask[:, ::-1], teacher[:, ::-1]
            elif aug_id % 4 == 2:
                image, mask, teacher = image[:, ::-1, :], mask[::-1, :], teacher[::-1, :]
            elif aug_id % 4 == 3:
                image = np.rot90(image, 1, axes=(1, 2))
                mask = np.rot90(mask, 1)
                teacher = np.rot90(teacher, 1)
            k = (aug_id // 4) % 4
            if k:
                image = np.rot90(image, k, axes=(1, 2))
                mask = np.rot90(mask, k)
                teacher = np.rot90(teacher, k)

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        teacher = np.ascontiguousarray(np.eye(3, dtype=np.float32)[teacher].transpose(2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(teacher), torch.from_numpy(mask), stem


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SingleInputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class SegFormerWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        logits = self.decoder(features)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class CMXWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image, teacher)
        logits = self.decoder(features)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def decoder_head() -> type[nn.Module]:
    module = load_module(
        "cmx_mlp_decoder_standalone", ROOT / "baselines" / "CMX" / "models" / "decoders" / "MLPDecoder.py"
    )
    return module.DecoderHead


def replace_projection_input(projection: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        projection.out_channels,
        projection.kernel_size,
        projection.stride,
        projection.padding,
        bias=projection.bias is not None,
    )


def build_segformer() -> nn.Module:
    registry = types.SimpleNamespace(register_module=lambda *args, **kwargs: (lambda cls: cls))
    mmseg = types.ModuleType("mmseg")
    mmseg_models = types.ModuleType("mmseg.models")
    mmseg_builder = types.ModuleType("mmseg.models.builder")
    mmseg_builder.BACKBONES = registry
    mmseg_utils = types.ModuleType("mmseg.utils")
    mmseg_utils.get_root_logger = lambda: None
    mmcv = types.ModuleType("mmcv")
    mmcv_runner = types.ModuleType("mmcv.runner")
    mmcv_runner.load_checkpoint = lambda *args, **kwargs: None
    sys.modules.update(
        {
            "mmseg": mmseg,
            "mmseg.models": mmseg_models,
            "mmseg.models.builder": mmseg_builder,
            "mmseg.utils": mmseg_utils,
            "mmcv": mmcv,
            "mmcv.runner": mmcv_runner,
        }
    )
    module = load_module(
        "official_segformer_b0",
        ROOT / "baselines" / "SegFormer" / "mmseg" / "models" / "backbones" / "mix_transformer.py",
    )
    encoder = module.mit_b0()
    encoder.patch_embed1.proj = replace_projection_input(encoder.patch_embed1.proj, 32)
    decoder = decoder_head()(in_channels=[32, 64, 160, 256], num_classes=3, embed_dim=128)
    return SegFormerWrapper(encoder, decoder)


def build_deeplab() -> nn.Module:
    source = ROOT / "baselines" / "DeepLabV3Plus-Pytorch"
    sys.path.insert(0, str(source))
    from network import modeling

    model = modeling.deeplabv3plus_mobilenet(num_classes=3, output_stride=16, pretrained_backbone=False)
    first = model.backbone.low_level_features[0][0]
    model.backbone.low_level_features[0][0] = nn.Conv2d(
        32, first.out_channels, first.kernel_size, first.stride, first.padding, bias=False
    )
    return SingleInputWrapper(model)


def build_um2former() -> nn.Module:
    module = load_module(
        "official_um2former",
        ROOT / "baselines" / "UM2Former" / "UM2Former" / "Nets" / "UM2former.py",
    )
    return SingleInputWrapper(module.UM2Former(num_classes=3, in_channel=32))


def build_cmx() -> nn.Module:
    source = ROOT / "baselines" / "CMX"
    sys.path.insert(0, str(source))
    from models.encoders.dual_segformer import mit_b0

    encoder = mit_b0()
    encoder.patch_embed1.proj = replace_projection_input(encoder.patch_embed1.proj, 32)
    encoder.extra_patch_embed1.proj = replace_projection_input(encoder.extra_patch_embed1.proj, 3)
    decoder = decoder_head()(in_channels=[32, 64, 160, 256], num_classes=3, embed_dim=128)
    return CMXWrapper(encoder, decoder)


def build_model(name: str) -> nn.Module:
    return {
        "deeplabv3plus": build_deeplab,
        "segformer_b0": build_segformer,
        "um2former": build_um2former,
        "cmx_b0": build_cmx,
    }[name]()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    cm = np.zeros((3, 3), dtype=np.int64)
    total_loss = 0.0
    total = 0
    for image, teacher, target, _ in loader:
        image, teacher, target = image.to(device), teacher.to(device), target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(image, teacher)
            loss = criterion(logits, target)
        if training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        prediction = logits.argmax(1).detach().cpu().numpy()
        cm += confusion_matrix(prediction, target.detach().cpu().numpy(), num_classes=3)
        total_loss += float(loss.item()) * image.shape[0]
        total += image.shape[0]
    return total_loss / max(total, 1), metrics(cm), cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train official open-source segmentation baselines on OHS/UAV patches.")
    parser.add_argument("--model", required=True, choices=["deeplabv3plus", "segformer_b0", "um2former", "cmx_b0"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(split_dir / "train.csv")
    mean, std = compute_stats(train_rows)
    train_ds = MultisourceDataset(split_dir / "train.csv", mean, std, True, args.augment_factor)
    val_ds = MultisourceDataset(split_dir / "val.csv", mean, std, False, 1)
    test_ds = MultisourceDataset(split_dir / "test.csv", mean, std, False, 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    weights = class_weights(train_rows, "mask_tri", 3, "sqrt_inverse").to(device)
    criterion = SegmentationLoss(weights, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best = -1.0
    log = []
    checkpoint = out_dir / "best_model.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_metrics, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        log.append(row)
        if val_metrics["fg_mIoU"] > best:
            best = val_metrics["fg_mIoU"]
            torch.save(model.state_dict(), checkpoint)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, test_metrics, test_cm = run_epoch(model, test_loader, criterion, device)
    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "best_val_fg_mIoU": best,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_confusion": test_cm.tolist(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_sec": time.time() - started,
    }
    write_csv(out_dir / "training_log.csv", log)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
