from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_hsi_unet import (
    HSIUNet,
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


class HFZ3Dataset(Dataset):
    def __init__(self, csv_path: Path, mean: np.ndarray, std: np.ndarray, train: bool, augment_factor: int = 1):
        self.rows = read_rows(csv_path)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.train = train
        self.augment_factor = max(1, augment_factor) if train else 1
        self.images = [np.load(row["image"])["image"] for row in self.rows]
        self.targets = [
            np.asarray(Image.open(label_path(row, "mask_tri")), dtype=np.uint8) for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows) * self.augment_factor

    def __getitem__(self, index: int):
        copy_id, row_id = divmod(index, len(self.rows))
        row = self.rows[row_id]
        image = self.images[row_id].astype(np.float32)
        target = self.targets[row_id].astype(np.int64)
        image = (image - self.mean) / self.std

        if self.train and copy_id > 0:
            k = random.randrange(4)
            if k:
                image = np.rot90(image, k, axes=(1, 2))
                target = np.rot90(target, k)
            if random.random() < 0.5:
                image = image[:, :, ::-1]
                target = target[:, ::-1]
            if random.random() < 0.5:
                image = image[:, ::-1, :]
                target = target[::-1, :]

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(target)),
            Path(row["image"]).stem,
        )


class LegacySingleInputAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image, None)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int = 128):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for size in (1, 2, 3, 6)
            ]
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        size = feature.shape[-2:]
        pooled = [feature]
        for stage in self.stages:
            pooled.append(F.interpolate(stage(feature), size=size, mode="bilinear", align_corners=False))
        return torch.cat(pooled, dim=1)


class PSPNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet34", pretrained=False, in_chans=32, features_only=True, out_indices=(4,)
        )
        encoder_channels = self.encoder.feature_info.channels()[0]
        branch_channels = 128
        self.ppm = PyramidPooling(encoder_channels, branch_channels)
        self.head = nn.Sequential(
            ConvBNReLU(encoder_channels + 4 * branch_channels, 256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        size = image.shape[-2:]
        feature = self.encoder(image)[0]
        logits = self.head(self.ppm(feature))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def build_unetformer() -> nn.Module:
    source = ROOT / "baselines" / "UM2Former" / "UM2Former"
    sys.path.insert(0, str(source))
    from UnetFormer.UnetFormer import UNetFormer

    return UNetFormer(num_classes=3)


def build_model(name: str) -> nn.Module:
    if name in {"deeplabv3plus", "segformer_b0"}:
        from train_open_source_baselines import build_deeplab, build_segformer
        return LegacySingleInputAdapter({"deeplabv3plus": build_deeplab, "segformer_b0": build_segformer}[name]())
    builders = {
        "unet": lambda: HSIUNet(in_ch=32, num_classes=3, base=24),
        "pspnet": PSPNet,
        "unetformer": build_unetformer,
    }
    return builders[name]()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, prediction_dir: Path | None = None):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((3, 3), dtype=np.int64)
    total_loss = 0.0
    total = 0
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for image, target, stems in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(image)
            loss = criterion(logits, target)
        if training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        prediction = logits.argmax(dim=1)
        confusion += confusion_matrix(prediction.detach().cpu().numpy(), target.detach().cpu().numpy(), 3)
        total_loss += float(loss.item()) * image.shape[0]
        total += image.shape[0]
        if prediction_dir is not None:
            for i, stem in enumerate(stems):
                Image.fromarray(prediction[i].cpu().numpy().astype(np.uint8)).save(prediction_dir / f"{stem}.png")
    return total_loss / max(total, 1), metrics(confusion), confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HFZ3-only fully supervised segmentation baselines.")
    parser.add_argument(
        "--model", required=True, choices=["unet", "deeplabv3plus", "segformer_b0", "pspnet", "unetformer"]
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(split_dir / "train.csv")
    mean, std = compute_stats(train_rows)
    (out_dir / "norm_stats.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2), encoding="utf-8"
    )

    train_ds = HFZ3Dataset(split_dir / "train.csv", mean, std, True, args.augment_factor)
    val_ds = HFZ3Dataset(split_dir / "val.csv", mean, std, False)
    test_ds = HFZ3Dataset(split_dir / "test.csv", mean, std, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    weights = class_weights(train_rows, "mask_tri", 3, "sqrt_inverse").to(device)
    criterion = SegmentationLoss(weights, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best = -1.0
    checkpoint = out_dir / "best_model.pt"
    log_rows: list[dict] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_metrics, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        log_rows.append(row)
        if val_metrics["mIoU"] > best:
            best = val_metrics["mIoU"]
            torch.save(model.state_dict(), checkpoint)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, test_metrics, test_confusion = run_epoch(
        model, test_loader, criterion, device, prediction_dir=out_dir / "test_predictions"
    )
    summary = {
        "model": args.model,
        "data_source": "HFZ3 32-band image and merge-12-23.shp mask only",
        "split_dir": str(split_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "augment_factor": args.augment_factor,
        "class_weighting": "sqrt_inverse",
        "dice_weight": 0.5,
        "selection_metric": "val_mIoU",
        "best_val_mIoU": best,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_confusion": test_confusion.tolist(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_sec": time.time() - started,
        "seed": args.seed,
    }
    write_csv(out_dir / "training_log.csv", log_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
