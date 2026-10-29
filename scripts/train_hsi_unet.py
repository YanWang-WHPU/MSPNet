from __future__ import annotations

import argparse
import csv
import json
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "hfz3"
sys.path.insert(0, str(ROOT))
from models import HSIUNet
from data_paths import read_rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def label_path(row: dict, label_key: str) -> Path:
    if row.get(label_key):
        return Path(row[label_key])
    if label_key == "mask_multi_spectral":
        image_path = Path(row["image"])
        return image_path.parent.parent / "masks_multiclass_spectral" / f"{image_path.stem}.png"
    if label_key == "mask_tri":
        image_path = Path(row["image"])
        return image_path.parent.parent / "masks_triclass" / f"{image_path.stem}.png"
    raise KeyError(label_key)


def teacher_boundary_path(row: dict, directory_name: str = "teacher_boundaries") -> Path:
    image_path = Path(row["image"])
    return image_path.parent.parent / directory_name / f"{image_path.stem}.png"


def teacher_valid_path(row: dict, directory_name: str = "teacher_valid") -> Path:
    image_path = Path(row["image"])
    return image_path.parent.parent / directory_name / f"{image_path.stem}.png"


def teacher_semantic_path(row: dict) -> Path:
    image_path = Path(row["image"])
    return image_path.parent.parent / "teacher_triclass" / f"{image_path.stem}.png"


def compute_stats(rows: list[dict], max_items: int = 200) -> tuple[np.ndarray, np.ndarray]:
    sums = None
    sq_sums = None
    count = 0
    for row in rows[:max_items]:
        arr = np.load(row["image"])["image"].astype(np.float32)
        valid = np.any(arr != 0, axis=0)
        flat = arr[:, valid]
        if flat.size == 0:
            continue
        if sums is None:
            sums = flat.sum(axis=1)
            sq_sums = (flat * flat).sum(axis=1)
        else:
            sums += flat.sum(axis=1)
            sq_sums += (flat * flat).sum(axis=1)
        count += flat.shape[1]
    if sums is None or count == 0:
        raise RuntimeError("No valid pixels available for normalization statistics.")
    mean = sums / count
    var = np.maximum(sq_sums / count - mean * mean, 1.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


class HSIRiceDataset(Dataset):
    def __init__(self, csv_path: Path, label_mode: str, mean: np.ndarray, std: np.ndarray, train: bool,
                 augment_factor: int = 1, use_teacher_boundary: bool = False,
                 teacher_boundary_dir: str = "teacher_boundaries", use_teacher_semantic: bool = False,
                 teacher_valid_dir: str = "teacher_valid"):
        self.rows = read_rows(csv_path)
        self.label_key = {
            "weak": "mask_binary",
            "spectral": "mask_spectral",
            "multiclass": "mask_multi",
            "multiclass_spectral": "mask_multi_spectral",
            "triclass": "mask_tri",
            "triclass_spectral": "mask_tri",
        }[label_mode]
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.train = train
        self.augment_factor = max(1, augment_factor) if train else 1
        self.use_teacher_boundary = use_teacher_boundary
        self.use_teacher_semantic = use_teacher_semantic
        self.teacher_boundary_dir = teacher_boundary_dir
        self.teacher_valid_dir = teacher_valid_dir
        self.images = [np.load(row["image"])["image"] for row in self.rows]
        self.masks = [
            np.asarray(Image.open(label_path(row, self.label_key)), dtype=np.uint8) for row in self.rows
        ]
        if self.use_teacher_boundary:
            self.boundaries = [
                np.asarray(
                    Image.open(teacher_boundary_path(row, self.teacher_boundary_dir)), dtype=np.uint8
                )
                for row in self.rows
            ]
        else:
            self.boundaries = None
        if self.use_teacher_boundary or self.use_teacher_semantic:
            self.boundary_valid_masks = [
                np.asarray(Image.open(teacher_valid_path(row, self.teacher_valid_dir)), dtype=np.uint8)
                if teacher_valid_path(row, self.teacher_valid_dir).exists()
                else np.full(self.masks[i].shape, 255, dtype=np.uint8)
                for i, row in enumerate(self.rows)
            ]
        else:
            self.boundary_valid_masks = None
        if self.use_teacher_semantic:
            self.teacher_semantics = [
                np.asarray(Image.open(teacher_semantic_path(row)), dtype=np.uint8) for row in self.rows
            ]
        else:
            self.teacher_semantics = None

    def __len__(self) -> int:
        return len(self.rows) * self.augment_factor

    def __getitem__(self, index: int):
        copy_id, row_id = divmod(index, len(self.rows))
        row = self.rows[row_id]
        image = self.images[row_id].astype(np.float32)
        mask = self.masks[row_id].astype(np.int64)
        if self.use_teacher_boundary:
            boundary = self.boundaries[row_id].astype(np.float32) / 255.0
        else:
            boundary = np.zeros(mask.shape, dtype=np.float32)
        if self.use_teacher_boundary or self.use_teacher_semantic:
            boundary_valid = self.boundary_valid_masks[row_id].astype(np.float32) / 255.0
        else:
            boundary_valid = np.zeros(mask.shape, dtype=np.float32)
        if self.use_teacher_semantic:
            teacher_semantic = self.teacher_semantics[row_id].astype(np.float32)
        else:
            teacher_semantic = np.zeros(mask.shape, dtype=np.float32)
        image = (image - self.mean) / self.std
        if self.train and copy_id > 0:
            k = random.randrange(4)
            if k:
                image = np.rot90(image, k, axes=(1, 2))
                mask = np.rot90(mask, k, axes=(0, 1))
                boundary = np.rot90(boundary, k, axes=(0, 1))
                boundary_valid = np.rot90(boundary_valid, k, axes=(0, 1))
                teacher_semantic = np.rot90(teacher_semantic, k, axes=(0, 1))
            if random.random() < 0.5:
                image = image[:, :, ::-1]
                mask = mask[:, ::-1]
                boundary = boundary[:, ::-1]
                boundary_valid = boundary_valid[:, ::-1]
                teacher_semantic = teacher_semantic[:, ::-1]
            if random.random() < 0.5:
                image = image[:, ::-1, :]
                mask = mask[::-1, :]
                boundary = boundary[::-1, :]
                boundary_valid = boundary_valid[::-1, :]
                teacher_semantic = teacher_semantic[::-1, :]
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        boundary = np.ascontiguousarray(boundary)
        boundary_valid = np.ascontiguousarray(boundary_valid)
        teacher_semantic = np.ascontiguousarray(teacher_semantic)
        boundary_targets = np.stack([boundary, boundary_valid, teacher_semantic], axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(boundary_targets), Path(row["image"]).stem


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 2, ignore_index: int = 255) -> np.ndarray:
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics(confusion: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pr = confusion.sum(axis=0).astype(np.float64)
    union = gt + pr - tp
    iou = tp / np.maximum(union, eps)
    recall = tp / np.maximum(gt, eps)
    precision = tp / np.maximum(pr, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    result = {
        "OA": float(tp.sum() / max(confusion.sum(), eps)),
        "mIoU": float(iou.mean()),
        "mF1": float(f1.mean()),
    }
    if len(iou) == 4:
        names = ["Background", "MidRice", "LateRice", "Lotus"]
    elif len(iou) == 3:
        names = ["Background", "Rice", "Lotus"]
    else:
        names = ["Background", "Rice"]
    for i, name in enumerate(names):
        result[f"{name}_IoU"] = float(iou[i])
        result[f"{name}_F1"] = float(f1[i])
        result[f"{name}_precision"] = float(precision[i])
        result[f"{name}_recall"] = float(recall[i])
    if len(iou) > 2:
        result["fg_mIoU"] = float(iou[1:].mean())
    return result


def class_weights(rows: list[dict], label_key: str, num_classes: int, scheme: str = "inverse") -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in rows:
        y = np.array(Image.open(label_path(row, label_key)), dtype=np.int64)
        valid = y != 255
        for cls in range(num_classes):
            counts[cls] += np.count_nonzero((y == cls) & valid)
    freq = counts / max(counts.sum(), 1.0)
    if scheme == "none":
        weights = np.ones_like(freq)
    elif scheme == "sqrt_inverse":
        weights = 1.0 / np.sqrt(np.maximum(freq, 1e-3))
    elif scheme == "power075":
        weights = 1.0 / np.power(np.maximum(freq, 1e-3), 0.75)
    else:
        weights = 1.0 / np.maximum(freq, 1e-3)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def image_spectral_prototypes(rows: list[dict], label_key: str, num_classes: int) -> np.ndarray:
    sums = np.zeros((num_classes - 1, 32), dtype=np.float64)
    counts = np.zeros(num_classes - 1, dtype=np.int64)
    for row in rows:
        image = np.load(row["image"])["image"].astype(np.float32)
        target = np.asarray(Image.open(label_path(row, label_key)), dtype=np.int64)
        image = image - image.mean(axis=0, keepdims=True)
        image = image / (np.sqrt(np.square(image).sum(axis=0, keepdims=True)) + 1e-6)
        for cls in range(1, num_classes):
            selected = target == cls
            if np.any(selected):
                sums[cls - 1] += image[:, selected].sum(axis=1)
                counts[cls - 1] += int(selected.sum())
    if np.any(counts == 0):
        raise RuntimeError(f"Cannot estimate image prototypes; class counts are {counts.tolist()}")
    prototypes = (sums / counts[:, None]).astype(np.float32)
    return prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-6)


def lovasz_gradient(foreground_sorted: torch.Tensor) -> torch.Tensor:
    count = len(foreground_sorted)
    total = foreground_sorted.sum()
    intersection = total - foreground_sorted.float().cumsum(0)
    union = total + (1.0 - foreground_sorted).float().cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-6)
    if count > 1:
        gradient[1:count] = gradient[1:count] - gradient[:-1].clone()
    return gradient


def lovasz_softmax_loss(probabilities: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, probabilities.shape[1])
    target = target.reshape(-1)
    valid = target != ignore_index
    probabilities = probabilities[valid]
    target = target[valid]
    losses = []
    for cls in range(probabilities.shape[1]):
        foreground = (target == cls).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, cls]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, lovasz_gradient(foreground_sorted)))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0


class SegmentationLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, dice_weight: float, ignore_index: int = 255,
                 focal_gamma: float = 0.0, lovasz_weight: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.lovasz_weight = lovasz_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.focal_gamma > 0:
            valid = target != self.ignore_index
            safe_target = target.masked_fill(~valid, 0)
            ce_map = F.cross_entropy(
                logits, target, weight=self.ce.weight, ignore_index=self.ignore_index, reduction="none"
            )
            probability = F.softmax(logits, dim=1).gather(1, safe_target[:, None]).squeeze(1)
            focal_map = ce_map * torch.pow(1.0 - probability, self.focal_gamma)
            if self.ce.weight is not None:
                normalizer = self.ce.weight[safe_target][valid].sum()
            else:
                normalizer = valid.sum()
            loss = focal_map[valid].sum() / normalizer.clamp_min(1.0)
        else:
            loss = self.ce(logits, target)
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_f = valid[:, None].float()
        probs = F.softmax(logits, dim=1) * valid_f
        one_hot = one_hot * valid_f
        # Foreground Dice excludes background.
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * (1.0 - dice.mean())
        if self.lovasz_weight > 0:
            loss = loss + self.lovasz_weight * lovasz_softmax_loss(
                F.softmax(logits, dim=1), target, self.ignore_index
            )
        return loss


def train_epoch(model, loader, optimizer, criterion, device, scaler, num_classes: int,
                proto: torch.Tensor | None = None, proto_weight: float = 0.0,
                mean: np.ndarray | None = None, std: np.ndarray | None = None,
                boundary_weight: float = 0.0, semantic_weight: float = 0.0) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    total = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y, boundary_target, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        boundary_target = boundary_target.to(device, non_blocking=True)
        boundary_valid = boundary_target[:, 1:2]
        teacher_semantic = boundary_target[:, 2].long()
        boundary_target = boundary_target[:, 0:1]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(x)
            logits, boundary_logits = output if isinstance(output, tuple) else (output, None)
            loss = criterion(logits, y)
            if proto is not None and proto_weight > 0 and num_classes > 2:
                raw = x * torch.as_tensor(std, device=device)[None, :, None, None]
                raw = raw + torch.as_tensor(mean, device=device)[None, :, None, None]
                raw = raw - raw.mean(dim=1, keepdim=True)
                raw = raw / (raw.square().sum(dim=1, keepdim=True).sqrt() + 1e-6)
                spectral_logits = torch.einsum("bchw,kc->bkhw", raw, proto) / 0.08
                valid_fg = (y > 0) & (y < num_classes)
                if valid_fg.any():
                    pred_fg = logits[:, 1:].permute(0, 2, 3, 1)[valid_fg]
                    target_fg = F.softmax(spectral_logits.permute(0, 2, 3, 1)[valid_fg], dim=-1)
                    loss = loss + proto_weight * F.kl_div(F.log_softmax(pred_fg, dim=-1), target_fg, reduction="batchmean")
            if boundary_logits is not None and boundary_weight > 0:
                valid_count = boundary_valid.sum()
                bce_map = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target, reduction="none")
                bce = (bce_map * boundary_valid).sum() / (valid_count + 1e-6)
                boundary_prob = torch.sigmoid(boundary_logits)
                inter = (boundary_prob * boundary_target * boundary_valid).sum()
                denom = ((boundary_prob + boundary_target) * boundary_valid).sum()
                boundary_dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
                loss = loss + boundary_weight * (bce + boundary_dice)
            if semantic_weight > 0:
                semantic_map = F.cross_entropy(logits, teacher_semantic, reduction="none")
                semantic_valid = boundary_valid[:, 0]
                semantic_loss = (semantic_map * semantic_valid).sum() / (semantic_valid.sum() + 1e-6)
                loss = loss + semantic_weight * semantic_loss
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        cm += confusion_matrix(pred, y.detach().cpu().numpy(), num_classes=num_classes)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1), metrics(cm)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, pred_dir: Path | None = None) -> tuple[float, dict, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for x, y, _, stems in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        cm += confusion_matrix(pred.cpu().numpy(), y.cpu().numpy(), num_classes=num_classes)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if pred_dir is not None:
            for i, stem in enumerate(stems):
                Image.fromarray(pred[i].cpu().numpy().astype(np.uint8)).save(pred_dir / f"{stem}.png")
    return total_loss / max(total, 1), metrics(cm), cm


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the 32-band HSI U-Net / MSPNet model and ablations.")
    parser.add_argument("--data-root", default=str(DATASET_ROOT))
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--label-mode", choices=["weak", "spectral", "multiclass", "multiclass_spectral", "triclass", "triclass_spectral"], default="weak")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prototype-loss", type=float, default=0.0)
    parser.add_argument("--prototype-json", default=None)
    parser.add_argument(
        "--prototype-input",
        action="store_true",
        help="Append cosine-similarity maps to the ground spectral prototypes as model inputs.",
    )
    parser.add_argument(
        "--ground-prototype-ratio",
        type=float,
        default=1.0,
        help="Ground-curve contribution after calibration with training-image class prototypes (0 to 1).",
    )
    parser.add_argument("--augment-factor", type=int, default=1)
    parser.add_argument("--class-weighting", choices=["inverse", "sqrt_inverse", "power075", "none"], default="inverse")
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--semantic-warmup-start", type=int, default=10)
    parser.add_argument("--semantic-warmup-end", type=int, default=30)
    parser.add_argument("--teacher-boundary-dir", default="teacher_boundaries")
    parser.add_argument("--teacher-valid-dir", default="teacher_valid")
    parser.add_argument(
        "--selection-metric",
        choices=["mIoU", "fg_mIoU"],
        default="mIoU",
        help="Validation metric used to select the checkpoint evaluated on the test split.",
    )
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "experiments" / f"unet_{args.label_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir) if args.split_dir else data_root / "splits"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"
    train_rows = read_rows(train_csv)
    if not train_rows:
        raise RuntimeError("Empty training split.")
    mean, std = compute_stats(train_rows)
    (out_dir / "norm_stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2), encoding="utf-8")

    use_teacher_boundary = args.boundary_weight > 0
    use_teacher_semantic = args.semantic_weight > 0
    train_ds = HSIRiceDataset(
        train_csv, args.label_mode, mean, std, train=True,
        augment_factor=args.augment_factor, use_teacher_boundary=use_teacher_boundary,
        teacher_boundary_dir=args.teacher_boundary_dir, use_teacher_semantic=use_teacher_semantic,
        teacher_valid_dir=args.teacher_valid_dir,
    )
    eval_label_mode = {
        "multiclass_spectral": "multiclass",
        "triclass_spectral": "triclass",
    }.get(args.label_mode, args.label_mode)
    val_ds = HSIRiceDataset(val_csv, eval_label_mode, mean, std, train=False)
    test_ds = HSIRiceDataset(test_csv, eval_label_mode, mean, std, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = 4 if args.label_mode.startswith("multiclass") else (3 if args.label_mode.startswith("triclass") else 2)
    key = {
        "weak": "mask_binary", "spectral": "mask_spectral",
        "multiclass": "mask_multi", "multiclass_spectral": "mask_multi_spectral",
        "triclass": "mask_tri", "triclass_spectral": "mask_tri",
    }[args.label_mode]
    proto = None
    if (args.prototype_loss > 0 or args.prototype_input) and num_classes > 2:
        default_proto = "threeclass_prototypes.json" if num_classes == 3 else "multiclass_spectral_metadata.json"
        proto_path = Path(args.prototype_json) if args.prototype_json else data_root / default_proto
        payload = json.loads(proto_path.read_text(encoding="utf-8"))
        proto = np.stack([payload["prototypes"][str(i)] for i in range(1, num_classes)]).astype(np.float32)
        proto = proto - proto.mean(axis=1, keepdims=True)
        proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-6)
        if not 0.0 <= args.ground_prototype_ratio <= 1.0:
            raise ValueError("--ground-prototype-ratio must be between 0 and 1")
        if args.ground_prototype_ratio < 1.0:
            image_proto = image_spectral_prototypes(train_rows, key, num_classes)
            proto = args.ground_prototype_ratio * proto + (1.0 - args.ground_prototype_ratio) * image_proto
            proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-6)
        proto = torch.from_numpy(proto)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = HSIUNet(
        in_ch=32,
        num_classes=num_classes,
        base=args.base,
        boundary_head=use_teacher_boundary,
        spectral_prototypes=proto if args.prototype_input else None,
        channel_mean=torch.from_numpy(mean),
        channel_std=torch.from_numpy(std),
    ).to(device)
    if proto is not None:
        proto = proto.to(device)
    weights = class_weights(train_rows, key, num_classes, args.class_weighting).to(device)
    criterion = SegmentationLoss(
        weights, args.dice_weight, ignore_index=255,
        focal_gamma=args.focal_gamma, lovasz_weight=args.lovasz_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    log_rows: list[dict] = []
    best_score = -1.0
    best_path = out_dir / "best_model.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.semantic_warmup_end <= args.semantic_warmup_start:
            semantic_ramp = 1.0
        else:
            semantic_ramp = np.clip(
                (epoch - args.semantic_warmup_start) / (args.semantic_warmup_end - args.semantic_warmup_start),
                0.0,
                1.0,
            )
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, num_classes,
            proto, args.prototype_loss, mean, std, args.boundary_weight,
            args.semantic_weight * float(semantic_ramp),
        )
        val_eval_ds = val_loader
        val_loss, val_metrics, _ = evaluate(model, val_eval_ds, criterion, device, num_classes)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score_key = args.selection_metric if num_classes > 2 else "Rice_IoU"
        if val_metrics[score_key] > best_score:
            best_score = val_metrics[score_key]
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics, test_cm = evaluate(model, test_loader, criterion, device, num_classes, out_dir / "test_predictions")
    summary = {
        "label_mode": args.label_mode,
        "split_dir": str(split_dir.resolve()),
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "base_channels": args.base,
        "selection_metric": score_key,
        "best_val_metric": best_score,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_confusion": test_cm.tolist(),
        "elapsed_sec": time.time() - start,
        "class_weights": weights.detach().cpu().tolist(),
        "augment_factor": args.augment_factor,
        "class_weighting": args.class_weighting,
        "dice_weight": args.dice_weight,
        "focal_gamma": args.focal_gamma,
        "lovasz_weight": args.lovasz_weight,
        "prototype_loss": args.prototype_loss,
        "prototype_input": args.prototype_input,
        "ground_prototype_ratio": args.ground_prototype_ratio,
        "boundary_weight": args.boundary_weight,
        "semantic_weight": args.semantic_weight,
        "semantic_warmup": [args.semantic_warmup_start, args.semantic_warmup_end],
        "teacher_boundary_dir": args.teacher_boundary_dir,
        "teacher_valid_dir": args.teacher_valid_dir,
    }
    write_csv(out_dir / "training_log.csv", log_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
