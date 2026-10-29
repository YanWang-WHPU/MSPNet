from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from train_hsi_unet import HSIUNet, confusion_matrix, metrics


from data_paths import read_rows


def class_boundary(labels: np.ndarray, valid: np.ndarray, class_id: int) -> np.ndarray:
    member = labels == class_id
    result = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (member[:, 1:] != member[:, :-1])
    vertical = valid[1:] & valid[:-1] & (member[1:] != member[:-1])
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    result[1:] |= vertical
    result[:-1] |= vertical
    return result


def boundary_f1(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
                class_id: int, tolerance: int) -> dict:
    pred_boundary = class_boundary(prediction, valid, class_id)
    target_boundary = class_boundary(target, valid, class_id)
    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())
    if pred_count == 0 or target_count == 0:
        return {"precision": 0.0, "recall": 0.0, "F1": 0.0,
                "prediction_boundary_pixels": pred_count, "target_boundary_pixels": target_count}
    target_distance = distance_transform_edt(~target_boundary)
    pred_distance = distance_transform_edt(~pred_boundary)
    precision = float((target_distance[pred_boundary] <= tolerance).mean())
    recall = float((pred_distance[target_boundary] <= tolerance).mean())
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "prediction_boundary_pixels": pred_count,
        "target_boundary_pixels": target_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HFZ3 U-Net checkpoint on another mask split.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    _, height, width = metadata["raster_shape"]

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    ckpt_args = checkpoint.get("args", {})
    prototypes = state.get("spectral_prototypes")
    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32)
    num_classes = int(state["head.weight"].shape[0])
    model = HSIUNet(
        in_ch=32,
        num_classes=num_classes,
        base=int(ckpt_args.get("base", 24)),
        boundary_head="boundary_head.weight" in state,
        spectral_prototypes=prototypes,
        channel_mean=mean,
        channel_std=std,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    mean_np = mean.numpy()[:, None, None]
    std_np = std.numpy()[:, None, None]

    target_canvas = np.full((height, width), 255, dtype=np.uint8)
    prediction_canvas = np.zeros((height, width), dtype=np.uint8)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    rows = read_rows(split_dir / "test.csv")
    with torch.no_grad():
        for row in rows:
            image = np.load(row["image"])["image"].astype(np.float32)
            target = np.asarray(Image.open(row["mask_tri"]), dtype=np.uint8)
            x = torch.from_numpy(((image - mean_np) / std_np)[None]).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(x)
                logits = output[0] if isinstance(output, tuple) else output
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            matrix += confusion_matrix(prediction, target, num_classes=num_classes)
            stem = Path(row["image"]).stem
            Image.fromarray(prediction).save(prediction_dir / f"{stem}.png")
            row0, col0 = int(row["row"]), int(row["col"])
            patch_height, patch_width = target.shape
            valid_patch = target != 255
            target_view = target_canvas[row0 : row0 + patch_height, col0 : col0 + patch_width]
            prediction_view = prediction_canvas[row0 : row0 + patch_height, col0 : col0 + patch_width]
            target_view[valid_patch] = target[valid_patch]
            prediction_view[valid_patch] = prediction[valid_patch]

    valid = target_canvas != 255
    boundary = {}
    for name, class_id in (("Rice", 1), ("Lotus", 2)):
        boundary[name] = {
            str(tolerance): boundary_f1(
                prediction_canvas, target_canvas, valid, class_id, tolerance
            )
            for tolerance in (1, 2)
        }
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "split_dir": str(split_dir.resolve()),
        "test_patches": len(rows),
        "valid_pixels": int(valid.sum()),
        "metrics": metrics(matrix),
        "confusion": matrix.tolist(),
        "boundary_f1": boundary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
