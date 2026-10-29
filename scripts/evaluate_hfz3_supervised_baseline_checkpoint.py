from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_hfz3_supervised_baselines import build_model
from train_hsi_unet import confusion_matrix, label_path, metrics, read_rows


class EvalDataset(Dataset):
    def __init__(self, csv_path: Path, mean: np.ndarray, std: np.ndarray):
        self.rows = read_rows(csv_path)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = np.load(row["image"])["image"].astype(np.float32)
        target = np.asarray(Image.open(label_path(row, "mask_tri")), dtype=np.int64)
        image = (image - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(target), Path(row["image"]).stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HFZ3-only baseline checkpoint on a target split.")
    parser.add_argument("--model", required=True, choices=["deeplabv3plus", "segformer_b0", "pspnet", "unetformer"])
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    stats = json.loads(Path(args.norm_stats).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    dataset = EvalDataset(split_dir / "test.csv", mean, std)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    matrix = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for image, target, stems in loader:
            image = image.to(device)
            logits = model(image)
            prediction = logits.argmax(dim=1).cpu().numpy()
            target_np = target.numpy()
            matrix += confusion_matrix(prediction, target_np, 3)
            Image.fromarray(prediction[0].astype(np.uint8)).save(prediction_dir / f"{stems[0]}.png")

    summary = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split_dir": str(split_dir.resolve()),
        "test_patches": len(dataset),
        "metrics": metrics(matrix),
        "confusion": matrix.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
