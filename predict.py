"""Predict class IDs from an HSI patch using a local training checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import MSPNet


def load_model(checkpoint_path: Path, device: torch.device):
    # Historical project checkpoints also contain NumPy normalization arrays.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "mean" not in checkpoint or "std" not in checkpoint:
        raise ValueError("Expected a checkpoint produced by train.py / train_hsi_unet.py")
    state = checkpoint["model"]
    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32)
    if mean.shape != (32,) or std.shape != (32,) or not torch.all(std > 0):
        raise ValueError("Checkpoint must contain 32-band statistics with positive std")
    model = MSPNet(
        in_ch=32,
        num_classes=state["head.weight"].shape[0],
        base=state["enc1.0.0.weight"].shape[0],
        boundary_head="boundary_head.weight" in state,
        spectral_prototypes=state.get("spectral_prototypes"),
        channel_mean=mean,
        channel_std=std,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, mean.numpy(), std.numpy()


@torch.no_grad()
def predict(model, image: np.ndarray, mean: np.ndarray, std: np.ndarray, device):
    if image.ndim != 3 or image.shape[0] != 32:
        raise ValueError("Expected image shape (32, H, W)")
    if min(image.shape[1:]) < 1 or not np.all(np.isfinite(image)):
        raise ValueError("Image must be nonempty and finite")
    height, width = image.shape[1:]
    normalized = (image.astype(np.float32) - mean[:, None, None]) / std[:, None, None]
    x = torch.from_numpy(np.ascontiguousarray(normalized))[None].to(device)
    x = F.pad(x, (0, (-width) % 16, 0, (-height) % 16), mode="replicate")
    output = model(x)
    logits = output[0] if isinstance(output, tuple) else output
    return logits.argmax(1)[0, :height, :width].cpu().numpy().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted local checkpoint from train.py")
    parser.add_argument("--image", required=True, type=Path, help="NPZ containing an unnormalized (32,H,W) image array")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG containing class IDs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, mean, std = load_model(args.checkpoint, device)
    with np.load(args.image, allow_pickle=False) as payload:
        labels = predict(model, payload["image"], mean, std, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
