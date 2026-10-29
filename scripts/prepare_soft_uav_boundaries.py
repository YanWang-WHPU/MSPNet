from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create alignment-tolerant UAV boundary targets.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "hfz3"))
    parser.add_argument("--source-dir", default="teacher_boundaries")
    parser.add_argument("--out-dir", default="teacher_boundaries_soft")
    parser.add_argument("--sigma", type=float, default=1.5)
    args = parser.parse_args()

    if args.sigma <= 0:
        raise ValueError("--sigma must be positive")
    data_root = Path(args.data_root)
    source_dir = data_root / args.source_dir
    valid_dir = data_root / "teacher_valid"
    out_dir = data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not list(source_dir.glob("*.png")):
        raise FileNotFoundError(f"No boundary PNGs found in {source_dir}")
    written = 0
    nonempty = 0
    for source in sorted(source_dir.glob("*.png")):
        boundary = np.asarray(Image.open(source)) > 0
        valid = np.asarray(Image.open(valid_dir / source.name)) > 0
        if np.any(boundary):
            distance = distance_transform_edt(~boundary)
            target = np.exp(-0.5 * np.square(distance / args.sigma)) * valid
            nonempty += 1
        else:
            target = np.zeros(boundary.shape, dtype=np.float32)
        Image.fromarray(np.round(target * 255).astype(np.uint8)).save(out_dir / source.name)
        written += 1
    print({"written": written, "nonempty": nonempty, "sigma": args.sigma, "out_dir": str(out_dir.resolve())})


if __name__ == "__main__":
    main()
