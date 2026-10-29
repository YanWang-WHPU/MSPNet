from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


from data_paths import read_rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create class-balanced, nonoverlapping spatial-block splits.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-name", default="splits_spatial_blocks")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--force-train-valid-dir",
        default=None,
        help="Optional directory of binary validity masks. Blocks containing valid pixels are assigned to train.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = read_rows(data_root / "splits" / "all.csv")
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["row"]) // args.block_size, int(row["col"]) // args.block_size)
        groups[key].append(row)
    keys = list(groups)
    block_counts = {
        key: np.asarray(
            [
                sum(int(row["background_pixels"]) for row in groups[key]),
                sum(int(row["rice_pixels"]) for row in groups[key]),
                sum(int(row["lotus_pixels"]) for row in groups[key]),
            ],
            dtype=np.float64,
        )
        for key in keys
    }
    total_classes = sum(block_counts.values())
    ratios = np.asarray([args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio])
    block_targets = np.maximum(1, np.round(ratios * len(keys)).astype(int))
    block_targets[0] += len(keys) - int(block_targets.sum())
    forced_train_keys = set()
    forced_train_patches = 0
    if args.force_train_valid_dir:
        valid_dir = Path(args.force_train_valid_dir)
        for row in rows:
            stem = Path(row["image"]).stem
            valid_path = valid_dir / f"{stem}.png"
            if valid_path.exists() and np.any(np.asarray(Image.open(valid_path)) > 0):
                forced_train_keys.add((int(row["row"]) // args.block_size, int(row["col"]) // args.block_size))
                forced_train_patches += 1
        if len(forced_train_keys) > block_targets[0]:
            raise ValueError(
                f"{len(forced_train_keys)} forced training blocks exceed the {block_targets[0]}-block train target"
            )
    free_keys = [key for key in keys if key not in forced_train_keys]
    free_targets = block_targets.copy()
    free_targets[0] -= len(forced_train_keys)
    rng = random.Random(args.seed)
    best = None
    for _ in range(args.trials):
        shuffled = free_keys.copy()
        rng.shuffle(shuffled)
        cuts = np.cumsum(free_targets)[:-1]
        free_partitions = np.split(np.asarray(shuffled, dtype=object), cuts)
        partitions = [
            np.asarray(list(forced_train_keys) + free_partitions[0].tolist(), dtype=object),
            free_partitions[1],
            free_partitions[2],
        ]
        class_sums = []
        patch_counts = []
        feasible = True
        for partition in partitions:
            selected = [tuple(value) for value in partition.tolist()]
            values = sum((block_counts[key] for key in selected), np.zeros(3))
            if np.any(values <= 0):
                feasible = False
                break
            class_sums.append(values)
            patch_counts.append(sum(len(groups[key]) for key in selected))
        if not feasible:
            continue
        class_sums = np.stack(class_sums)
        class_ratios = class_sums / total_classes[None]
        patch_ratios = np.asarray(patch_counts) / len(rows)
        score = float(np.square((class_ratios - ratios[:, None]) / np.maximum(ratios[:, None], 0.05)).mean())
        score += float(2.0 * np.square(patch_ratios - ratios).mean())
        if best is None or score < best[0]:
            best = (score, [[tuple(value) for value in part.tolist()] for part in partitions])
    if best is None:
        raise RuntimeError("Could not find a split containing all classes")

    names = ("train", "val", "test")
    output = {}
    summary = {
        "score": best[0],
        "block_size": args.block_size,
        "strict_spatial_blocks": True,
        "forced_train_valid_dir": str(Path(args.force_train_valid_dir).resolve()) if args.force_train_valid_dir else None,
        "forced_train_blocks": [list(key) for key in sorted(forced_train_keys)],
        "forced_train_patches": forced_train_patches,
    }
    for name, partition in zip(names, best[1]):
        output[name] = [row for key in partition for row in groups[key]]
    out_dir = data_root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "all.csv", [row for name in names for row in output[name]])
    for name in names:
        write_rows(out_dir / f"{name}.csv", output[name])
        summary[name] = {
            "patches": len(output[name]),
            "blocks": [list(key) for key in best[1][names.index(name)]],
            "class_pixels": [
                sum(int(row[column]) for row in output[name])
                for column in ("background_pixels", "rice_pixels", "lotus_pixels")
            ],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
