from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import arcpy
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


from data_paths import read_rows


def boundary_within_valid(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (labels[1:, :] != labels[:-1, :])
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare coverage-aware UAV teacher targets for full HFZ3 patches.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "hfz3"))
    parser.add_argument("--teacher-work", required=True, help="UAV labels already aligned to the HSI grid; see docs/data.md.")
    parser.add_argument("--prototype-json", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    hsi_path = Path(metadata["hsi"])
    patch = int(metadata["patch"])
    hsi = arcpy.Raster(str(hsi_path))
    height, width = int(hsi.height), int(hsi.width)
    full_teacher = np.zeros((height, width), dtype=np.uint8)
    full_valid = np.zeros((height, width), dtype=bool)
    lower_left = arcpy.Point(hsi.extent.XMin, hsi.extent.YMin)

    sources = sorted(Path(args.teacher_work).glob("*_teacher_ohs.tif"))
    if not sources:
        raise FileNotFoundError("No *_teacher_ohs.tif files found in --teacher-work")
    for source in sources:
        array = arcpy.RasterToNumPyArray(
            str(source), lower_left, width, height, nodata_to_value=255
        ).astype(np.uint16)
        valid = array != 255
        tri = np.zeros(array.shape, dtype=np.uint8)
        tri[(array == 1) | (array == 2)] = 1
        tri[array == 3] = 2
        full_teacher[valid] = tri[valid]
        full_valid |= valid

    full_boundary = boundary_within_valid(full_teacher, full_valid)
    boundary_dir = data_root / "teacher_boundaries"
    valid_dir = data_root / "teacher_valid"
    teacher_dir = data_root / "teacher_triclass"
    for directory in (boundary_dir, valid_dir, teacher_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_rows(data_root / "splits" / "all.csv")
    patches_with_uav = 0
    valid_pixels = 0
    boundary_pixels = 0
    for row in rows:
        stem = Path(row["image"]).stem
        r, c = int(row["row"]), int(row["col"])
        target = full_teacher[r : r + patch, c : c + patch]
        valid = full_valid[r : r + patch, c : c + patch]
        boundary = full_boundary[r : r + patch, c : c + patch]
        Image.fromarray(target).save(teacher_dir / f"{stem}.png")
        Image.fromarray(valid.astype(np.uint8) * 255).save(valid_dir / f"{stem}.png")
        Image.fromarray(boundary.astype(np.uint8) * 255).save(boundary_dir / f"{stem}.png")
        count = int(valid.sum())
        if count:
            patches_with_uav += 1
            valid_pixels += count
            boundary_pixels += int(boundary.sum())

    prototype_target = data_root / "threeclass_prototypes.json"
    if Path(args.prototype_json).resolve() != prototype_target.resolve():
        shutil.copy2(args.prototype_json, prototype_target)
    summary = {
        "teacher_sources": [path.name for path in sorted(Path(args.teacher_work).glob("*_teacher_ohs.tif"))],
        "patches": len(rows),
        "patches_with_uav_coverage": patches_with_uav,
        "uav_valid_pixels": valid_pixels,
        "uav_boundary_pixels": boundary_pixels,
        "coverage_policy": "UAV auxiliary loss is zero outside teacher_valid.",
        "prototype_json": str((data_root / "threeclass_prototypes.json").resolve()),
    }
    (data_root / "auxiliary_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
