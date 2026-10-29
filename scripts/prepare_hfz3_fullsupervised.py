from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import arcpy
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "data" / "hfz3"


def find_one(pattern: str) -> Path:
    matches = sorted(DATA.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def lower_left(raster: arcpy.Raster, row: int, col: int, size: int) -> arcpy.Point:
    return arcpy.Point(
        raster.extent.XMin + col * raster.meanCellWidth,
        raster.extent.YMax - (row + size) * raster.meanCellHeight,
    )


def robust_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.stack([image[12], image[6], image[2]], axis=-1).astype(np.float32)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        band = rgb[..., channel]
        valid = band > 0
        if not np.any(valid):
            continue
        low, high = np.percentile(band[valid], [2, 98])
        if high > low:
            output[..., channel] = np.clip((band - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full-supervised HFZ3 patches from the corresponding OHS mask.")
    parser.add_argument("--hsi", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--label-field", default="gridcode_1")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    args = parser.parse_args()

    hsi_path = Path(args.hsi) if args.hsi else find_one("**/*TOA_32band_rpc_dem_mosaic_clip.tif")
    label_path = Path(args.label) if args.label else find_one("**/高光谱anxiang_lable/merge-12-23.shp")
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out_root}")
    images_dir = out_root / "images"
    masks_dir = out_root / "masks_triclass"
    previews_dir = out_root / "previews"
    work_dir = out_root / "_work"
    split_dir = out_root / "splits"
    for directory in (images_dir, masks_dir, previews_dir, work_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raster = arcpy.Raster(str(hsi_path))
    desc = arcpy.Describe(str(hsi_path))
    rows = int(arcpy.management.GetRasterProperties(str(hsi_path), "ROWCOUNT").getOutput(0))
    cols = int(arcpy.management.GetRasterProperties(str(hsi_path), "COLUMNCOUNT").getOutput(0))
    bands = int(arcpy.management.GetRasterProperties(str(hsi_path), "BANDCOUNT").getOutput(0))
    if bands != 32:
        raise ValueError(f"Expected 32 bands, got {bands}")

    gdb = work_dir / "labels.gdb"
    arcpy.management.CreateFileGDB(str(work_dir), gdb.name)
    projected = str(gdb / "ohs_labels_wgs84")
    arcpy.management.Project(str(label_path), projected, desc.spatialReference)
    label_raster = work_dir / "ohs_gridcode.tif"
    with arcpy.EnvManager(
        snapRaster=str(hsi_path),
        extent=str(hsi_path),
        cellSize=str(hsi_path),
        outputCoordinateSystem=desc.spatialReference,
        overwriteOutput=True,
    ):
        arcpy.conversion.PolygonToRaster(
            projected, args.label_field, str(label_raster), "MAXIMUM_AREA", None, raster.meanCellWidth
        )

    multiclass = arcpy.RasterToNumPyArray(str(label_raster), nodata_to_value=0).astype(np.uint8)
    tri = np.zeros_like(multiclass)
    tri[(multiclass == 1) | (multiclass == 2)] = 1
    tri[multiclass == 3] = 2

    records = []
    preview_count = 0
    class_totals = np.zeros(3, dtype=np.int64)
    for row in range(0, rows - args.patch + 1, args.patch):
        for col in range(0, cols - args.patch + 1, args.patch):
            image = arcpy.RasterToNumPyArray(
                str(hsi_path), lower_left(raster, row, col, args.patch), args.patch, args.patch, nodata_to_value=0
            )
            if image.ndim == 2:
                image = image[None]
            valid = np.any(image != 0, axis=0)
            if valid.mean() < args.min_valid_fraction:
                continue
            mask = tri[row : row + args.patch, col : col + args.patch]
            stem = f"HFZ3_r{row:05d}_c{col:05d}"
            image_path = images_dir / f"{stem}.npz"
            mask_path = masks_dir / f"{stem}.png"
            np.savez_compressed(image_path, image=image.astype(np.uint16))
            Image.fromarray(mask).save(mask_path, compress_level=3)

            counts = np.bincount(mask.ravel(), minlength=3)
            class_totals += counts
            records.append(
                {
                    "image": str(image_path.resolve()),
                    "mask_tri": str(mask_path.resolve()),
                    "scene_id": "HFZ3_20250326",
                    "row": row,
                    "col": col,
                    "background_pixels": int(counts[0]),
                    "rice_pixels": int(counts[1]),
                    "lotus_pixels": int(counts[2]),
                }
            )
            if preview_count < 60 and counts[1:].sum() > 0:
                rgb = robust_rgb(image)
                palette = np.asarray([[0, 0, 0], [40, 190, 70], [220, 70, 90]], dtype=np.uint8)
                overlay = rgb.astype(np.float32)
                foreground = mask > 0
                overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * palette[mask[foreground]]
                Image.fromarray(np.concatenate([rgb, overlay.astype(np.uint8), palette[mask]], axis=1)).save(
                    previews_dir / f"{stem}.jpg", quality=90
                )
                preview_count += 1

    if not records:
        raise RuntimeError("No valid patches generated")
    write_csv(split_dir / "all.csv", records)
    metadata = {
        "hsi": str(hsi_path.resolve()),
        "label": str(label_path.resolve()),
        "label_field": args.label_field,
        "label_mapping": {"0": "Background", "1/2": "Rice", "3": "Lotus"},
        "patch": args.patch,
        "stride": args.patch,
        "strict_nonoverlap": True,
        "raster_shape": [bands, rows, cols],
        "patches": len(records),
        "class_pixels": class_totals.tolist(),
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
