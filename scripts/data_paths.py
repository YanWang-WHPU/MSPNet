"""Read portable split manifests without changing their ordering."""
from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value and (key == "image" or key.startswith("mask_")):
                entry = Path(value)
                if not entry.is_absolute():
                    entry = path.parent / entry
                row[key] = str(entry.resolve())
    return rows
