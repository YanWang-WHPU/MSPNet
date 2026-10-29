"""Fetch the exact upstream baseline revisions used by the local experiments."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "DeepLabV3Plus-Pytorch": ("https://github.com/VainF/DeepLabV3Plus-Pytorch.git", "4e1087de98bc49d55b9239ae92810ef7368660db"),
    "SegFormer": ("https://github.com/NVlabs/SegFormer.git", "65fa8cfa9b52b6ee7e8897a98705abf8570f9e32"),
    "CMX": ("https://github.com/huaaaliu/RGBX_Semantic_Segmentation.git", "e251d860aebc2f583a6c4919877e6bebe7f1aff3"),
    "UM2Former": ("https://github.com/ZhaohuiXue/UM2Former.git", "c4611a354fac1dde5c721f38e08f1457ddf65fc9"),
}


def run(*args):
    subprocess.run(["git", *map(str, args)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    for name in args.models:
        url, revision = SOURCES[name]
        destination = ROOT / "baselines" / name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}; move it aside to download again")
        run("clone", "--filter=blob:none", "--no-checkout", url, destination)
        run("-C", destination, "checkout", "--detach", revision)
        if name == "UM2Former":
            run("-C", destination, "apply", ROOT / "baselines/patches/unetformer-small-input.patch")


if __name__ == "__main__":
    main()
