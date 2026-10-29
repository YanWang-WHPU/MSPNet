"""Train MSPNet using the released configuration; CLI options override JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mspnet.json")
    args, overrides = parser.parse_known_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                options.append(flag)
        elif value is not None:
            options.extend([flag, str(value)])
    from train_hsi_unet import main as train
    train(options + overrides)


if __name__ == "__main__":
    main()
