"""Release checks using temporary synthetic inputs, never research data."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import MSPNet
from predict import load_model, predict
from data_paths import read_rows
from train_hsi_unet import SegmentationLoss, metrics, train_epoch


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def model(self, boundary=True):
        prototypes = torch.randn(2, 32)
        prototypes = torch.nn.functional.normalize(prototypes, dim=1)
        return MSPNet(base=2, boundary_head=boundary, spectral_prototypes=prototypes,
                      channel_mean=torch.zeros(32), channel_std=torch.ones(32))

    def test_model_outputs_and_checkpoint_roundtrip(self):
        model = self.model().eval()
        x = torch.randn(1, 32, 32, 32)
        with torch.no_grad():
            logits, boundary = model(x)
        self.assertEqual(tuple(logits.shape), (1, 3, 32, 32))
        self.assertEqual(tuple(boundary.shape), (1, 1, 32, 32))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model": model.state_dict(), "mean": np.zeros(32, np.float32),
                        "std": np.ones(32, np.float32)}, checkpoint)
            restored, mean, std = load_model(checkpoint, torch.device("cpu"))
            with torch.no_grad():
                torch.testing.assert_close(logits, restored(x)[0])
            labels = predict(restored, x[0].numpy(), mean, std, torch.device("cpu"))
            np.testing.assert_array_equal(labels, logits.argmax(1)[0].numpy())
            odd = predict(restored, np.zeros((32, 29, 35), np.float32), mean, std, torch.device("cpu"))
            self.assertEqual(odd.shape, (29, 35))
            self.assertTrue(np.all(odd < 3))

    def test_uncovered_boundary_head_has_zero_gradient(self):
        model = self.model()
        batch = (torch.randn(2, 32, 32, 32), torch.randint(0, 3, (2, 32, 32)),
                 torch.zeros(2, 3, 32, 32), ["a", "b"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        criterion = SegmentationLoss(torch.ones(3), dice_weight=0.5)
        loss, _ = train_epoch(model, [batch], optimizer, criterion, torch.device("cpu"),
                              None, 3, boundary_weight=0.05)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(torch.count_nonzero(model.boundary_head.weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(model.boundary_head.bias.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(model.head.weight.grad).item(), 0)

    def test_portable_csv_and_metric_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits" / "train.csv"
            path.parent.mkdir()
            path.write_text("image,mask_tri\n../images/a.npz,../masks_triclass/a.png\n", encoding="utf-8-sig")
            row = read_rows(path)[0]
            self.assertEqual(Path(row["image"]), Path(directory).resolve() / "images/a.npz")
            self.assertEqual(Path(row["mask_tri"]), Path(directory).resolve() / "masks_triclass/a.png")
        result = metrics(np.eye(3, dtype=np.int64))
        self.assertEqual(result["mF1"], 1.0)
        self.assertEqual(result["mIoU"], 1.0)

    def run_cli(self, *arguments):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_training_prediction_and_evaluation_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            for name in ("images", "masks_triclass", "teacher_boundaries", "teacher_valid", "splits"):
                (data / name).mkdir(parents=True)
            rng = np.random.default_rng(2026)
            rows = []
            for i in range(4):
                stem = f"synthetic_{i}"
                np.savez_compressed(data / "images" / f"{stem}.npz", image=rng.uniform(10, 100, (32, 32, 32)).astype(np.float32))
                Image.fromarray(rng.integers(0, 3, (32, 32), dtype=np.uint8)).save(data / "masks_triclass" / f"{stem}.png")
                edge = np.zeros((32, 32), np.uint8)
                edge[:, 16] = 255
                valid = np.full((32, 32), 255 if i == 0 else 0, np.uint8)
                Image.fromarray(edge).save(data / "teacher_boundaries" / f"{stem}.png")
                Image.fromarray(valid).save(data / "teacher_valid" / f"{stem}.png")
                rows.append({"image": f"../images/{stem}.npz", "mask_tri": f"../masks_triclass/{stem}.png", "row": 0, "col": 0})
            for name, selected in (("train", rows[:2]), ("val", rows[2:3]), ("test", rows[3:])):
                with (data / "splits" / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(selected)
            (data / "threeclass_prototypes.json").write_text(json.dumps({"prototypes": {str(i): rng.random(32).tolist() for i in (1, 2)}}), encoding="utf-8")
            (data / "metadata.json").write_text(json.dumps({"raster_shape": [32, 32, 32]}), encoding="utf-8")
            output = Path(directory) / "output"
            self.run_cli("scripts/prepare_soft_uav_boundaries.py", "--data-root", data)
            self.run_cli("train.py", "--data-root", data, "--split-dir", data / "splits",
                         "--out-dir", output, "--epochs", "1", "--base", "2", "--batch-size", "2", "--device", "cpu")
            checkpoint = output / "best_model.pt"
            self.assertTrue(checkpoint.is_file())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(np.isfinite(summary["test_loss"]))
            self.assertIn("mF1", summary["test_metrics"])
            self.run_cli("predict.py", "--checkpoint", checkpoint, "--image", data / "images/synthetic_3.npz",
                         "--output", output / "prediction.png", "--device", "cpu")
            self.run_cli("scripts/evaluate_hfz3_checkpoint.py", "--data-root", data,
                         "--split-dir", data / "splits", "--checkpoint", checkpoint, "--out-dir", output / "evaluation", "--device", "cpu")
            evaluation = json.loads((output / "evaluation/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation["test_patches"], 1)
            self.assertEqual(sum(map(sum, evaluation["confusion"])), 32 * 32)


if __name__ == "__main__":
    unittest.main()
