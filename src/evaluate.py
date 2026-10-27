"""Evaluate a trained model artifact on the Google Speech Commands test set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import SpeechCommandsSubset
from src.model import AudioCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "artifacts/model.pt"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    args = parser.parse_args()

    model_path = Path(args.model)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SpeechCommandsSubset(
        args.data_dir,
        "testing",
        classes,
        download=True,
        max_per_class=args.max_samples_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.numel()

    result = {
        "accuracy": round(100.0 * correct / total, 2),
        "correct": correct,
        "total": total,
        "model_size_kb": round(model_path.stat().st_size / 1024, 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
