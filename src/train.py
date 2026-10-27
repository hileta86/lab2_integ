"""Reproducible training script generating artifacts/model.pt, labels.json, metrics.json."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import SAMPLE_RATE, SpeechCommandsSubset
from src.model import AudioCNN

DEFAULT_CLASSES = ["yes", "no", "up", "down"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return 100.0 * correct / total


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cpu")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = SpeechCommandsSubset(
        args.data_dir,
        "training",
        args.classes,
        max_per_class=args.max_samples_per_class,
        seed=args.seed,
    )
    test_set = SpeechCommandsSubset(
        args.data_dir,
        "testing",
        args.classes,
        max_per_class=args.max_test_samples_per_class,
        seed=args.seed,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AudioCNN(num_classes=len(args.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss_sum / len(train_loader):.4f}", flush=True)

    test_accuracy = compute_accuracy(model, test_loader, device)
    model_path = output_dir / "model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": list(args.classes),
        "sample_rate": SAMPLE_RATE,
        "architecture": "AudioCNN",
        "model_version": args.model_version,
    }
    torch.save(checkpoint, model_path)

    labels = {str(index): label for index, label in enumerate(args.classes)}
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    metrics: dict[str, object] = {
        "dataset": "Google Speech Commands v0.02",
        "classes": list(args.classes),
        "architecture": "AudioCNN",
        "accuracy": round(test_accuracy, 2),
        "model_size_kb": round(model_path.stat().st_size / 1024, 2),
        "model_version": args.model_version,
        "epochs": args.epochs,
        "seed": args.seed,
        "training_samples": len(train_set),
        "test_samples": len(test_set),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "5")))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--model-version", default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=int(os.getenv("MAX_SAMPLES_PER_CLASS", "0")),
    )
    parser.add_argument("--max-test-samples-per-class", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
