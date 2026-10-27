"""FastAPI inference service for the keyword spotting model."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import load_audio_features
from src.model import AudioCNN

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("kws-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(MODEL_PATH.with_name("metrics.json"))))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(MODEL_PATH.with_name("labels.json"))))
DEVICE = torch.device("cpu")

state: dict[str, object] = {}


def load_model() -> None:
    """Load the model artifact once at service startup."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if LABELS_PATH.is_file():
        labels_map = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        classes = [labels_map[str(i)] for i in range(len(labels_map))]
    else:
        classes = list(checkpoint["classes"])

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.is_file() else {}
    state.update(
        model=model,
        classes=classes,
        checkpoint=checkpoint,
        metrics=metrics,
    )
    logger.info("Loaded model from %s (classes=%s)", MODEL_PATH, classes)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield
    state.clear()


app = FastAPI(title="Keyword Spotting API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    checkpoint = state["checkpoint"]
    metrics = state["metrics"]
    return {
        "architecture": checkpoint.get("architecture", "AudioCNN"),
        "classes": state["classes"],
        "model_size_kb": round(MODEL_PATH.stat().st_size / 1024, 2),
        "accuracy": metrics.get("accuracy"),
        "model_version": checkpoint.get("model_version", metrics.get("model_version", "1.0.0")),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    filename = file.filename or "audio.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav audio files are supported")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name

        started = time.perf_counter()
        features = load_audio_features(temp_path).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = state["model"](features)
            probs_tensor = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted WAV file: {exc}") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    classes: list[str] = state["classes"]
    probabilities = {
        label: round(float(probs_tensor[idx]), 4) for idx, label in enumerate(classes)
    }
    best_idx = int(probs_tensor.argmax())
    checkpoint = state["checkpoint"]
    return {
        "label": classes[best_idx],
        "confidence": round(float(probs_tensor[best_idx]), 4),
        "probabilities": probabilities,
        "latency_ms": round(latency_ms, 2),
        "model_version": checkpoint.get("model_version", "1.0.0"),
    }
