"""Dataset loading and Mel-spectrogram preprocessing for Google Speech Commands."""
from __future__ import annotations

import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 8_000
NUM_SAMPLES = SAMPLE_RATE
N_MELS = 32
SPEECH_COMMANDS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_SUBDIR = Path("SpeechCommands") / "speech_commands_v0.02"

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    hop_length=128,
    n_mels=N_MELS,
)
_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")


def ensure_speech_commands(root: Path, classes: Sequence[str]) -> Path:
    """Ensure the Speech Commands dataset directory exists with required classes."""
    extract_dir = root / DATASET_SUBDIR
    required_ready = (extract_dir / "testing_list.txt").is_file() and all(
        (extract_dir / label).is_dir() for label in classes
    )
    if required_ready:
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    archive_path = root / "speech_commands_v0.02.tar.gz"
    if not archive_path.is_file():
        print(f"Downloading {SPEECH_COMMANDS_URL} ...", flush=True)
        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, archive_path)

    allowed_roots = set(classes) | {
        "testing_list.txt",
        "validation_list.txt",
        "LICENSE",
        "README.md",
    }
    print(f"Extracting target classes {list(classes)} ...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            clean_name = member.name.lstrip("./")
            top_name = clean_name.split("/", 1)[0]
            if top_name in allowed_roots:
                members.append(member)
        tar.extractall(path=extract_dir, members=members)

    archive_path.unlink(missing_ok=True)
    return extract_dir


class SpeechCommandsSubset(Dataset):
    """Filtered Google Speech Commands dataset returning normalized log-Mel spectrograms."""

    def __init__(
        self,
        root: str | Path,
        subset: str,
        classes: Sequence[str],
        download: bool = True,
        max_per_class: int = 0,
        seed: int = 42,
    ) -> None:
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"Unsupported subset: {subset}")

        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if download:
            extract_dir = ensure_speech_commands(root_path, classes)
        else:
            extract_dir = root_path / DATASET_SUBDIR

        testing_set = {
            line.strip()
            for line in (extract_dir / "testing_list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        validation_set = {
            line.strip()
            for line in (extract_dir / "validation_list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        self.classes = list(classes)
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        rng = random.Random(seed)
        self.samples: list[tuple[Path, int]] = []

        for label in self.classes:
            class_dir = extract_dir / label
            files = sorted(class_dir.glob("*.wav"))
            filtered: list[Path] = []
            for file_path in files:
                rel_key = f"{label}/{file_path.name}"
                if subset == "testing" and rel_key in testing_set:
                    filtered.append(file_path)
                elif subset == "validation" and rel_key in validation_set:
                    filtered.append(file_path)
                elif subset == "training" and rel_key not in testing_set and rel_key not in validation_set:
                    filtered.append(file_path)

            rng.shuffle(filtered)
            if max_per_class > 0:
                filtered = filtered[:max_per_class]
            label_idx = self.class_to_idx[label]
            self.samples.extend((path, label_idx) for path in filtered)

        if not self.samples:
            raise RuntimeError(f"No samples found for classes: {', '.join(self.classes)}")

        self._cache: dict[int, tuple[torch.Tensor, int]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        file_path, label_idx = self.samples[index]
        features = load_audio_features(file_path)
        item = (features, label_idx)
        self._cache[index] = item
        return item


def normalize_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Convert waveform to mono, resample to 8 kHz, and pad/trim to 1 second."""
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError("Audio must contain at least one non-empty channel")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.shape[1] < NUM_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :NUM_SAMPLES]
    return waveform


def waveform_to_features(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Compute standardized log-Mel spectrogram features from waveform."""
    waveform = normalize_waveform(waveform, sample_rate)
    features = _db_transform(_mel_transform(waveform))
    mean = features.mean()
    std = features.std().clamp_min(1e-6)
    return (features - mean) / std


def load_audio_features(path: str | Path) -> torch.Tensor:
    """Load a WAV file from disk and convert it to model input features."""
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform_to_features(waveform, sample_rate)
