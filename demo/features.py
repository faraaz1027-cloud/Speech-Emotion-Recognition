"""Acoustic feature extraction matching the team's CSE424 notebook.

This module is an audio feature explorer, NOT a trained classifier.
It extracts exactly 58 features: 40 MFCC, 12 chroma, 5 spectral/energy,
and one tempo. The original notebook used 22,050 Hz and silence trimming.
"""
from __future__ import annotations

from io import BytesIO
import numpy as np
import librosa

SAMPLE_RATE = 22050
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SECONDS = 30


def load_audio(payload: bytes):
    """Decode an uploaded audio file in memory; normalize as in coursework."""
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Please upload a WAV file smaller than 15 MB.")
    try:
        y, sr = librosa.load(BytesIO(payload), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError("Could not decode the audio. Please use a standard WAV file.") from exc

    if not np.all(np.isfinite(y)) or not len(y):
        raise ValueError("Audio is empty or contains invalid values.")
    if len(y) / sr > MAX_SECONDS:
        raise ValueError("Please upload a recording 30 seconds or shorter.")
    y, _ = librosa.effects.trim(y, top_db=25)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= 1e-7:
        raise ValueError("The clip is silent. Upload speech with audible content.")
    y = (y / peak).astype(np.float32)
    return y, int(sr)


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Return ordered feature names/values consistent with original notebook."""
    if y.ndim != 1 or len(y) < 512:
        raise ValueError("Recording too short for acoustic analysis.")
    features: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for idx in range(20):
        features[f"mfcc_{idx+1}_mean"] = float(np.mean(mfcc[idx]))
        features[f"mfcc_{idx+1}_std"] = float(np.std(mfcc[idx]))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for idx in range(12):
        features[f"chroma_{idx+1}_mean"] = float(np.mean(chroma[idx]))
    features["spectral_centroid_mean"] = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    )
    features["spectral_bandwidth_mean"] = float(
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    )
    features["spectral_rolloff_mean"] = float(
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    )
    features["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)))
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        features["tempo"] = 0.0

    if len(features) != 58 or not np.all(np.isfinite(list(features.values()))):
        raise ValueError("Could not compute a valid 58-feature vector.")
    return features
