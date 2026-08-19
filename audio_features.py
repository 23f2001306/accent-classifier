"""Shared audio feature extraction for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import librosa
import numpy as np


AudioInput = Union[str, Path, BinaryIO]


def extract_mfcc(file_path_or_file_object: AudioInput) -> np.ndarray:
    """Return the 26 MFCC summary features used by the classifier.

    Audio is converted to mono and resampled to 16 kHz.  Thirteen MFCCs are
    calculated per frame, then their means and standard deviations are joined.

    Raises:
        ValueError: If the supplied audio is empty, corrupt, or cannot be read.
    """
    try:
        # Uploaded files can have been read already; reset them before librosa
        # asks soundfile/audioread to decode their contents.
        if hasattr(file_path_or_file_object, "seek"):
            file_path_or_file_object.seek(0)

        audio, _ = librosa.load(file_path_or_file_object, sr=16_000, mono=True)
    except Exception as exc:
        raise ValueError(
            "Unable to read this WAV file. Please upload a valid, non-empty WAV recording."
        ) from exc

    if audio is None or audio.size == 0:
        raise ValueError("The audio file is empty or contains no readable samples.")
    if not np.isfinite(audio).all():
        raise ValueError("The audio file contains invalid sample values.")

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=16_000, n_mfcc=13)
    except Exception as exc:
        raise ValueError("Could not extract MFCC features from this audio file.") from exc

    if mfccs.shape[1] == 0 or not np.isfinite(mfccs).all():
        raise ValueError("The audio file does not contain usable speech features.")

    features = np.concatenate((mfccs.mean(axis=1), mfccs.std(axis=1))).astype(np.float64)
    if features.shape != (26,):
        raise ValueError("Feature extraction did not produce the expected 26 values.")
    return features
