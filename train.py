"""Train the English accent SVM classifier from a local AccentDataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audio_features import extract_mfcc


ACCENT_DIRECTORIES = ("indian_accent", "British_accent", "American_accent")


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read valid WAV files below the expected class directories."""
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []

    for label in ACCENT_DIRECTORIES:
        class_directory = dataset_path / label
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Expected accent directory was not found: {class_directory}")

        wav_files = sorted(path for path in class_directory.rglob("*") if path.suffix.lower() == ".wav")
        for wav_file in wav_files:
            try:
                feature_rows.append(extract_mfcc(wav_file))
                labels.append(label)
            except ValueError as exc:
                print(f"Skipping unreadable file {wav_file}: {exc}")

    if not feature_rows:
        raise ValueError("No valid WAV files were found in the dataset.")
    if len(set(labels)) != len(ACCENT_DIRECTORIES):
        raise ValueError("At least one valid WAV file is required for each accent class.")

    return np.vstack(feature_rows), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the English accent SVM model.")
    parser.add_argument("dataset_path", type=Path, help="Path containing the three accent folders")
    parser.add_argument(
        "--output", type=Path, default=Path("model.pkl"), help="Destination model file (default: model.pkl)"
    )
    args = parser.parse_args()

    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {args.dataset_path}")

    features, labels = load_dataset(args.dataset_path)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
        ]
    )
    model.fit(features, labels)
    joblib.dump(model, args.output)

    print(f"Training samples: {len(labels)}")
    print(f"Classes: {', '.join(model.classes_)}")
    print(f"Model saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
