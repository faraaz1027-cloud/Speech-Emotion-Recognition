"""Optional NEW training pipeline; not the fitted model from original coursework.

Usage (locally, after obtaining and licensing the training audio):
    python training/train.py --data-dir /path/to/five-emotion-folders

Each emotion must have its own folder. Label "Neutral" is normalized to
"Natural", matching the team notebook. No training data is bundled.

IMPORTANT: If speakers overlap between train/test, results can overestimate
generalization. Use speaker IDs/group-wise splitting when possible.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from features import extract_features, load_audio  # noqa: E402

LABELS = ("Angry", "Happy", "Natural", "Sad", "Surprised")
EXTENSIONS = {".wav"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing five labeled emotion subfolders")
    parser.add_argument("--manifest", type=Path,
                        help="Optional CSV with filename and speaker_id columns for group split")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models" / "ser_svm_pipeline.joblib")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error("Training data directory not found.")
    speakers = {}
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        if not {"filename", "speaker_id"}.issubset(manifest.columns):
            parser.error("Manifest needs filename and speaker_id columns")
        speakers = dict(zip(manifest["filename"].astype(str),
                            manifest["speaker_id"].astype(str)))

    rows, labels, groups = [], [], []
    for folder in sorted(args.data_dir.iterdir()):
        if not folder.is_dir():
            continue
        emotion = {"Neutral": "Natural"}.get(folder.name, folder.name)
        if emotion not in LABELS:
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() not in EXTENSIONS:
                continue
            try:
                payload = f.read_bytes()
                y, sr = load_audio(payload)
                feats = extract_features(y, sr)
            except (ValueError, OSError) as e:
                print(f"Skipping {f.name}: {e}")
                continue
            rel = str(f.relative_to(args.data_dir))
            rows.append(feats)
            labels.append(emotion)
            groups.append(speakers.get(rel))
    if not rows or set(labels) != set(LABELS):
        parser.error("Need readable audio for all five emotions.")
    counts = Counter(labels)
    if min(counts.values()) < 3:
        parser.error("Need at least three valid examples per emotion to split.")
    X, y = pd.DataFrame(rows), np.array(labels)

    if args.manifest:
        if any(g is None for g in groups):
            parser.error("Manifest does not contain speaker_id for every recording.")
        group_values = np.array(groups)
        if len(set(group_values)) < 6:
            parser.error("Need more distinct speakers for a group-wise evaluation.")
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(split.split(X, y, groups=group_values))
        if set(y[test_idx]) != set(LABELS) or set(y[train_idx]) != set(LABELS):
            parser.error("Group-wise split omitted an emotion; collect more speakers.")
        split_description = "speaker-disjoint holdout; review speaker source metadata"
    else:
        print("WARNING: No speaker manifest supplied. This is a recording-stratified "
              "split and may overestimate speaker-independent accuracy.")
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )
        split_description = "recording-stratified, NOT speaker-disjoint"

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    clf.fit(X.iloc[train_idx], y[train_idx])
    pred = clf.predict(X.iloc[test_idx])
    result = {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
        "rows": len(y),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "split": split_description,
        "classes": clf.classes_.tolist(),
        "classification_report": classification_report(
            y[test_idx], pred, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "warning": "This is a NEW training experiment, not the original CSE424 evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"Saved pipeline to {args.output}")
    print(f"Saved evaluation to {metrics_path}")
    print(f"Holdout accuracy={result['accuracy']:.3f}; macro F1={result['macro_f1']:.3f}")
    print(f"Split: {split_description}")


if __name__ == "__main__":
    main()
