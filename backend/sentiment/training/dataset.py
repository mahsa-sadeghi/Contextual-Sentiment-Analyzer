"""Load the local multi-domain English sentiment dataset."""
from __future__ import annotations

import csv
from pathlib import Path

from backend.core.config import DATASET_PATH

DATASET_VERSION = "multidomain-calibrated-extended-2026-09-30"
VALID_SENTIMENTS = {"positive", "negative"}
VALID_SPLITS = {"train", "calibration", "test"}


def load_records(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"text", "topic", "target", "sentiment", "emotion", "split"}
    if not rows:
        raise ValueError("Dataset is empty.")
    if not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    for row in rows:
        for field in required:
            row[field] = row[field].strip()
        row["topic"] = row["topic"].lower()
        row["target"] = row["target"].lower()
        row["sentiment"] = row["sentiment"].lower()
        row["emotion"] = row["emotion"].lower()
        row["split"] = row["split"].lower()

        if row["sentiment"] not in VALID_SENTIMENTS:
            raise ValueError(f"Invalid sentiment label: {row['sentiment']}")
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Invalid split label: {row['split']}")

    return rows


def records_for(split: str) -> list[dict[str, str]]:
    return [row for row in records if row["split"] == split]


records = load_records()
data = [(row["text"], row["sentiment"]) for row in records]
