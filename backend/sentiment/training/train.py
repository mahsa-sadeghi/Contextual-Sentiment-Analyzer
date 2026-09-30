"""Train the local Naive Bayes sentiment model."""
from __future__ import annotations

import json
import math
from pathlib import Path

from backend.core.config import MODEL_PATH
from backend.sentiment.model.naive_bayes import NaiveBayesModel
from backend.sentiment.preprocessing.tokenizer import tokenize
from backend.sentiment.training.dataset import DATASET_VERSION
from backend.sentiment.training.split_data import calibration_data, train_data


def build_model_data(data=None):
    training_data = data or train_data
    pos_counts: dict[str, int] = {}
    neg_counts: dict[str, int] = {}
    pos_total = 0
    neg_total = 0
    vocab: set[str] = set()
    positive_docs = 0
    negative_docs = 0

    for sentence, label in training_data:
        if label == "positive":
            positive_docs += 1
        else:
            negative_docs += 1

        for word in tokenize(sentence):
            vocab.add(word)
            if label == "positive":
                pos_counts[word] = pos_counts.get(word, 0) + 1
                pos_total += 1
            else:
                neg_counts[word] = neg_counts.get(word, 0) + 1
                neg_total += 1

    prior_positive = positive_docs / len(training_data)
    prior_negative = negative_docs / len(training_data)
    word_score: dict[str, float] = {}

    for word in vocab:
        p_pos = (pos_counts.get(word, 0) + 1) / (pos_total + len(vocab))
        p_neg = (neg_counts.get(word, 0) + 1) / (neg_total + len(vocab))
        word_score[word] = math.log(p_pos / p_neg)

    top_positive = sorted(
        [(word, score) for word, score in word_score.items() if score > 0],
        key=lambda item: item[1],
        reverse=True,
    )
    top_negative = sorted(
        [(word, score) for word, score in word_score.items() if score < 0],
        key=lambda item: item[1],
    )

    model_data = {
        "dataset_version": DATASET_VERSION,
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "vocab": sorted(vocab),
        "word_score": word_score,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "prior_positive": prior_positive,
        "prior_negative": prior_negative,
        "temperature": 1.0,
    }
    model_data["temperature"] = fit_temperature(model_data, calibration_data)
    return model_data


def fit_temperature(model_data, calibration_rows):
    """Choose one temperature on the calibration split to reduce overconfidence."""
    if not calibration_rows:
        return 1.0

    model = NaiveBayesModel(model_data)
    margins = []
    for text, label in calibration_rows:
        margin, known_count = model.margin(text)
        if known_count:
            margins.append((margin, 1 if label == "positive" else 0))

    if not margins:
        return 1.0

    best_temperature = 1.0
    best_loss = float("inf")

    # A small deterministic grid is enough for this project-sized model.
    for step in range(10, 161):
        temperature = step / 10
        loss = 0.0
        for margin, actual_positive in margins:
            scaled = max(min(margin / temperature, 50), -50)
            probability = 1 / (1 + math.exp(-scaled))
            probability = min(max(probability, 1e-8), 1 - 1e-8)
            if actual_positive:
                loss -= math.log(probability)
            else:
                loss -= math.log(1 - probability)
        loss /= len(margins)

        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature

    return round(best_temperature, 2)


def train_model(output_path=MODEL_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = build_model_data()
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(model_data, file, ensure_ascii=False, indent=2)
    return model_data


if __name__ == "__main__":
    model = train_model()
    print("Model saved successfully.")
    print("Vocabulary size:", len(model["vocab"]))
    print("Temperature:", model["temperature"])
