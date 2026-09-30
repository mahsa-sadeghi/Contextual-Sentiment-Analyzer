from __future__ import annotations

import json

from backend.core.config import MODEL_PATH
from backend.core.logger import logger
from backend.sentiment.model.naive_bayes import NaiveBayesModel
from backend.sentiment.training.dataset import DATASET_VERSION
from backend.sentiment.training.train import train_model


def _model_needs_training() -> bool:
    if not MODEL_PATH.exists():
        return True

    try:
        with MODEL_PATH.open("r", encoding="utf-8") as file:
            model_data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError):
        return True

    return model_data.get("dataset_version") != DATASET_VERSION


def load_model() -> NaiveBayesModel:
    if _model_needs_training():
        logger.info("Training local sentiment model for the current dataset.")
        train_model(MODEL_PATH)

    logger.info("Loading local Naive Bayes sentiment model...")
    with MODEL_PATH.open("r", encoding="utf-8") as file:
        model_data = json.load(file)
    return NaiveBayesModel(model_data)


model = load_model()
