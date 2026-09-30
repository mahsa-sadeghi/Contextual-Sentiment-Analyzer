import csv

from backend.core.config import AMBIGUITY_TEST_PATH, BENCHMARK_PATH
from backend.sentiment.training.dataset import records
from backend.sentiment.training.split_data import calibration_records, test_records, train_records

MAIN_TOPICS = {
    "sports", "politics", "technology", "business", "entertainment",
    "health", "science", "education", "travel",
}


def test_dataset_has_expected_schema():
    required = {"text", "topic", "target", "sentiment", "emotion", "split"}
    assert records
    assert all(required.issubset(row) for row in records)


def test_dataset_is_balanced_for_sentiment():
    positives = sum(row["sentiment"] == "positive" for row in records)
    negatives = sum(row["sentiment"] == "negative" for row in records)
    assert positives == negatives == 90


def test_dataset_covers_all_main_topics():
    assert {row["topic"] for row in records} == MAIN_TOPICS
    assert all(sum(row["topic"] == topic for row in records) == 20 for topic in MAIN_TOPICS)


def test_dataset_splits_are_disjoint():
    train_texts = {row["text"] for row in train_records}
    calibration_texts = {row["text"] for row in calibration_records}
    test_texts = {row["text"] for row in test_records}

    assert len(train_records) == 90
    assert len(calibration_records) == 36
    assert len(test_records) == 54
    assert train_texts.isdisjoint(calibration_texts)
    assert train_texts.isdisjoint(test_texts)
    assert calibration_texts.isdisjoint(test_texts)


def test_benchmark_is_separate_from_training_and_calibration():
    with BENCHMARK_PATH.open(encoding="utf-8", newline="") as file:
        benchmark = list(csv.DictReader(file))

    project_texts = {row["text"] for row in records}
    benchmark_texts = {row["text"] for row in benchmark}

    assert len(benchmark) == 90
    assert {row["topic"] for row in benchmark} == MAIN_TOPICS
    assert len(benchmark_texts) == len(benchmark)
    assert project_texts.isdisjoint(benchmark_texts)


def test_ambiguity_test_is_balanced():
    with AMBIGUITY_TEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    labels = [row["is_ambiguous"] for row in rows]
    assert len(rows) == 30
    assert labels.count("true") == 15
    assert labels.count("false") == 15
