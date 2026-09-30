import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from backend.analysis.pipeline import semantic_pipeline
from backend.core.config import AMBIGUITY_TEST_PATH, BENCHMARK_PATH, EVALUATION_REPORT_PATH
from backend.sentiment.model.loader import model
from backend.sentiment.training.dataset import DATASET_VERSION
from backend.sentiment.training.split_data import calibration_records, test_records, train_records
from evaluation.baselines import compare_reference_models
from evaluation.metrics import calibration_metrics, multiclass_metrics


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _normalize_target(value: str) -> str:
    value = value.lower().replace("’s", "").replace("'s", "")
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    return " ".join(value.split())


def _target_scores(expected: str, predicted: list[str]) -> tuple[bool, bool]:
    expected_normalized = _normalize_target(expected)
    predicted_normalized = [_normalize_target(item) for item in predicted if item]

    exact = expected_normalized in predicted_normalized
    partial = exact or any(
        expected_normalized in item or item in expected_normalized
        for item in predicted_normalized
        if item
    )
    return exact, partial


def _confidence_summary(samples: list[dict]) -> dict:
    correct = [item["confidence"] for item in samples if item["actual"] == item["predicted"]]
    wrong = [item["confidence"] for item in samples if item["actual"] != item["predicted"]]
    high_confidence_errors = [
        item for item in samples
        if item["actual"] != item["predicted"] and item["confidence"] >= 80
    ]

    def average(values):
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "average": average([item["confidence"] for item in samples]),
        "average_correct": average(correct),
        "average_wrong": average(wrong),
        "high_confidence_error_count": len(high_confidence_errors),
        "high_confidence_errors": high_confidence_errors,
        "calibration": calibration_metrics(samples),
    }


def _per_topic(rows: list[dict], predictions: list[dict]) -> dict:
    result = {}
    for topic in sorted({row["topic"] for row in rows}):
        pairs = [
            (row, prediction)
            for row, prediction in zip(rows, predictions)
            if row["topic"] == topic
        ]
        total = len(pairs)
        sentiment_correct = sum(
            row["sentiment"] == prediction["sentiment"]["label"]
            for row, prediction in pairs
        )
        topic_correct = sum(
            row["topic"] == prediction["topic"]["label"]
            for row, prediction in pairs
        )
        result[topic] = {
            "samples": total,
            "sentiment_accuracy": round(sentiment_correct / total * 100, 2) if total else 0.0,
            "topic_accuracy": round(topic_correct / total * 100, 2) if total else 0.0,
        }
    return result


def evaluate_rows(rows: list[dict]) -> dict:
    predictions = [semantic_pipeline.analyze(row["text"], use_context=False) for row in rows]

    sentiment_true = [row["sentiment"] for row in rows]
    sentiment_pred = [item["sentiment"]["label"] for item in predictions]
    topic_true = [row["topic"] for row in rows]
    topic_pred = [item["topic"]["label"] for item in predictions]
    emotion_true = [row["emotion"] for row in rows]
    emotion_pred = [item["emotion"]["label"] for item in predictions]

    sentiment = multiclass_metrics(sentiment_true, sentiment_pred)
    topic = multiclass_metrics(topic_true, topic_pred)
    emotion = multiclass_metrics(emotion_true, emotion_pred)
    positive = sentiment["per_class"].get("positive", {})

    samples = []
    exact_matches = 0
    partial_matches = 0

    for row, prediction in zip(rows, predictions):
        predicted_targets = [item["target"] for item in prediction["targets"]]
        exact, partial = _target_scores(row["target"], predicted_targets)
        exact_matches += int(exact)
        partial_matches += int(partial)

        samples.append({
            "text": row["text"],
            "topic": row["topic"],
            "actual": row["sentiment"],
            "predicted": prediction["sentiment"]["label"],
            "confidence": float(prediction["sentiment"]["confidence"]),
            "expected_target": row["target"],
            "predicted_targets": predicted_targets,
            "target_exact": exact,
            "target_partial": partial,
        })

    total = len(rows)
    targets = {
        "samples": total,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "exact_accuracy": round(exact_matches / total * 100, 2) if total else 0.0,
        "partial_accuracy": round(partial_matches / total * 100, 2) if total else 0.0,
    }

    return {
        "samples": total,
        "sentiment": sentiment,
        "topic": topic,
        "emotion": emotion,
        "targets": targets,
        "per_topic": _per_topic(rows, predictions),
        "confidence": _confidence_summary(samples),
        "misclassified": [item for item in samples if item["actual"] != item["predicted"]],
        "positive_precision": positive.get("precision", 0.0),
        "positive_recall": positive.get("recall", 0.0),
        "positive_f1": positive.get("f1", 0.0),
    }


def evaluate_ambiguity() -> dict:
    rows = _load_csv(AMBIGUITY_TEST_PATH)
    y_true = []
    y_pred = []
    details = []

    for row in rows:
        actual = row["is_ambiguous"].strip().lower() == "true"
        result = semantic_pipeline.analyze(row["text"], use_context=False)["ambiguity"]
        predicted = bool(result["is_ambiguous"])

        actual_label = "ambiguous" if actual else "clear"
        predicted_label = "ambiguous" if predicted else "clear"
        y_true.append(actual_label)
        y_pred.append(predicted_label)

        if actual != predicted:
            details.append({
                "text": row["text"],
                "actual": actual_label,
                "predicted": predicted_label,
                "reasons": result.get("reasons", []),
            })

    metrics = multiclass_metrics(y_true, y_pred)
    ambiguous = metrics["per_class"].get("ambiguous", {})
    return {
        **metrics,
        "precision": ambiguous.get("precision", 0.0),
        "recall": ambiguous.get("recall", 0.0),
        "f1": ambiguous.get("f1", 0.0),
        "misclassified": details,
    }


def run_evaluation() -> dict:
    local = evaluate_rows(test_records)
    benchmark_rows = _load_csv(BENCHMARK_PATH)
    benchmark = evaluate_rows(benchmark_rows)
    ambiguity = evaluate_ambiguity()
    reference_models = compare_reference_models(train_records, benchmark_rows)

    local_sentiment = local["sentiment"]
    matrix = local_sentiment["confusion_matrix"]
    tp = matrix.get("positive", {}).get("positive", 0)
    tn = matrix.get("negative", {}).get("negative", 0)
    fp = matrix.get("negative", {}).get("positive", 0)
    fn = matrix.get("positive", {}).get("negative", 0) + matrix.get("positive", {}).get("unknown", 0)

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "accuracy": local_sentiment["accuracy"],
        "precision": local["positive_precision"],
        "recall": local["positive_recall"],
        "f1_score": local["positive_f1"],
        "macro_precision": local_sentiment["macro_precision"],
        "macro_recall": local_sentiment["macro_recall"],
        "macro_f1": local_sentiment["macro_f1"],
        "average_confidence": local["confidence"]["average"],
        "calibration_temperature": model.temperature,
        "confusion_matrix": local_sentiment["confusion_matrix"],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "test_samples": local["samples"],
        "correct": local_sentiment["correct"],
        "wrong": local_sentiment["wrong"],
        "local": local,
        "benchmark": benchmark,
        "ambiguity": ambiguity,
        "reference_models": reference_models,
        "topic": local["topic"],
        "emotion": local["emotion"],
        "targets": local["targets"],
        "dataset": {
            "version": DATASET_VERSION,
            "train_samples": len(train_records),
            "calibration_samples": len(calibration_records),
            "test_samples": len(test_records),
            "benchmark_samples": len(benchmark_rows),
            "ambiguity_samples": ambiguity["samples"],
        },
        "benchmark_note": (
            "The benchmark is kept separate from training and calibration. It contains 90 hand-written English "
            "sentences with different wording and harder negation/compositional cases. It is larger and stricter "
            "than the local test split, but it is still a project benchmark rather than a public research dataset."
        ),
    }

    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVALUATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    report = run_evaluation()
    print("Evaluation saved to", EVALUATION_REPORT_PATH)
    print("Local sentiment accuracy:", report["accuracy"], "%")
    print("Benchmark sentiment accuracy:", report["benchmark"]["sentiment"]["accuracy"], "%")
    print("Benchmark topic accuracy:", report["benchmark"]["topic"]["accuracy"], "%")
    print("Benchmark ECE:", report["benchmark"]["confidence"]["calibration"]["ece"], "%")
    print("Ambiguity F1:", report["ambiguity"]["f1"], "%")
    print("Target exact match:", report["targets"]["exact_accuracy"], "%")
    print("Target partial match:", report["targets"]["partial_accuracy"], "%")
