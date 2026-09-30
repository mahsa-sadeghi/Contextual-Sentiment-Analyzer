from collections import defaultdict


def multiclass_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    total = len(y_true)
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in labels if actual != label)
        false_negative = sum(matrix[label][predicted] for predicted in labels if predicted != label)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[label] = {
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": sum(matrix[label].values()),
        }

    return {
        "accuracy": round((correct / total * 100) if total else 0.0, 2),
        "macro_precision": round((sum(precisions) / len(precisions) * 100) if precisions else 0.0, 2),
        "macro_recall": round((sum(recalls) / len(recalls) * 100) if recalls else 0.0, 2),
        "macro_f1": round((sum(f1_scores) / len(f1_scores) * 100) if f1_scores else 0.0, 2),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "samples": total,
        "correct": correct,
        "wrong": total - correct,
    }


def calibration_metrics(samples: list[dict], bin_size: int = 10) -> dict:
    if not samples:
        return {"ece": 0.0, "brier": 0.0, "bins": []}

    brier = 0.0
    bins = defaultdict(list)

    for item in samples:
        probability = min(max(float(item["confidence"]) / 100, 0.0), 1.0)
        correct = 1.0 if item["actual"] == item["predicted"] else 0.0
        brier += (probability - correct) ** 2
        bucket = min(int(item["confidence"] // bin_size) * bin_size, 90)
        bins[bucket].append((probability, correct))

    ece = 0.0
    reliability = []
    total = len(samples)

    for bucket in sorted(bins):
        values = bins[bucket]
        average_confidence = sum(probability for probability, _ in values) / len(values)
        accuracy = sum(correct for _, correct in values) / len(values)
        ece += len(values) / total * abs(average_confidence - accuracy)
        reliability.append({
            "range": f"{bucket}-{bucket + bin_size}",
            "samples": len(values),
            "average_confidence": round(average_confidence * 100, 2),
            "accuracy": round(accuracy * 100, 2),
        })

    return {
        "ece": round(ece * 100, 2),
        "brier": round(brier / total, 4),
        "bins": reliability,
    }
