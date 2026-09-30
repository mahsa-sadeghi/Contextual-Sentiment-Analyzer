from backend.sentiment.training.split_data import test_records
from evaluation.evaluate import evaluate_ambiguity, evaluate_rows, run_evaluation


def test_evaluation_reports_strict_and_partial_target_scores():
    result = evaluate_rows(test_records[:6])
    targets = result["targets"]
    assert 0 <= targets["exact_accuracy"] <= 100
    assert 0 <= targets["partial_accuracy"] <= 100
    assert targets["partial_accuracy"] >= targets["exact_accuracy"]


def test_evaluation_includes_confidence_calibration():
    result = evaluate_rows(test_records)
    calibration = result["confidence"]["calibration"]
    assert 0 <= calibration["ece"] <= 100
    assert 0 <= calibration["brier"] <= 1
    assert isinstance(calibration["bins"], list)


def test_ambiguity_evaluation_has_binary_metrics():
    result = evaluate_ambiguity()
    assert result["samples"] == 30
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_full_report_contains_benchmark_and_reference_models():
    report = run_evaluation()
    assert report["dataset"]["benchmark_samples"] == 90
    assert "tfidf_logistic_regression" in report["reference_models"]
    assert "tfidf_linear_svm" in report["reference_models"]


def test_benchmark_regression_thresholds():
    report = run_evaluation()
    benchmark = report["benchmark"]
    assert benchmark["sentiment"]["accuracy"] >= 75
    assert benchmark["emotion"]["accuracy"] >= 75
    assert benchmark["targets"]["exact_accuracy"] >= 70
    assert benchmark["confidence"]["high_confidence_error_count"] <= 2
