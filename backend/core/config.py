import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.json"
DATASET_PATH = BASE_DIR / "data" / "sentiment_dataset.csv"
BENCHMARK_PATH = BASE_DIR / "data" / "benchmark.csv"
AMBIGUITY_TEST_PATH = BASE_DIR / "data" / "ambiguity_test.csv"

FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

EVALUATION_DIR = BASE_DIR / "evaluation"
REPORTS_DIR = EVALUATION_DIR / "reports"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"

TOP_WORDS_COUNT = int(os.getenv("TOP_WORDS_COUNT", "8"))
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "80"))
CONFIDENCE_MEDIUM = float(os.getenv("CONFIDENCE_MEDIUM", "60"))

DEFAULT_TOPIC_LABELS = tuple(
    label.strip()
    for label in os.getenv(
        "TOPIC_LABELS",
        "sports,politics,technology,business,entertainment,health,science,education,travel,general",
    ).split(",")
    if label.strip()
)

CONTEXT_LOOKUP_ENABLED = os.getenv("CONTEXT_LOOKUP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
CONTEXT_TOP_K = max(1, min(int(os.getenv("CONTEXT_TOP_K", "2")), 3))
CONTEXT_TIMEOUT_SECONDS = float(os.getenv("CONTEXT_TIMEOUT_SECONDS", "2.5"))
