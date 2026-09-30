import json

from backend.core.config import MODEL_PATH


def test_model_artifact_is_json_not_pickle():
    assert MODEL_PATH.suffix == ".json"
    with MODEL_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    assert "vocab" in data
    assert "temperature" in data
