from backend.sentiment.training.dataset import records_for

train_records = records_for("train")
calibration_records = records_for("calibration")
test_records = records_for("test")

train_data = [(row["text"], row["sentiment"]) for row in train_records]
calibration_data = [(row["text"], row["sentiment"]) for row in calibration_records]
test_data = [(row["text"], row["sentiment"]) for row in test_records]
