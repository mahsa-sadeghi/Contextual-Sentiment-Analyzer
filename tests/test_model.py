from backend.sentiment.model.loader import model

def test_model_loaded():
    assert model is not None

def test_model_has_vocabulary():
    assert len(model.vocab) > 0

def test_model_has_priors():
    assert model.prior_positive > 0
    assert model.prior_negative > 0

def test_model_uses_calibration_temperature():
    assert model.temperature > 0
