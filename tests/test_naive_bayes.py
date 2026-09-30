from backend.sentiment.model.loader import model


def test_prediction_output():
    result, confidence, explanation = model.predict(
        "this movie is amazing"
    )

    assert result in [
        "positive",
        "negative"
    ]

    assert confidence >= 0
    assert confidence <= 100

    assert isinstance(
        explanation,
        dict
    )


def test_prediction_details():
    _, _, explanation = model.predict(
        "bad movie"
    )

    assert "positive_score" in explanation
    assert "negative_score" in explanation
    assert "important_words" in explanation
    assert "unknown_words" in explanation
    assert "sentence_details" in explanation

def test_unseen_negation_uses_base_word_sentiment():
    result, confidence, explanation = model.predict(
        "I don't love this movie"
    )

    assert result == "negative"

def test_log_input_does_not_crash():
    text = "amazing " * 1000
    result, confidence, explanation = model.predict( text)
    assert result in ["positive", "negative"]
    assert 0 <= confidence <= 100


def test_empty_input_returns_unknown():
    result, confidence, explanation = model.predict("")

    assert result == "unknown"
    assert confidence == 0

def test_oov_input_returns_unknown():
    result, confidence, explanation = model.predict("qwertyuiop")

    assert result == "unknown"
    assert confidence == 0
    assert "qwertyuiop" in explanation["unknown_words"]

def test_oov_word_effect_is_unknown():
    result, confidence, explanation = model.predict("qwertyuiop")
    effect = explanation["sentence_details"][0][4]

    assert effect == "Unknown"



def test_never_negation_can_use_base_word_sentiment():
    result, confidence, _ = model.predict("I never love this movie")
    assert result == "negative"
