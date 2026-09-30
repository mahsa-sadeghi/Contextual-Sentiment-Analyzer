from backend.sentiment.preprocessing.tokenizer import tokenize


def test_tokenize_lowercase():
    result = tokenize("I LOVE This Movie")
    assert "love" in result
    assert "movie" in result


def test_remove_stop_words():
    result = tokenize("this is a good movie")
    assert "this" not in result
    assert "is" not in result
    assert "a" not in result


def test_negation_scope_reaches_later_content_word():
    result = tokenize("useful without being difficult to configure")
    assert "useful" in result
    assert "not-difficult" in result


def test_cessation_negates_following_description():
    result = tokenize("stops being funny after ten minutes")
    assert "not-funny" in result


def test_negation_contraction():
    result = tokenize("I don't love this update")
    assert "not-love" in result
    assert "love" not in result


def test_clause_punctuation_resets_negation():
    result = tokenize("not good, useful")
    assert "not-good" in result
    assert "useful" in result
    assert "not-useful" not in result
