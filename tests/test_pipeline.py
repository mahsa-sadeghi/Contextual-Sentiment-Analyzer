from backend.analysis.pipeline import SemanticAnalysisPipeline, semantic_pipeline


def test_result_contract():
    result = semantic_pipeline.analyze(
        "I hate the World Cup coverage but I love club football.",
        use_context=False,
    )

    assert result["topic"]["label"] == "sports"
    assert result["sentiment"]["label"] in {"positive", "negative", "unknown"}
    assert isinstance(result["keywords"], list)
    assert isinstance(result["targets"], list)
    assert "ambiguity" in result
    assert "summary" in result

    targets = {item["target"]: item for item in result["targets"]}
    world_cup = next(item for name, item in targets.items() if "world cup" in name)
    club_football = next(item for name, item in targets.items() if "club football" in name)

    assert world_cup["sentiment"] == "negative"
    assert club_football["sentiment"] == "positive"


def test_unknown_is_not_forced_to_negative():
    result = semantic_pipeline.analyze("qwertyuiop", use_context=False)
    assert result["sentiment"]["label"] == "unknown"


def test_ambiguity_gate_detects_unresolved_reference():
    result = semantic_pipeline.analyze("It was terrible.", use_context=False)
    assert result["ambiguity"]["is_ambiguous"] is True
    assert result["context_retrieval"]["status"] == "disabled"


class FakeContextRetriever:
    def resolve(self, text, query, reasons):
        return {
            "status": "context_found",
            "resolution": "Possible context: Example",
            "sources": [{"title": "Example", "snippet": "Context", "url": "https://example.test", "score": 1.0}],
        }


def test_context_retrieval_runs_only_after_ambiguity_gate():
    pipeline = SemanticAnalysisPipeline(context_retriever=FakeContextRetriever())
    result = pipeline.analyze("It was terrible.", use_context=True)

    assert result["ambiguity"]["is_ambiguous"] is True
    assert result["context_retrieval"]["status"] == "context_found"


def test_deictic_reference_is_ambiguous():
    result = semantic_pipeline.analyze("That one is terrible.", use_context=False)

    assert result["ambiguity"]["is_ambiguous"] is True
    assert "underspecified reference" in result["ambiguity"]["reasons"]


def test_keywords_are_separate_from_opinion_words():
    result = semantic_pipeline.analyze(
        "I hate the World Cup coverage but I love club football.",
        use_context=False,
    )

    keywords = {item["text"] for item in result["keywords"]}
    opinions = {item["text"] for item in result["opinion_terms"]}

    assert "world cup coverage" in keywords
    assert "club football" in keywords
    assert "hate" not in keywords
    assert {"hate", "love"}.issubset(opinions)


def test_clear_lexical_signal_can_correct_low_confidence_model():
    result = semantic_pipeline.sentiment.analyze("Customer support solved my problem in minutes.")
    assert result["label"] == "positive"


def test_unseen_router_sentence_uses_technology_topic():
    result = semantic_pipeline.analyze(
        "The new router is excellent but its setup guide is confusing.",
        use_context=False,
    )
    assert result["topic"]["label"] == "technology"


def test_terrible_is_recognized_as_strong_negative_emotion():
    result = semantic_pipeline.analyze(
        "The professor was helpful, although the exam instructions were terrible.",
        use_context=False,
    )
    targets = {item["target"]: item for item in result["targets"]}
    exam = next(item for name, item in targets.items() if "exam instructions" in name)
    assert exam["sentiment"] == "negative"
    assert exam["emotion"] == "disgust"


def test_plural_demonstrative_without_context_is_ambiguous():
    result = semantic_pipeline.analyze("Those were awful.", use_context=False)
    assert result["ambiguity"]["is_ambiguous"] is True


def test_topic_can_use_terms_from_training_targets():
    result = semantic_pipeline.topic.classify("The academy needs better facilities.")
    assert result["label"] == "sports"


def test_long_negation_scope_is_handled():
    result = semantic_pipeline.analyze(
        "The backup tool is useful without being difficult to configure.",
        use_context=False,
    )
    assert result["sentiment"]["label"] == "positive"


def test_stops_being_funny_is_negative():
    result = semantic_pipeline.analyze(
        "The final episode stops being funny after ten minutes.",
        use_context=False,
    )
    assert result["sentiment"]["label"] == "negative"


def test_non_english_input_is_reported_as_unsupported():
    result = semantic_pipeline.analyze("این برنامه خیلی خوبه", use_context=False)
    assert result["error"] == "unsupported_language"
    assert result["language"]["supported"] is False


def test_subject_style_target_extraction_is_more_precise():
    result = semantic_pipeline.analyze(
        "The new browser extension stops working after a few minutes.",
        use_context=False,
    )
    assert any(item["target"] == "browser extension" for item in result["targets"])
