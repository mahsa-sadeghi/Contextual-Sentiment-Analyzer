from backend.analysis.pipeline import semantic_pipeline


def analyze_text(text: str, use_context: bool = False):
    """Analyze topic, targets, sentiment, emotion, keywords and ambiguity."""
    return semantic_pipeline.analyze(text, use_context=use_context)


def analyze_sentiment(text: str):
    """Return only the sentiment part of the semantic analysis."""
    return semantic_pipeline.sentiment.analyze(text)
