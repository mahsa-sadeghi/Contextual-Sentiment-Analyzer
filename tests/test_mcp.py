from backend.mcp.tools import analyze_sentiment


def test_analyze_sentiment():
    result = analyze_sentiment("I love this movie")
    assert result["label"] == "positive"
    assert "confidence" in result
