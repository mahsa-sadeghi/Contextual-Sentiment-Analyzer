from backend.analysis.pipeline import semantic_pipeline
from backend.core.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM


def confidence_label(confidence):
    if confidence >= CONFIDENCE_HIGH:
        return "High"
    if confidence >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def analyze_sentence():
    text = input("\nEnter text: ").strip()
    result = semantic_pipeline.analyze(text, use_context=False)
    summary = result["summary"]

    print("\nSemantic Analysis")
    print("-----------------")
    print(f"Topic      : {summary['topic']}")
    print(f"Target     : {summary['target'] or 'not resolved'}")
    print(f"Sentiment  : {summary['sentiment']}")
    print(f"Emotion    : {summary['emotion']}")
    print(f"Confidence : {summary['confidence']:.2f}% ({confidence_label(summary['confidence'])})")

    print("\nKeywords")
    for item in result["keywords"][:8]:
        print(f"- {item['text']}: {item['relevance']:.2f}")

    if result["targets"]:
        print("\nTarget-level sentiment")
        for target in result["targets"]:
            print(
                f"- {target['target']} -> {target['sentiment']} / {target['emotion']} "
                f"({target['sentiment_confidence']:.2f}%)"
            )

    ambiguity = result["ambiguity"]
    print(f"\nAmbiguous  : {ambiguity['is_ambiguous']} (score={ambiguity['score']})")
    if ambiguity["reasons"]:
        print("Reasons    : " + "; ".join(ambiguity["reasons"]))
