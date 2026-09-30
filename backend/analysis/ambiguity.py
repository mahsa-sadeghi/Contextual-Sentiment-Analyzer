import re

PERSONAL_REFERENCES = re.compile(r"\b(it|they|them|he|she|him|her)\b", re.I)
DEMONSTRATIVE_PRONOUNS = re.compile(r"\b(this|that|these|those)\b(?!\s+[a-z][a-z-]{2,})", re.I)
DEMONSTRATIVE_COPULA = re.compile(r"\b(this|that|these|those)\s+(is|are|was|were|seems?|feels?)\b", re.I)
DEICTIC_PHRASES = re.compile(
    r"\b(over there|that one|this one|the other one|that part|this part|the thing|what happened|you know)\b",
    re.I,
)


class AmbiguityDetector:
    def detect(self, text: str, sentiment: dict, topic: dict, targets: list[dict]) -> dict:
        reasons = []
        points = 0

        if sentiment.get("label") == "unknown" or sentiment.get("confidence", 0) < 55:
            points += 1
            reasons.append("low sentiment certainty")

        if topic.get("confidence", 0) < 40:
            points += 1
            reasons.append("low topic certainty")

        if DEICTIC_PHRASES.search(text):
            points += 2
            reasons.append("underspecified reference")
        elif PERSONAL_REFERENCES.search(text) or DEMONSTRATIVE_PRONOUNS.search(text) or DEMONSTRATIVE_COPULA.search(text):
            points += 2
            reasons.append("possible unresolved pronoun")

        if not targets:
            points += 1
            reasons.append("no clear sentiment target")

        if len(text.split()) <= 3 and not targets:
            points += 1
            reasons.append("very short text")

        score = min(points / 4, 1.0)
        return {
            "is_ambiguous": points >= 2,
            "score": round(score, 2),
            "confidence": round(score * 100, 1),
            "reasons": reasons,
        }
