from collections import Counter

from backend.analysis.sentiment import NEGATIVE_WORDS, POSITIVE_WORDS
from backend.analysis.text import STOPWORDS, words

OPINION_TERMS = POSITIVE_WORDS | NEGATIVE_WORDS | {
    "admire", "admired", "prefer", "preferred", "excited", "thrilled", "fear", "afraid", "worried",
}

BOUNDARY_WORDS = {"but", "however", "while", "whereas", "although", "though", "yet", "because"}


class KeywordExtractor:
    def extract(self, text: str, top_n: int = 8) -> dict:
        phrases = self._candidate_phrases(text)
        if not phrases:
            return {"items": []}

        counts = Counter(phrases)
        first_position = {phrase: phrases.index(phrase) for phrase in counts}
        ranked = sorted(
            counts,
            key=lambda phrase: (-counts[phrase], -len(phrase.split()), first_position[phrase]),
        )[:top_n]

        max_count = max(counts.values())
        items = [
            {
                "text": phrase,
                "relevance": round(counts[phrase] / max_count, 2),
            }
            for phrase in ranked
        ]
        return {"items": items}

    def _candidate_phrases(self, text: str) -> list[str]:
        phrases = []
        current = []

        def flush():
            if current:
                phrases.append(" ".join(current[:4]))
                current.clear()

        for token in words(text):
            if token in STOPWORDS or token in OPINION_TERMS or token in BOUNDARY_WORDS:
                flush()
                continue
            current.append(token)
            if len(current) == 4:
                flush()

        flush()
        return [phrase for phrase in phrases if len(phrase) > 1]

    def opinion_terms(self, text: str, limit: int = 8) -> list[dict]:
        result = []
        seen = set()

        for token in words(text):
            if token in seen:
                continue
            if token in POSITIVE_WORDS:
                result.append({"text": token, "polarity": "positive", "score": 1.0})
                seen.add(token)
            elif token in NEGATIVE_WORDS:
                result.append({"text": token, "polarity": "negative", "score": -1.0})
                seen.add(token)

            if len(result) >= limit:
                break

        return result
