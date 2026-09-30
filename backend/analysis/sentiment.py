from backend.sentiment.model.loader import model
from backend.sentiment.preprocessing.tokenizer import tokenize

POSITIVE_WORDS = {
    "love", "loved", "like", "liked", "enjoy", "enjoyed", "good", "great", "excellent", "amazing",
    "wonderful", "fantastic", "brilliant", "impressive", "helpful", "useful", "clear", "easy", "fast",
    "kind", "clean", "beautiful", "thoughtful", "promising", "organized", "thrilling", "exciting", "better",
    "best", "success", "successful", "reliable", "smooth", "friendly", "recommend", "perfect", "strong", "happy",
    "easier", "impressed", "clearly", "solved", "comfortable", "relaxing", "concise", "appreciated",
    "improvement", "straightforward", "convincing", "funny", "carefully", "stable", "efficiently", "saved",
    "reassuring", "transparent", "effective", "believable", "quiet", "calm", "calmly",
}

NEGATIVE_WORDS = {
    "hate", "hated", "dislike", "disliked", "bad", "terrible", "awful", "horrible", "worst", "boring",
    "broken", "slow", "poor", "difficult", "confusing", "crash", "crashes", "delay", "delayed", "overpriced",
    "ruined", "dismissive", "disappointed", "disappointing", "dirty", "angry", "disgusting", "weak", "failure",
    "failed", "unreliable", "annoying", "frustrating", "worse", "sad", "scared", "confusion", "overheats",
    "drains", "bugs", "unreasonable", "expensive", "dull", "unconvincing", "unanswered", "incomplete",
    "stressful", "painfully", "irritating", "mess", "broke", "freezes", "tedious", "exhausting", "cluttered",
    "poorly", "unspecified", "late", "waste", "wasted", "distracting", "crowded", "outdated", "missing",
}


class SentimentAnalyzer:
    def analyze(self, text: str) -> dict:
        nb_label, nb_confidence, explanation = model.predict(text)
        lexical = self._lexical_score(text)

        label = nb_label
        confidence = float(nb_confidence)

        if lexical["score"]:
            lexical_label = "positive" if lexical["score"] > 0 else "negative"
            strength = min(abs(lexical["score"]), 3)

            if nb_label == "unknown":
                label = lexical_label
                confidence = 55.0 + strength * 5.0
            elif nb_label == lexical_label:
                confidence = min(95.0, confidence + strength * 1.5)
            elif confidence < 70 or strength >= 2:
                label = lexical_label
                confidence = 58.0 + strength * 5.0
            else:
                # Conflicting evidence should lower certainty instead of producing a 90%+ answer.
                confidence = min(confidence, 65.0)

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "positive_score": explanation["positive_score"],
            "negative_score": explanation["negative_score"],
            "important_words": explanation["important_words"],
            "unknown_words": explanation["unknown_words"],
            "lexical_evidence": lexical["evidence"],
        }

    @staticmethod
    def _lexical_score(text: str) -> dict:
        score = 0
        evidence = []

        for token in tokenize(text):
            negated = token.startswith("not-")
            word = token[4:] if negated else token

            value = 0
            if word in POSITIVE_WORDS:
                value = 1
            elif word in NEGATIVE_WORDS:
                value = -1

            if value:
                if negated:
                    value *= -1
                score += value
                evidence.append({"word": token, "value": value})

        return {"score": score, "evidence": evidence}
