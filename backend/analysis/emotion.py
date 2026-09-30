from collections import Counter

from backend.sentiment.preprocessing.tokenizer import tokenize

EMOTION_WORDS = {
    "joy": {
        "happy", "love", "loved", "enjoy", "enjoyed", "wonderful", "great", "amazing", "fantastic",
        "comfortable", "relaxing", "funny", "easier", "beautifully", "instantly", "quiet", "clean", "stable",
    },
    "admiration": {
        "admire", "respect", "impressive", "brilliant", "excellent", "inspiring", "helpful", "useful", "clear",
        "clearer", "clearly", "concise", "organized", "thoughtful", "reliable", "friendly", "promising",
        "appreciated", "improvement", "straightforward", "convincing", "carefully", "efficiently", "reassuring",
        "transparent", "effective", "believable", "calm", "calmly", "better", "simple", "composed", "saved",
        "answered", "explained", "practical", "plain",
    },
    "excitement": {"excited", "exciting", "thrilled", "thrilling", "awesome"},
    "anger": {
        "angry", "furious", "rage", "annoyed", "annoying", "outraged", "frustrating", "unreasonable",
        "expensive", "overpriced", "slow", "delayed", "stressful", "painfully", "dismissive", "broken",
        "irritating", "mess", "broke", "freezes", "late", "waste", "wasted", "drains", "charge", "fee",
        "exhausting", "calls",
    },
    "disgust": {"disgust", "disgusting", "gross", "revolting", "hate", "hated", "awful", "terrible", "horrible", "dirty"},
    "sadness": {
        "sad", "depressed", "heartbroken", "miserable", "tragic", "disappointed", "disappointing", "boring",
        "dull", "weak", "poor", "messy", "confusing", "unreliable", "unconvincing", "difficult", "unanswered",
        "incomplete", "tedious", "cluttered", "poorly", "unspecified", "outdated", "missing", "hard", "small",
    },
    "fear": {"afraid", "fear", "scared", "terrified", "worried", "anxious"},
}

NEGATED_EMOTIONS = {
    "clean": "disgust",
    "funny": "sadness",
    "helpful": "sadness",
    "convincing": "sadness",
    "believable": "sadness",
    "effective": "sadness",
    "clear": "sadness",
    "transparent": "sadness",
    "reliable": "sadness",
    "difficult": "admiration",
    "confusing": "admiration",
    "boring": "admiration",
    "weak": "admiration",
    "distracting": "admiration",
    "freezes": "joy",
    "dislike": "joy",
    "cluttered": "joy",
    "working": "anger",
    "answered": "anger",
    "explained": "anger",
    "explains": "anger",
    "matched": "anger",
    "address": "sadness",
    "addresses": "sadness",
}


class EmotionAnalyzer:
    def analyze(self, text: str) -> dict:
        scores = Counter()
        evidence: dict[str, list[str]] = {}

        for token in tokenize(text):
            negated = token.startswith("not-")
            word = token[4:] if negated else token

            if negated and word in NEGATED_EMOTIONS:
                self._add(scores, evidence, NEGATED_EMOTIONS[word], token, 3)
                continue

            if not negated:
                for emotion, terms in EMOTION_WORDS.items():
                    if word in terms:
                        self._add(scores, evidence, emotion, token, 3)

        if not scores:
            return {"label": "neutral", "confidence": 35.0, "evidence": []}

        emotion, score = scores.most_common(1)[0]
        return {
            "label": emotion,
            "confidence": min(85.0, 50.0 + score * 5.0),
            "evidence": evidence.get(emotion, []),
        }

    @staticmethod
    def _add(scores, evidence, emotion, token, weight):
        scores[emotion] += weight
        evidence.setdefault(emotion, []).append(token)
