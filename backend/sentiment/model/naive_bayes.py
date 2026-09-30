import math

from backend.core.config import TOP_WORDS_COUNT
from backend.sentiment.preprocessing.tokenizer import tokenize

NEGATION_PREFIX = "not-"


class NaiveBayesModel:
    def __init__(self, model_data):
        self.pos_counts = model_data["pos_counts"]
        self.neg_counts = model_data["neg_counts"]
        self.pos_total = model_data["pos_total"]
        self.neg_total = model_data["neg_total"]
        self.vocab = set(model_data["vocab"])
        self.word_score = model_data["word_score"]
        self.top_positive = [tuple(item) for item in model_data["top_positive"]]
        self.top_negative = [tuple(item) for item in model_data["top_negative"]]
        self.prior_positive = model_data["prior_positive"]
        self.prior_negative = model_data["prior_negative"]
        self.temperature = max(float(model_data.get("temperature", 1.0)), 0.1)

    def predict(self, text):
        pos_score, neg_score, explanation = self.score(text)
        known_count = explanation["known_count"]

        if known_count == 0:
            label = "unknown"
            confidence = 0.0
        else:
            label = "positive" if pos_score > neg_score else "negative"
            confidence = self._calibrated_confidence(pos_score, neg_score)

        explanation.pop("known_count", None)
        return label, confidence, explanation

    def score(self, text):
        tokens = tokenize(text)
        vocab_size = max(len(self.vocab), 1)
        pos_score = math.log(self.prior_positive)
        neg_score = math.log(self.prior_negative)
        known_count = 0
        important_words = []
        unknown_words = []
        sentence_details = []

        for token in tokens:
            p_pos, p_neg, word_score, inferred_negation = self._word_probabilities(token, vocab_size)

            if p_pos is None:
                unknown_words.append(token)
                effect = "Unknown"
                detail_pos = detail_neg = 0
            else:
                known_count += 1
                pos_score += math.log(p_pos)
                neg_score += math.log(p_neg)
                effect = "Positive" if word_score >= 0 else "Negative"
                detail_pos, detail_neg = p_pos, p_neg

                if token in self.word_score or inferred_negation:
                    important_words.append((token, word_score))

            sentence_details.append((
                token,
                round(detail_pos, 6),
                round(detail_neg, 6),
                round(word_score, 3),
                effect,
            ))

        important_words.sort(key=lambda item: abs(item[1]), reverse=True)
        important_words = self._unique_words(important_words)[:TOP_WORDS_COUNT]

        explanation = {
            "positive_score": round(pos_score, 3),
            "negative_score": round(neg_score, 3),
            "important_words": important_words,
            "unknown_words": sorted(set(unknown_words)),
            "sentence_details": sentence_details,
            "known_count": known_count,
        }
        return pos_score, neg_score, explanation

    def margin(self, text):
        pos_score, neg_score, explanation = self.score(text)
        return pos_score - neg_score, explanation["known_count"]

    def _word_probabilities(self, token, vocab_size):
        if token in self.vocab:
            p_pos = (self.pos_counts.get(token, 0) + 1) / (self.pos_total + vocab_size)
            p_neg = (self.neg_counts.get(token, 0) + 1) / (self.neg_total + vocab_size)
            return p_pos, p_neg, self.word_score.get(token, 0), False

        if token.startswith(NEGATION_PREFIX):
            base_word = token[len(NEGATION_PREFIX):]
            if base_word in self.vocab:
                base_pos = (self.pos_counts.get(base_word, 0) + 1) / (self.pos_total + vocab_size)
                base_neg = (self.neg_counts.get(base_word, 0) + 1) / (self.neg_total + vocab_size)
                return base_neg, base_pos, -self.word_score.get(base_word, 0), True

        return None, None, 0, False

    @staticmethod
    def _unique_words(items):
        seen = set()
        result = []
        for word, score in items:
            if word not in seen:
                result.append((word, score))
                seen.add(word)
        return result

    def _calibrated_confidence(self, pos_score, neg_score):
        margin = abs(pos_score - neg_score) / self.temperature
        probability = 1 / (1 + math.exp(-min(margin, 50)))
        return round(probability * 100, 2)
