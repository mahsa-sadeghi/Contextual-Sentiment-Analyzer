import re

from backend.analysis.emotion import EmotionAnalyzer
from backend.analysis.keywords import OPINION_TERMS
from backend.analysis.sentiment import NEGATIVE_WORDS, POSITIVE_WORDS, SentimentAnalyzer
from backend.analysis.text import STOPWORDS, words
from backend.sentiment.training.split_data import train_records

CLAUSE_SPLIT = re.compile(
    r"\s*(?:[,;]|\bbut\b|\bhowever\b|\bwhile\b|\bwhereas\b|\byet\b|\balthough\b|\bthough\b)\s*",
    re.I,
)
OPINION_VERBS = {
    "love", "loved", "like", "liked", "enjoy", "enjoyed", "hate", "hated", "dislike", "disliked",
    "admire", "admired", "prefer", "preferred", "recommend", "recommended",
}
COPULAS = {"is", "are", "was", "were", "feel", "feels", "felt", "seem", "seems", "look", "looks"}
GENERIC_TARGETS = {"thing", "things", "something", "anything", "everything", "it", "this", "that", "they", "one"}
SUBJECT_WORDS = {"i", "we", "you", "he", "she", "they", "my", "our", "your", "their"}
SENTIMENT_WORDS = POSITIVE_WORDS | NEGATIVE_WORDS | OPINION_TERMS
TRAILING_FILLERS = {"today", "again", "anymore", "now", "before", "overall", "really", "much", "never", "not"}
SUBJECT_MODIFIERS = {"new", "latest", "several", "some", "slower", "quiet", "local", "our", "my", "your", "its", "their"}
SUBJECT_VERBS = COPULAS | {
    "kept", "keeps", "made", "makes", "wakes", "broke", "breaks", "freezes", "appeared", "appears",
    "repeats", "gives", "took", "takes", "produced", "produces", "explains", "explained", "leaves", "logs",
    "arrived", "added", "works", "stops", "stopped", "handled", "matched", "address", "addresses",
    "answered", "answers", "drains", "becomes", "gets", "felt", "feels", "seems", "looked", "looks",
}
PREPOSITION_CUT = {"in", "on", "at", "for", "with", "without", "after", "before", "during", "outside", "inside", "from", "into", "because", "when", "once", "until"}

KNOWN_TARGETS = sorted(
    {row["target"] for row in train_records if row["target"]},
    key=lambda value: (-len(value.split()), -len(value)),
)


class TargetAnalyzer:
    def __init__(self, sentiment: SentimentAnalyzer, emotion: EmotionAnalyzer):
        self.sentiment = sentiment
        self.emotion = emotion

    def analyze(self, text: str, keyword_items: list[dict], opinion_terms: list[dict]) -> list[dict]:
        results = []

        for target, clause in self.extract_targets(text, keyword_items):
            sentiment = self.sentiment.analyze(clause)
            emotion = self.emotion.analyze(clause)
            local_opinions = [
                item["text"]
                for item in opinion_terms
                if re.search(rf"\b{re.escape(item['text'])}\b", clause, re.I)
            ]
            results.append({
                "target": target,
                "sentiment": sentiment["label"],
                "sentiment_confidence": sentiment["confidence"],
                "emotion": emotion["label"],
                "emotion_confidence": emotion["confidence"],
                "opinion_terms": local_opinions,
                "context": clause,
            })

        return results

    def extract_targets(self, text: str, keyword_items: list[dict], max_targets: int = 4) -> list[tuple[str, str]]:
        clauses = [part.strip(" .!?") for part in CLAUSE_SPLIT.split(text) if part.strip(" .!?")]
        found: list[tuple[str, str]] = []

        for clause in clauses:
            targets = self._targets_from_rules(clause)
            if not targets:
                targets = self._known_targets(clause)
            if not targets:
                targets = self._targets_from_keywords(clause, keyword_items)

            for target in targets:
                if target not in [existing for existing, _ in found]:
                    found.append((target, clause))
                if len(found) >= max_targets:
                    return found

        return found

    @staticmethod
    def _known_targets(clause: str) -> list[str]:
        lower = clause.lower()
        matches = []
        for target in KNOWN_TARGETS:
            if re.search(rf"\b{re.escape(target)}\b", lower):
                if not any(target in existing or existing in target for existing in matches):
                    matches.append(target)
            if len(matches) == 2:
                break
        return matches

    def _targets_from_rules(self, clause: str) -> list[str]:
        tokens = words(clause)
        targets = []

        subject = self._subject_target(tokens)
        if subject:
            targets.append(subject)

        for index, token in enumerate(tokens):
            if token in OPINION_VERBS:
                target = self._clean_target(tokens[index + 1:])
                if target:
                    targets.append(target)

            if token in COPULAS and index > 0:
                right_side = tokens[index + 1:index + 5]
                if any(word in SENTIMENT_WORDS for word in right_side):
                    target = self._clean_target(tokens[:index])
                    if target:
                        targets.append(target)

        return list(dict.fromkeys(targets))

    @staticmethod
    def _subject_target(tokens: list[str]) -> str | None:
        boundary = next((index for index, token in enumerate(tokens) if token in SUBJECT_VERBS), None)
        if boundary is None or boundary == 0:
            return None

        candidate = tokens[:boundary]
        if candidate and candidate[0] in SUBJECT_WORDS:
            candidate = candidate[1:]

        for index, token in enumerate(candidate):
            if token in PREPOSITION_CUT:
                candidate = candidate[:index]
                break

        cleaned = [
            token for token in candidate
            if token not in STOPWORDS
            and token not in SUBJECT_WORDS
            and token not in SUBJECT_MODIFIERS
            and token not in TRAILING_FILLERS
        ]
        if not cleaned:
            return None
        target = " ".join(cleaned[-4:])
        return None if target in GENERIC_TARGETS else target

    def _targets_from_keywords(self, clause: str, keyword_items: list[dict]) -> list[str]:
        lower = clause.lower()
        targets = []

        for item in keyword_items:
            phrase = item["text"].lower()
            if phrase not in lower:
                continue
            target = self._clean_target(phrase.split())
            if target:
                targets.append(target)
            if len(targets) == 2:
                break

        return targets

    @staticmethod
    def _clean_target(tokens: list[str]) -> str | None:
        candidate = []
        for token in tokens:
            if token in PREPOSITION_CUT:
                break
            candidate.append(token)

        cleaned = []
        for token in candidate:
            if token in STOPWORDS or token in OPINION_TERMS or token in SUBJECT_WORDS:
                continue
            if token in SENTIMENT_WORDS or token in TRAILING_FILLERS or token in SUBJECT_MODIFIERS:
                continue
            cleaned.append(token)

        if not cleaned:
            return None

        target = " ".join(cleaned[:5]).strip()
        if target in GENERIC_TARGETS:
            return None
        return target
