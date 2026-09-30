from __future__ import annotations

import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*|[\u0600-\u06FF]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?؟])\s+|\n+")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "i", "you", "he", "she", "we", "they", "of", "to", "and", "or", "in", "on", "for", "with",
    "at", "by", "from", "as", "but", "if", "then", "than", "very", "really", "my", "your", "our",
    "their", "its", "do", "does", "did", "have", "has", "had", "about", "into", "over", "after",
    "before", "so", "because", "while", "during", "can", "could", "would", "should", "will", "just",
}


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def ngrams(tokens: list[str], min_n: int = 1, max_n: int = 3) -> list[str]:
    result: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            result.append(" ".join(tokens[i:i+n]))
    return result


def lexical_overlap(a: str, b: str) -> float:
    sa, sb = set(content_words(a)), set(content_words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def frequency_scores(text: str) -> dict[str, float]:
    tokens = content_words(text)
    counts = Counter(tokens)
    total = max(sum(counts.values()), 1)
    return {token: count / total for token, count in counts.items()}
