import re

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "i", "you", "he", "she", "we", "they", "of", "to", "and", "in", "on", "for", "with", "at",
    "by", "from", "as", "do", "does", "did", "have", "has", "had", "very", "really", "just",
}
NEGATION_WORDS = {"not", "never", "no", "hardly", "barely"}
WITHOUT_NEGATION_VERBS = {"being", "becoming", "making", "getting", "feeling", "sounding"}
CESSATION_WORDS = {"stop", "stops", "stopped", "cease", "ceases", "ceased"}
CONTRAST_WORDS = {"but", "however", "yet", "though", "although", "whereas", "because"}
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
CLAUSE_BREAK_RE = re.compile(r"[.!?;,]+")

CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
}


def _normalize(text: str) -> str:
    lowered = text.lower()
    for contraction, expanded in CONTRACTIONS.items():
        lowered = lowered.replace(contraction, expanded)
    return lowered


def _tokenize_clause(clause: str) -> list[str]:
    raw_words = WORD_RE.findall(clause)
    tokens: list[str] = []
    negate_left = 0

    for index, word in enumerate(raw_words):
        if word in CONTRAST_WORDS:
            negate_left = 0
            continue

        if word == "without":
            following = raw_words[index + 1] if index + 1 < len(raw_words) else ""
            if following in WITHOUT_NEGATION_VERBS:
                negate_left = 4
            continue

        if word in NEGATION_WORDS or word in CESSATION_WORDS:
            negate_left = 4
            continue

        if word in STOP_WORDS:
            continue

        token = f"not-{word}" if negate_left else word
        tokens.append(token)

        if negate_left:
            negate_left -= 1

    return tokens


def tokenize(text: str) -> list[str]:
    """Tokenize English text and keep short negation scope in the token names."""
    normalized = _normalize(text)
    clauses = CLAUSE_BREAK_RE.split(normalized)
    tokens: list[str] = []
    for clause in clauses:
        tokens.extend(_tokenize_clause(clause))
    return tokens
