import re

LATIN_WORD = re.compile(r"[A-Za-z]")
NON_LATIN_LETTER = re.compile(r"[^\x00-\x7F]")


def check_english_support(text: str) -> dict:
    """The analyzer is intentionally English-only; this is a support check, not language detection."""
    has_latin = bool(LATIN_WORD.search(text))
    has_non_ascii = bool(NON_LATIN_LETTER.search(text))
    supported = has_latin and not has_non_ascii
    return {
        "code": "en" if supported else "unsupported",
        "supported": supported,
        "message": "English text is supported." if supported else "Only English text is supported by this analyzer.",
    }
