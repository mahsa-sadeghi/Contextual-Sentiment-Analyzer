from backend.sentiment.pipeline import pipeline


def show_positive_words():
    count = _read_count()
    print("\nTop Positive Words:")
    for word, score in pipeline.get_top_positive_words()[:count]:
        print(word, "->", round(score, 3))


def show_negative_words():
    count = _read_count()
    print("\nTop Negative Words:")
    for word, score in pipeline.get_top_negative_words()[:count]:
        print(word, "->", round(score, 3))


def _read_count():
    try:
        return max(1, int(input("How many words? ")))
    except ValueError:
        print("Invalid number. Showing 10 words.")
        return 10
