from backend.analysis.text import content_words
from backend.core.config import DEFAULT_TOPIC_LABELS
from backend.sentiment.training.split_data import train_records

WEAK_TOPIC_WORDS = {"team", "app", "report", "performance", "room", "guide"}

TOPIC_WORDS = {
    "sports": {
        "sport", "football", "soccer", "basketball", "tennis", "world", "cup", "team", "match",
        "league", "player", "coach", "goal", "fifa", "striker", "goalkeeper", "derby", "referee",
        "tournament", "defending", "supporters",
    },
    "politics": {
        "politics", "government", "election", "president", "parliament", "parliamentary", "minister",
        "policy", "vote", "voting", "voter", "voters", "party", "senate", "campaign", "debate",
        "proposal",
    },
    "technology": {
        "technology", "software", "computer", "ai", "model", "internet", "phone", "app", "code",
        "robot", "data", "laptop", "camera", "backup", "keyboard", "browser", "extension", "device",
        "battery", "release", "bugs", "router",
    },
    "business": {
        "business", "market", "company", "stock", "price", "economy", "bank", "investment", "revenue",
        "customer", "store", "support", "checkout", "delivery", "invoice", "refund", "subscription",
        "billing", "quarterly", "report",
    },
    "entertainment": {
        "movie", "film", "music", "actor", "series", "show", "game", "cinema", "song", "album",
        "concert", "documentary", "soundtrack", "episode", "dialogue", "performance",
    },
    "health": {
        "health", "doctor", "hospital", "medicine", "patient", "disease", "treatment", "fitness", "diet",
        "nurse", "exercise", "clinic", "prescription", "appointment", "procedure",
    },
    "science": {
        "science", "research", "study", "experiment", "physics", "chemistry", "biology", "space", "climate",
        "laboratory", "telescope", "galaxy", "measurement", "sample", "method",
    },
    "education": {
        "school", "university", "student", "teacher", "course", "class", "exam", "education", "learning",
        "professor", "assignment", "lecture", "textbook", "revision", "lesson",
    },
    "travel": {
        "travel", "trip", "hotel", "flight", "city", "country", "tour", "vacation", "airport", "beach",
        "train", "journey", "booking", "room", "bus", "guide", "breakfast",
    },
}


GENERIC_TARGET_WORDS = {
    "new", "old", "first", "last", "process", "system", "page", "instructions",
    "report", "team", "room", "guide", "service", "plan", "website", "tool",
}


def _training_topic_words() -> dict[str, set[str]]:
    learned = {label: set() for label in TOPIC_WORDS}
    for row in train_records:
        topic = row["topic"]
        if topic not in learned:
            continue
        for word in content_words(row["target"]):
            if word not in GENERIC_TARGET_WORDS:
                learned[topic].add(word)
    return learned


TRAINING_TOPIC_WORDS = _training_topic_words()


class TopicLabeler:
    def __init__(self, labels=None):
        self.labels = tuple(labels or DEFAULT_TOPIC_LABELS)

    def classify(self, text: str) -> dict:
        tokens = set(content_words(text))
        scores = []

        for label in self.labels:
            base_words = TOPIC_WORDS.get(label, {label})
            learned_words = TRAINING_TOPIC_WORDS.get(label, set())
            matches = tokens & (base_words | learned_words)
            score = sum(1 if word in WEAK_TOPIC_WORDS else 2 for word in matches)
            scores.append((label, score, sorted(matches)))

        scores.sort(key=lambda item: item[1], reverse=True)
        best_label, best_count, best_matches = scores[0] if scores else ("general", 0, [])

        if best_count == 0:
            return {
                "label": "general",
                "confidence": 25.0,
                "matches": [],
                "alternatives": [],
            }

        confidence = min(90.0, 55.0 + best_count * 10.0)
        alternatives = [
            {"label": label, "matches": matches}
            for label, count, matches in scores[1:3]
            if count > 0
        ]

        return {
            "label": best_label,
            "confidence": confidence,
            "matches": best_matches,
            "alternatives": alternatives,
        }
