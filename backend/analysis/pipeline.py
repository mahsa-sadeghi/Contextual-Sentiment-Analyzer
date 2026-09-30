from backend.analysis.ambiguity import AmbiguityDetector
from backend.analysis.emotion import EmotionAnalyzer
from backend.analysis.keywords import KeywordExtractor
from backend.analysis.language import check_english_support
from backend.analysis.context_retrieval import WikipediaContextRetriever
from backend.analysis.sentiment import SentimentAnalyzer
from backend.analysis.targets import TargetAnalyzer
from backend.analysis.text import sentences
from backend.analysis.topic import TopicLabeler
from backend.core.config import CONTEXT_LOOKUP_ENABLED


class SemanticAnalysisPipeline:
    def __init__(self, context_retriever=None):
        self.sentiment = SentimentAnalyzer()
        self.emotion = EmotionAnalyzer()
        self.keywords = KeywordExtractor()
        self.topic = TopicLabeler()
        self.targets = TargetAnalyzer(self.sentiment, self.emotion)
        self.ambiguity = AmbiguityDetector()
        self.context_retriever = context_retriever or WikipediaContextRetriever()

    def analyze(self, text: str, use_context: bool | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            return self._empty_result()

        language = check_english_support(text)
        if not language["supported"]:
            result = self._empty_result()
            result["text"] = text
            result["language"] = language
            result["error"] = "unsupported_language"
            return result

        sentiment = self.sentiment.analyze(text)
        emotion = self.emotion.analyze(text)
        keyword_result = self.keywords.extract(text)
        opinion_terms = self.keywords.opinion_terms(text)
        topic = self.topic.classify(text)
        targets = self.targets.analyze(text, keyword_result["items"], opinion_terms)
        ambiguity = self.ambiguity.detect(text, sentiment, topic, targets)

        context_enabled = CONTEXT_LOOKUP_ENABLED if use_context is None else use_context
        context_result = self._run_context_retrieval(text, targets, keyword_result["items"], ambiguity, context_enabled)

        return {
            "text": text,
            "language": language,
            "sentiment": sentiment,
            "emotion": emotion,
            "topic": topic,
            "targets": targets,
            "keywords": keyword_result["items"],
            "opinion_terms": opinion_terms,
            "important_sentences": self._important_sentences(text, keyword_result["items"]),
            "ambiguity": ambiguity,
            "context_retrieval": context_result,
            "summary": self._summary(topic, targets, sentiment, emotion),
        }

    def _run_context_retrieval(self, text, targets, keywords, ambiguity, enabled):
        if not ambiguity["is_ambiguous"]:
            return {"status": "not_needed", "sources": []}

        if not enabled:
            return {
                "status": "disabled",
                "resolution": "Ambiguity detected. Enable Wikipedia lookup to search for context.",
                "sources": [],
            }

        query_parts = [item["target"] for item in targets[:2]]
        if not query_parts:
            query_parts = [item["text"] for item in keywords[:3]]
        query = " ".join(query_parts).strip() or text
        return self.context_retriever.resolve(text, query, ambiguity["reasons"])

    def _important_sentences(self, text: str, keywords: list[dict], limit: int = 3) -> list[dict]:
        ranked = []
        for sentence in sentences(text) or [text]:
            lower = sentence.lower()
            relevance = sum(item["relevance"] for item in keywords if item["text"].lower() in lower)
            topic = self.topic.classify(sentence)
            ranked.append({
                "text": sentence,
                "relevance": round(relevance, 2),
                "topic": topic["label"],
                "topic_confidence": topic["confidence"],
            })

        ranked.sort(key=lambda item: item["relevance"], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _summary(topic: dict, targets: list[dict], sentiment: dict, emotion: dict) -> dict:
        if targets:
            primary = targets[0]
            return {
                "topic": topic["label"],
                "target": primary["target"],
                "sentiment": primary["sentiment"],
                "emotion": primary["emotion"],
                "confidence": primary["sentiment_confidence"],
            }

        return {
            "topic": topic["label"],
            "target": None,
            "sentiment": sentiment["label"],
            "emotion": emotion["label"],
            "confidence": sentiment["confidence"],
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "text": "",
            "language": {"code": "unsupported", "supported": False, "message": "Only English text is supported by this analyzer."},
            "sentiment": {"label": "unknown", "confidence": 0.0},
            "emotion": {"label": "neutral", "confidence": 0.0, "evidence": []},
            "topic": {"label": "general", "confidence": 0.0, "matches": [], "alternatives": []},
            "targets": [],
            "keywords": [],
            "opinion_terms": [],
            "important_sentences": [],
            "ambiguity": {"is_ambiguous": True, "score": 1.0, "confidence": 100.0, "reasons": ["empty input"]},
            "context_retrieval": {"status": "not_run", "sources": []},
            "summary": {"topic": "general", "target": None, "sentiment": "unknown", "emotion": "neutral", "confidence": 0.0},
        }


semantic_pipeline = SemanticAnalysisPipeline()
