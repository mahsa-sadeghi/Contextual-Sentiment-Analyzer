"""Small reference classifiers used only for evaluation.

They are implemented here instead of using scikit-learn so the comparison stays
readable and reproducible on the same lightweight project.
"""
from __future__ import annotations

import math
from collections import Counter

from backend.sentiment.preprocessing.tokenizer import tokenize


def _features(text: str) -> list[str]:
    tokens = tokenize(text)
    bigrams = [f"{tokens[index]}::{tokens[index + 1]}" for index in range(len(tokens) - 1)]
    return tokens + bigrams


class TfidfVectorizer:
    def fit(self, texts: list[str]):
        document_frequency = Counter()
        for text in texts:
            document_frequency.update(set(_features(text)))

        total_documents = max(len(texts), 1)
        self.idf = {
            term: math.log((1 + total_documents) / (1 + count)) + 1.0
            for term, count in document_frequency.items()
            if count >= 2
        }
        return self

    def transform_one(self, text: str) -> dict[str, float]:
        terms = [term for term in _features(text) if term in self.idf]
        if not terms:
            return {}

        counts = Counter(terms)
        total = sum(counts.values())
        vector = {
            term: (count / total) * self.idf[term]
            for term, count in counts.items()
        }

        length = math.sqrt(sum(value * value for value in vector.values()))
        if length:
            vector = {term: value / length for term, value in vector.items()}
        return vector

    def transform(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.transform_one(text) for text in texts]


class LogisticRegressionClassifier:
    def __init__(self, epochs: int = 90, learning_rate: float = 0.25, l2: float = 0.002):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights: dict[str, float] = {}
        self.bias = 0.0

    def fit(self, vectors: list[dict[str, float]], labels: list[str]):
        for epoch in range(self.epochs):
            rate = self.learning_rate / (1 + epoch * 0.03)
            for vector, label in zip(vectors, labels):
                actual = 1.0 if label == "positive" else 0.0
                probability = self._sigmoid(self._score(vector))
                error = actual - probability

                for term, value in vector.items():
                    current = self.weights.get(term, 0.0)
                    self.weights[term] = current + rate * (error * value - self.l2 * current)
                self.bias += rate * error
        return self

    def predict(self, vector: dict[str, float]) -> str:
        return "positive" if self._score(vector) >= 0 else "negative"

    def _score(self, vector: dict[str, float]) -> float:
        return self.bias + sum(self.weights.get(term, 0.0) * value for term, value in vector.items())

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(min(value, 40), -40)
        return 1 / (1 + math.exp(-value))


class LinearSVMClassifier:
    def __init__(self, epochs: int = 80, learning_rate: float = 0.12, l2: float = 0.002):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights: dict[str, float] = {}
        self.bias = 0.0

    def fit(self, vectors: list[dict[str, float]], labels: list[str]):
        for epoch in range(self.epochs):
            rate = self.learning_rate / (1 + epoch * 0.03)
            shrink = max(0.0, 1 - rate * self.l2)

            for vector, label in zip(vectors, labels):
                actual = 1.0 if label == "positive" else -1.0
                score = self._score(vector)

                for term in list(self.weights):
                    self.weights[term] *= shrink

                if actual * score < 1:
                    for term, value in vector.items():
                        self.weights[term] = self.weights.get(term, 0.0) + rate * actual * value
                    self.bias += rate * actual
        return self

    def predict(self, vector: dict[str, float]) -> str:
        return "positive" if self._score(vector) >= 0 else "negative"

    def _score(self, vector: dict[str, float]) -> float:
        return self.bias + sum(self.weights.get(term, 0.0) * value for term, value in vector.items())


def compare_reference_models(train_rows: list[dict], benchmark_rows: list[dict]) -> dict:
    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["sentiment"] for row in train_rows]
    benchmark_texts = [row["text"] for row in benchmark_rows]
    actual = [row["sentiment"] for row in benchmark_rows]

    vectorizer = TfidfVectorizer().fit(train_texts)
    train_vectors = vectorizer.transform(train_texts)
    benchmark_vectors = vectorizer.transform(benchmark_texts)

    result = {}
    for name, classifier in (
        ("tfidf_logistic_regression", LogisticRegressionClassifier()),
        ("tfidf_linear_svm", LinearSVMClassifier()),
    ):
        classifier.fit(train_vectors, train_labels)
        predicted = [classifier.predict(vector) for vector in benchmark_vectors]
        correct = sum(expected == prediction for expected, prediction in zip(actual, predicted))
        result[name] = {
            "accuracy": round(correct / len(actual) * 100, 2) if actual else 0.0,
            "correct": correct,
            "samples": len(actual),
        }

    result["vocabulary_size"] = len(vectorizer.idf)
    return result
