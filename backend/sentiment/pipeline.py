from backend.sentiment.model.loader import model


class SentimentPipeline:

    def __init__(self):
        self.model = model

    def predict(self, text):
        return self.model.predict(text)

    def explain(self, text):
        _, _, explanation = self.model.predict(text)
        return explanation

    def get_top_positive_words(self):
        return self.model.top_positive

    def get_top_negative_words(self):
        return self.model.top_negative

pipeline = SentimentPipeline()