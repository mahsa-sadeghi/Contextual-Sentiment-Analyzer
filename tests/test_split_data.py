from backend.sentiment.training.split_data import train_data, test_data

def test_no_train_test_sentence_overlap():
    train_sentences = { sentence for sentence, _ in train_data }
    test_sentences = { sentence for sentence, _ in test_data}

    assert train_sentences.isdisjoint(test_sentences)