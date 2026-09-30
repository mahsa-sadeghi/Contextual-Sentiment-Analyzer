# Model Card

## Model

The project uses a small multinomial-style Naive Bayes sentiment classifier implemented directly in Python. It is combined with a small explicit sentiment lexicon for common out-of-vocabulary sentiment words.

No scikit-learn, TensorFlow, PyTorch, Transformers, or pretrained model is used for the core sentiment classifier.

## Intended use

The analyzer is designed for short **English** sentences and short English paragraphs in these domains:

- sports
- politics
- technology
- business
- entertainment
- health
- science
- education
- travel

It is intended for coursework, demonstrations, and lightweight experimentation. It is not intended for high-stakes decisions or production moderation.

## Training data

`data/sentiment_dataset.csv` contains 180 hand-written project examples:

- 90 training rows
- 36 calibration rows
- 54 local test rows

The training split is used for Naive Bayes counts and topic vocabulary. The calibration split is used only to fit the confidence temperature. The local test split is not used for fitting.

## Benchmark

`data/benchmark.csv` contains 90 separately written English examples, ten per topic. It includes harder negation and compositional wording and is not used for training or confidence calibration.

The benchmark is still a project-maintained dataset, not an independent public research benchmark.

## Confidence

Naive Bayes confidence is temperature-scaled using the calibration split. The final sentiment layer lowers confidence when lexical and model evidence disagree. Evaluation reports Expected Calibration Error (ECE), Brier score, confidence on incorrect answers, and high-confidence errors.

Confidence is an estimate, not a guarantee that a prediction is correct.

## Other components

Topic, target, emotion, ambiguity, keyword extraction, and Wikipedia context retrieval use readable project-specific rules and dictionaries. They are evaluated separately from sentiment.

Wikipedia context retrieval is retrieval only. It is not a full retrieval-augmented generation system because no generator/LLM is used.

## Language

Only English is supported. Non-English input is returned as unsupported rather than being analyzed as if it were English.

## Limitations

- The datasets are small and hand-written.
- Topic rules can miss unfamiliar vocabulary.
- Target extraction is rule-based and exact boundaries remain difficult.
- Emotion labels are coarse and dictionary-driven.
- Negation handling is improved but not a full syntactic parser.
- Confidence calibration is based on a small calibration split.
- Wikipedia retrieval requires network access and may retrieve the wrong context.

See `evaluation/reports/evaluation_report.json` for the latest measured results.
