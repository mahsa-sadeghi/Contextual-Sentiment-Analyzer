# Contextual Sentiment Analyzer

A lightweight English NLP project that analyzes short text without using ready-made machine-learning libraries for the core models.

The project focuses on four tasks:

1. positive / negative sentiment classification
2. important keyword and opinion-word extraction
3. ambiguity detection with optional Wikipedia context retrieval
4. topic, target, and emotion labeling

The main sentiment classifier is a Naive Bayes model implemented directly in Python. The evaluation folder also contains small TF-IDF + Logistic Regression and TF-IDF + Linear SVM reference implementations written in Python for comparison.

No scikit-learn, TensorFlow, PyTorch, Transformers, or pretrained NLP model is used.

## Example

Input:

```text
I hate the World Cup coverage but I love club football.
```

Typical target-level result:

```text
Topic: sports

world cup coverage -> negative / disgust
club football      -> positive / joy
```

## Language support

The analyzer supports **English only**.

Non-English text is returned as unsupported instead of being passed through the English rules and model.

## Data

### Main dataset

`data/sentiment_dataset.csv` contains 180 hand-written examples across nine domains:

- sports
- politics
- technology
- business
- entertainment
- health
- science
- education
- travel

Each row contains:

```text
text, topic, target, sentiment, emotion, split
```

The split is fixed:

```text
90 training examples
36 calibration examples
54 local test examples
```

The training split is used for Naive Bayes counts and training-derived topic vocabulary.

The calibration split is used only for confidence calibration. It is not used for the word counts.

The local test split is not used for fitting either the classifier or confidence calibration.

### Benchmark

`data/benchmark.csv` contains 90 separately written English examples, ten per topic.

The benchmark includes wording that is intentionally different from the training data, including cases such as:

```text
without being difficult
stops being funny
no longer confusing
never answered
not reliable
```

The benchmark is not used for training, topic vocabulary, or confidence calibration.

It is still a project-maintained benchmark, not a public research benchmark.

### Ambiguity test

`data/ambiguity_test.csv` contains 30 labeled sentences for ambiguity evaluation.

## How the main components work

### Sentiment

The main classifier uses multinomial-style Naive Bayes written in this repository.

It implements:

- token counts
- class priors
- Laplace smoothing
- log probabilities
- positive / negative word scores
- prediction

A small explicit sentiment lexicon is used as a fallback when a common sentiment word is missing from the model vocabulary or when lexical evidence clearly conflicts with a weak model prediction.

### Confidence calibration

Raw Naive Bayes probability can be overconfident. The project therefore keeps a separate calibration split and fits one temperature value on that split.

Evaluation also reports:

- Expected Calibration Error (ECE)
- Brier score
- average confidence on correct answers
- average confidence on incorrect answers
- number of incorrect predictions above 80% confidence

Confidence should still be treated as an estimate, not a guarantee.

### Negation

Negation is handled in preprocessing rather than being duplicated across multiple analyzers.

The tokenizer supports common cases such as:

```text
not useful
never answered
no longer confusing
without being difficult
stops being funny
```

Negation scope is intentionally small and rule-based. This is more capable than the original single-word negation handling, but it is not a full syntactic parser.

### Topic labeling

Topic labeling uses an explicit domain vocabulary plus useful words from target labels in the **training split only**.

The benchmark and test sets never add vocabulary to the topic labeler.

### Target extraction

Target extraction uses three readable stages:

1. sentence/clause structure and likely subject phrases
2. opinion-verb and copula patterns
3. known training targets or extracted keyphrases as fallback

This improves exact target boundaries while keeping the code rule-based and inspectable.

### Emotion

Emotion detection uses a small explicit vocabulary for:

```text
joy
admiration
excitement
anger
disgust
sadness
fear
neutral
```

It also handles several negated emotion phrases. Emotion remains a simpler component than sentiment and is evaluated separately.

### Keywords

The keyword extractor builds short content phrases after removing common stopwords and opinion terms.

Opinion words are returned separately so concepts and opinions are not mixed together.

### Ambiguity

The ambiguity detector checks practical warning signs such as:

- unresolved pronouns
- underspecified references
- weak topic evidence
- missing targets
- very short unclear text

### Wikipedia context retrieval

If ambiguity is detected and context lookup is enabled, the project performs a small Wikipedia search and returns relevant snippets.

This is **context retrieval**, not a full RAG system. There is no LLM or generation step.

## Evaluation

Run:

```bash
python -m evaluation.evaluate
```

The latest report is written to:

```text
evaluation/reports/evaluation_report.json
```

Current measured results on the included datasets are approximately:

```text
Local sentiment accuracy:       100.00%
Benchmark sentiment accuracy:    80.00%
Benchmark topic accuracy:        74.44%
Benchmark emotion accuracy:      86.67%
Benchmark target exact match:    76.67%
Benchmark target partial match:  93.33%
Ambiguity F1:                   100.00%
Benchmark ECE:                    7.08%
Benchmark Brier score:            0.1418
```

The local test is intentionally easier than the benchmark and should mainly be treated as a regression test.

### Reference model comparison

The evaluation also trains two small reference classifiers from scratch on the same training split:

```text
TF-IDF + Logistic Regression
TF-IDF + Linear SVM
```

They are implemented in `evaluation/baselines.py` without scikit-learn.

On the included benchmark, the current report shows approximately:

```text
Project Naive Bayes + rules:     80.00%
TF-IDF + Logistic Regression:    60.00%
TF-IDF + Linear SVM:             58.89%
```

These results are specific to this small project dataset and should not be generalized to other datasets.

## Dashboard

Start the application:

```bash
python app.py
```

No `pip install` step is required for the web application. It uses Python's standard library for the HTTP server and Wikipedia requests.

Open:

```text
http://127.0.0.1:8000/dashboard
```

The dashboard shows:

- Accuracy / Precision / Recall / F1
- confusion matrix
- benchmark performance
- performance by topic
- emotion accuracy
- target exact / partial match
- confidence calibration (ECE and Brier score)
- high-confidence errors
- misclassified benchmark examples
- model comparison
- top positive and negative words

## Install

No package installation is required to run the application. Python 3.10 or newer is enough.

A virtual environment is optional. If you want one:

```bash
python -m venv .venv
```

The runtime `requirements.txt` intentionally contains no packages.

Only the development test suite needs an extra tool:

```bash
pip install -r requirements-dev.txt
```

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## API

Main endpoint:

```http
POST /analyze
```

Example:

```json
{
  "text": "I hate the World Cup coverage but I love club football.",
  "use_context": false
}
```

`use_context` only enables Wikipedia lookup when the ambiguity detector marks the input as ambiguous.

## CLI

```bash
python -m backend.cli.main
```

## MCP

MCP is optional:

```bash
pip install -r requirements-mcp.txt
python -m backend.mcp.server
```

## Configuration

Optional environment settings:

```env
CONTEXT_LOOKUP_ENABLED=false
CONTEXT_TOP_K=2
CONTEXT_TIMEOUT_SECONDS=2.5
```

## Model artifact

The trained Naive Bayes artifact is stored as:

```text
artifacts/model.json
```

JSON is used instead of pickle so loading the model does not execute arbitrary serialized Python objects.

If the dataset version changes, the loader retrains the model and writes a new JSON artifact.

## Continuous integration

A small GitHub Actions workflow is included in:

```text
.github/workflows/tests.yml
```

It runs the test suite and evaluation on each push and pull request.

## Model card

See:

```text
MODEL_CARD.md
```

for intended use, data boundaries, confidence notes, language support, and limitations.

## Project structure

```text
data/
  sentiment_dataset.csv
  benchmark.csv
  ambiguity_test.csv

backend/
  analysis/
  api/
  cli/
  core/
  mcp/
  sentiment/

evaluation/
  baselines.py
  evaluate.py
  metrics.py
  reports/

frontend/
  static/
  templates/

tests/
artifacts/
```

## Limitations

- English only.
- The datasets are still small and hand-written.
- The benchmark is separate from training but is maintained inside the project.
- Negation handling is rule-based, not syntactic parsing.
- Topic detection can miss unfamiliar vocabulary.
- Target extraction can still choose imperfect phrase boundaries.
- Emotion labels are coarse and depend on explicit vocabulary.
- Confidence calibration uses a small calibration set.
- Wikipedia context retrieval may return irrelevant context and requires internet access.

The project intentionally keeps these limitations visible rather than hiding them behind a more complex architecture.
