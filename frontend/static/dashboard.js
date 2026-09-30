function percent(value) {
  return `${Number(value || 0).toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadDashboard() {
  const response = await fetch("/dashboard-data");
  const data = await response.json();

  if (data.error) {
    document.getElementById("metrics").innerHTML = `<article class="panel error">${escapeHtml(data.error)}</article>`;
    return;
  }

  document.getElementById("accuracy").textContent = percent(data.accuracy);
  document.getElementById("precision").textContent = percent(data.precision);
  document.getElementById("recall").textContent = percent(data.recall);
  document.getElementById("f1").textContent = percent(data.f1_score);
  document.getElementById("matrix-tp").textContent = data.tp;
  document.getElementById("matrix-tn").textContent = data.tn;
  document.getElementById("matrix-fp").textContent = data.fp;
  document.getElementById("matrix-fn").textContent = data.fn;

  const benchmark = data.benchmark || {};
  document.getElementById("benchmark-sentiment").textContent = percent(benchmark.sentiment?.accuracy);
  document.getElementById("benchmark-topic").textContent = percent(benchmark.topic?.accuracy);
  document.getElementById("ambiguity-f1").textContent = percent(data.ambiguity?.f1);
  document.getElementById("benchmark-target").textContent = percent(benchmark.targets?.exact_accuracy);
  document.getElementById("benchmark-note").textContent = data.benchmark_note || "";

  const confidence = benchmark.confidence || {};
  document.getElementById("confidence-details").innerHTML = `
    <div class="result-row"><strong>Local average</strong><span>${percent(data.average_confidence)}</span></div>
    <div class="result-row"><strong>Benchmark average</strong><span>${percent(confidence.average)}</span></div>
    <div class="result-row"><strong>Average on benchmark errors</strong><span>${percent(confidence.average_wrong)}</span></div>
    <div class="result-row"><strong>High-confidence errors (≥80%)</strong><span>${confidence.high_confidence_error_count ?? 0}</span></div>
    <div class="result-row"><strong>Calibration ECE</strong><span>${percent(confidence.calibration?.ece)}</span></div>
    <div class="result-row"><strong>Brier score</strong><span>${Number(confidence.calibration?.brier ?? 0).toFixed(4)}</span></div>`;


  const references = data.reference_models || {};
  const modelRows = [
    ["Naive Bayes + lightweight rules", benchmark.sentiment?.accuracy],
    ["TF-IDF + Logistic Regression", references.tfidf_logistic_regression?.accuracy],
    ["TF-IDF + Linear SVM", references.tfidf_linear_svm?.accuracy],
  ];
  document.getElementById("model-table").innerHTML = modelRows
    .map(([name, value]) => `<tr><td>${escapeHtml(name)}</td><td>${percent(value)}</td></tr>`)
    .join("");

  const topicRows = Object.entries(benchmark.per_topic || {}).map(([topic, values]) => `
    <tr>
      <td>${escapeHtml(topic)}</td>
      <td>${values.samples}</td>
      <td>${percent(values.sentiment_accuracy)}</td>
      <td>${percent(values.topic_accuracy)}</td>
    </tr>`);
  document.getElementById("topic-table").innerHTML = topicRows.join("");

  const localTargets = data.local?.targets || {};
  const benchmarkTargets = benchmark.targets || {};
  document.getElementById("target-table").innerHTML = `
    <tr><td>Local test</td><td>${localTargets.samples ?? 0}</td><td>${percent(localTargets.exact_accuracy)}</td><td>${percent(localTargets.partial_accuracy)}</td></tr>
    <tr><td>External challenge</td><td>${benchmarkTargets.samples ?? 0}</td><td>${percent(benchmarkTargets.exact_accuracy)}</td><td>${percent(benchmarkTargets.partial_accuracy)}</td></tr>`;

  const errors = benchmark.misclassified || [];
  const errorTable = document.getElementById("error-table");
  if (errors.length) {
    errorTable.innerHTML = errors.slice(0, 12).map(item => `
      <tr>
        <td class="sentence-cell">${escapeHtml(item.text)}</td>
        <td><span class="chip ${item.actual === "positive" ? "good" : "bad"}">${escapeHtml(item.actual)}</span></td>
        <td><span class="chip ${item.predicted === "positive" ? "good" : item.predicted === "negative" ? "bad" : "neutral"}">${escapeHtml(item.predicted)}</span></td>
        <td>${percent(item.confidence)}</td>
      </tr>`).join("");
  } else {
    document.getElementById("no-errors").classList.remove("hidden");
  }

  const details = [
    ["Local Macro F1", percent(data.macro_f1)],
    ["Calibration temperature", Number(data.calibration_temperature ?? 1).toFixed(2)],
    ["Local topic accuracy", percent(data.local?.topic?.accuracy)],
    ["Local emotion accuracy", percent(data.local?.emotion?.accuracy)],
    ["Benchmark emotion accuracy", percent(benchmark.emotion?.accuracy)],
    ["Ambiguity accuracy", percent(data.ambiguity?.accuracy)],
    ["Ambiguity precision", percent(data.ambiguity?.precision)],
    ["Ambiguity recall", percent(data.ambiguity?.recall)],
    ["Train samples", data.dataset?.train_samples ?? "--"],
    ["Local test samples", data.dataset?.test_samples ?? "--"],
    ["Benchmark samples", data.dataset?.benchmark_samples ?? "--"],
    ["Calibration samples", data.dataset?.calibration_samples ?? "--"],
    ["Ambiguity samples", data.dataset?.ambiguity_samples ?? "--"],
  ];
  document.getElementById("metric-table").innerHTML = details
    .map(row => `<tr><td>${row[0]}</td><td>${row[1]}</td></tr>`)
    .join("");

  const wordResponse = await fetch("/word-insights");
  const wordData = await wordResponse.json();

  function renderWords() {
    const type = document.getElementById("word-type").value;
    const count = Number(document.getElementById("word-count").value);
    const words = (wordData[type] || []).slice(0, count);
    document.getElementById("word-list").innerHTML = words
      .map(item => `<div class="result-row"><strong>${escapeHtml(item[0])}</strong><span>${Number(item[1]).toFixed(3)}</span></div>`)
      .join("");
  }

  document.getElementById("explore-btn").addEventListener("click", renderWords);
  renderWords();
}

loadDashboard();
