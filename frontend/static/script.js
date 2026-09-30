function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pill(text, cls = "") {
  return `<span class="chip ${cls}">${esc(text)}</span>`;
}

async function analyze() {
  const text = document.getElementById("sentence").value.trim();
  const useContext = document.getElementById("use-context").checked;
  const errorBox = document.getElementById("error");
  errorBox.classList.add("hidden");

  if (!text) {
    errorBox.textContent = "Please enter text to analyze.";
    errorBox.classList.remove("hidden");
    return;
  }

  const button = document.getElementById("analyze-btn");
  button.disabled = true;
  button.textContent = "Analyzing…";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, use_context: useContext})
    });
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    const data = await response.json();
    if (data.error === "unsupported_language") {
      document.getElementById("result-section").classList.add("hidden");
      errorBox.textContent = data.language?.message || "Only English text is supported.";
      errorBox.classList.remove("hidden");
      return;
    }
    render(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = "Analyze text";
  }
}

function render(data) {
  document.getElementById("result-section").classList.remove("hidden");

  document.getElementById("topic").textContent = (data.topic?.label || "general").toUpperCase();
  document.getElementById("topic-confidence").textContent = `${Number(data.topic?.confidence || 0).toFixed(1)}% confidence`;

  const sentiment = data.sentiment?.label || "unknown";
  const sentimentEl = document.getElementById("sentiment");
  sentimentEl.textContent = sentiment.toUpperCase();
  sentimentEl.className = sentiment;
  document.getElementById("sentiment-confidence").textContent = `${Number(data.sentiment?.confidence || 0).toFixed(1)}% confidence`;

  document.getElementById("emotion").textContent = (data.emotion?.label || "neutral").toUpperCase();
  document.getElementById("emotion-confidence").textContent = `${Number(data.emotion?.confidence || 0).toFixed(1)}% confidence`;
  document.getElementById("primary-target").textContent = data.summary?.target || "Not resolved";

  const targets = document.getElementById("targets");
  targets.innerHTML = "";
  if (!data.targets?.length) {
    targets.innerHTML = `<p class="muted">No stable target was extracted.</p>`;
  } else {
    data.targets.forEach(t => {
      const cls = t.sentiment === "positive" ? "good" : t.sentiment === "negative" ? "bad" : "neutral";
      targets.innerHTML += `<div class="result-row">
        <div><strong>${esc(t.target)}</strong><small>${esc(t.context)}</small></div>
        <div class="right">${pill(t.sentiment, cls)} ${pill(t.emotion)}<small>${Number(t.sentiment_confidence).toFixed(1)}%</small></div>
      </div>`;
    });
  }

  const keywords = document.getElementById("keywords");
  keywords.innerHTML = (data.keywords || []).map(k => {
    return pill(`${k.text} · ${Number(k.relevance).toFixed(2)}`);
  }).join("") || `<span class="muted">None</span>`;

  const opinions = document.getElementById("opinions");
  opinions.innerHTML = (data.opinion_terms || []).map(o => pill(o.text, o.polarity === "positive" ? "good" : "bad")).join("") || `<span class="muted">None</span>`;

  const important = document.getElementById("important-sentences");
  important.innerHTML = (data.important_sentences || []).map(s => `<div class="result-row"><div><strong>${esc(s.topic)}</strong><small>${esc(s.text)}</small></div><span>${Number(s.topic_confidence).toFixed(1)}%</span></div>`).join("");

  const ambiguity = document.getElementById("ambiguity");
  const a = data.ambiguity || {};
  ambiguity.innerHTML = `<div class="result-row"><strong>${a.is_ambiguous ? "Ambiguous" : "Clear enough"}</strong><span>score ${Number(a.score || 0).toFixed(2)}</span></div>`;
  if (a.reasons?.length) ambiguity.innerHTML += `<ul>${a.reasons.map(r => `<li>${esc(r)}</li>`).join("")}</ul>`;

  const contextBox = document.getElementById("context-retrieval");
  const r = data.context_retrieval || {};
  contextBox.innerHTML = `<p><strong>Web context:</strong> ${esc(r.status || "not_run")}</p>`;
  if (r.resolution) contextBox.innerHTML += `<p>${esc(r.resolution)}</p>`;
  if (r.sources?.length) {
    contextBox.innerHTML += `<div class="sources">${r.sources.map(s => `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>`).join("")}</div>`;
  }
}
