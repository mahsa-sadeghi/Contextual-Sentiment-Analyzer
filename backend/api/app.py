import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from backend.analysis.pipeline import semantic_pipeline
from backend.core.config import EVALUATION_REPORT_PATH, STATIC_DIR, TEMPLATES_DIR
from backend.sentiment.pipeline import pipeline as local_sentiment_pipeline

MAX_REQUEST_BYTES = 100_000


def health_data():
    return {"status": "ok", "application": "contextual-sentiment-analyzer"}


def capabilities_data():
    return {
        "features": [
            "sentiment",
            "emotion",
            "keywords",
            "opinion_terms",
            "topic",
            "targets",
            "important_sentences",
            "ambiguity_detection",
            "wikipedia_context_lookup",
        ]
    }


def dashboard_data():
    if not EVALUATION_REPORT_PATH.exists():
        return {"error": "Evaluation report not found. Run: python -m evaluation.evaluate"}

    with EVALUATION_REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def word_insights_data():
    return {
        "positive": local_sentiment_pipeline.get_top_positive_words()[:50],
        "negative": local_sentiment_pipeline.get_top_negative_words()[:50],
    }


def analyze_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")

    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("'text' must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("'text' cannot be empty.")
    if len(text) > 10_000:
        raise ValueError("'text' is too long.")

    use_context = payload.get("use_context")
    if use_context is not None and not isinstance(use_context, bool):
        raise ValueError("'use_context' must be true, false, or null.")

    return semantic_pipeline.analyze(text, use_context=use_context)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ContextualSentimentAnalyzer/1.0"

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._send_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/dashboard":
            self._send_file(TEMPLATES_DIR / "dashboard.html", "text/html; charset=utf-8")
            return
        if path == "/health":
            self._send_json(health_data())
            return
        if path == "/capabilities":
            self._send_json(capabilities_data())
            return
        if path == "/dashboard-data":
            self._send_json(dashboard_data())
            return
        if path == "/word-insights":
            self._send_json(word_insights_data())
            return
        if path.startswith("/static/"):
            self._serve_static(path)
            return

        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/analyze":
            self._send_json({"error": "not_found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "invalid_content_length"}, status=400)
            return

        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json({"error": "invalid_request_size"}, status=400)
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            result = analyze_payload(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "invalid_json"}, status=400)
            return
        except ValueError as exc:
            self._send_json({"error": "invalid_request", "message": str(exc)}, status=400)
            return

        self._send_json(result)

    def _serve_static(self, request_path):
        relative = unquote(request_path[len("/static/"):]).lstrip("/")
        target = (STATIC_DIR / relative).resolve()

        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "not_found"}, status=404)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._send_file(target, content_type)

    def _send_file(self, path: Path, content_type: str):
        if not path.is_file():
            self._send_json({"error": "not_found"}, status=404)
            return

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def create_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), RequestHandler)


def run(host="127.0.0.1", port=8000):
    server = create_server(host, port)
    print(f"Contextual Sentiment Analyzer running at http://{host}:{server.server_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    run()
