import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from backend.analysis.text import lexical_overlap
from backend.core.config import CONTEXT_TIMEOUT_SECONDS, CONTEXT_TOP_K


class WikipediaContextRetriever:
    def search(self, query: str) -> list[dict]:
        params = urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": CONTEXT_TOP_K,
        })
        request = Request(
            f"https://en.wikipedia.org/w/api.php?{params}",
            headers={"User-Agent": "contextual-sentiment-analyzer/1.0"},
        )

        with urlopen(request, timeout=CONTEXT_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = []
        for row in payload.get("query", {}).get("search", []):
            title = row.get("title", "").strip()
            snippet = re.sub(r"<[^>]+>", " ", row.get("snippet", ""))
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if not title:
                continue
            results.append({
                "title": title,
                "snippet": snippet,
                "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            })
        return results

    def resolve(self, text: str, query: str, reasons: list[str]) -> dict:
        try:
            results = self.search(query or text)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError):
            return {
                "status": "search_failed",
                "resolution": "Ambiguity was detected, but Wikipedia could not be reached.",
                "sources": [],
            }

        if not results:
            return {
                "status": "no_context",
                "resolution": "Ambiguity was detected, but Wikipedia returned no useful context.",
                "sources": [],
            }

        for item in results:
            item["score"] = lexical_overlap(text, f"{item['title']} {item['snippet']}")

        results.sort(key=lambda item: item["score"], reverse=True)
        sources = [
            {
                "title": item["title"],
                "url": item["url"],
                "snippet": item["snippet"],
                "score": round(item["score"], 3),
            }
            for item in results[:CONTEXT_TOP_K]
        ]

        return {
            "status": "context_found",
            "resolution": f"Possible context: {sources[0]['title']}",
            "sources": sources,
        }
