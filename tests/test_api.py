import http.client
import json
import threading

from backend.api.app import create_server


def request(server, method, path, body=None):
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    headers = {}
    payload = None
    if body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"

    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    data = response.read()
    content_type = response.getheader("Content-Type", "")
    connection.close()

    if "application/json" in content_type:
        return response.status, json.loads(data.decode("utf-8"))
    return response.status, data.decode("utf-8")


def start_server():
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_health_endpoint():
    server, thread = start_server()
    try:
        status, data = request(server, "GET", "/health")
        assert status == 200
        assert data["status"] == "ok"
    finally:
        stop_server(server, thread)


def test_analyze_endpoint():
    server, thread = start_server()
    try:
        status, data = request(
            server,
            "POST",
            "/analyze",
            {"text": "The football match was amazing.", "use_context": False},
        )
        assert status == 200
        assert data["topic"]["label"] == "sports"
        assert "sentiment" in data
        assert "targets" in data
    finally:
        stop_server(server, thread)


def test_dashboard_and_report_endpoint():
    server, thread = start_server()
    try:
        page_status, page = request(server, "GET", "/dashboard")
        report_status, report = request(server, "GET", "/dashboard-data")
        assert page_status == 200
        assert "Evaluation Dashboard" in page
        assert report_status == 200
        assert "benchmark" in report
        assert "ambiguity" in report
    finally:
        stop_server(server, thread)


def test_invalid_analyze_request_returns_400():
    server, thread = start_server()
    try:
        status, data = request(server, "POST", "/analyze", {"text": ""})
        assert status == 400
        assert data["error"] == "invalid_request"
    finally:
        stop_server(server, thread)
