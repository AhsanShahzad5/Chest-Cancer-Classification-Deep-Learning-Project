import json
import logging

from cnnClassifier import JsonFormatter, request_id_var
from tests.conftest import image_bytes


def make_record(message="hello", **extra):
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_log_line_is_valid_json_with_core_fields():
    line = JsonFormatter().format(make_record("hello"))
    entry = json.loads(line)
    assert entry["message"] == "hello"
    assert entry["level"] == "INFO"
    assert {"time", "module"} <= set(entry)


def test_extra_fields_become_top_level_keys():
    entry = json.loads(JsonFormatter().format(make_record("request", status=200, latency_ms=12.5)))
    assert entry["status"] == 200
    assert entry["latency_ms"] == 12.5


def test_request_id_is_added_while_a_request_is_active():
    token = request_id_var.set("abc123")
    try:
        entry = json.loads(JsonFormatter().format(make_record()))
    finally:
        request_id_var.reset(token)
    assert entry["request_id"] == "abc123"


def test_no_request_id_outside_a_request():
    assert "request_id" not in json.loads(JsonFormatter().format(make_record()))


def test_response_carries_a_request_id(make_client):
    response = make_client().get("/health")
    assert len(response.headers["X-Request-ID"]) > 0


def test_safe_client_request_id_is_kept(make_client):
    response = make_client().get("/health", headers={"X-Request-ID": "trace-42"})
    assert response.headers["X-Request-ID"] == "trace-42"


def test_unsafe_client_request_id_is_replaced(make_client):
    response = make_client().get("/health", headers={"X-Request-ID": "bad id!\n"})
    assert response.headers["X-Request-ID"] != "bad id!\n"


def test_each_request_is_logged_once_with_status_and_latency(make_client, caplog):
    client = make_client()
    with caplog.at_level(logging.INFO, logger="cnnClassifierLogger"):
        client.post("/predict", files={"file": ("a.png", image_bytes(), "image/png")})
    lines = [r for r in caplog.records if r.getMessage() == "request"]
    assert len(lines) == 1
    assert lines[0].path == "/predict"
    assert lines[0].method == "POST"
    assert lines[0].status == 200
    assert lines[0].latency_ms >= 0


def test_health_checks_are_not_logged_at_info(make_client, caplog):
    client = make_client()
    with caplog.at_level(logging.INFO, logger="cnnClassifierLogger"):
        client.get("/health")
        client.get("/ready")
    assert not [r for r in caplog.records if r.getMessage() == "request"]
