# The project-wide logger lives here, so any module can do:
#     from cnnClassifier import logger
#
# Every log line is one JSON object, so tools (CloudWatch, Loki, jq, grep) can
# filter on fields instead of parsing sentences. Extra fields passed like
#     logger.info("request", extra={"status": 200, "latency_ms": 41})
# become top-level keys, and `request_id` is added automatically while a web
# request is being handled.

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# set by the API's request middleware; every log line written during that request carries it
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# attributes every LogRecord has. Anything else came from `extra=` and is kept.
# ("color_message" is uvicorn's copy of the message with terminal color codes.)
_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

_handlers = [
    logging.FileHandler(log_filepath),  # saved in the logs folder
    logging.StreamHandler(sys.stdout),  # printed to the console (docker logs)
]
for _handler in _handlers:
    _handler.setFormatter(JsonFormatter())

logging.basicConfig(level=logging.INFO, handlers=_handlers)

logger = logging.getLogger("cnnClassifierLogger")
