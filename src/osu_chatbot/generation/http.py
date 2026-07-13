from __future__ import annotations

from urllib import error, request
from typing import Any
import json

from .base import GenerationError


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float,
    service_name: str,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise GenerationError(f"{service_name} returned HTTP {exc.code}.") from exc
    except (OSError, error.URLError) as exc:
        raise GenerationError(f"Could not reach {service_name}.") from exc

    try:
        data = json.loads(raw_body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GenerationError(f"{service_name} returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise GenerationError(f"{service_name} returned an unexpected response.")
    return data
