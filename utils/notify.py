"""Webhook notification for run management (AR-v0.5.0 / FR3b).

Sends short text notifications (failure digests, watchdog stall alerts) to
a webhook URL injected via the ``NANOGPT_WEBHOOK_URL`` environment variable.
Following the degradation philosophy, a missing URL or any transport
failure is silent (returns False) and never blocks the caller.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Mapping, Optional

WEBHOOK_ENV_VAR = "NANOGPT_WEBHOOK_URL"


def notify_webhook(
    message: str,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
) -> bool:
    """POST ``{"text": message}`` to the configured webhook URL.

    Parameters
    ----------
    message:
        Notification text (may be multi-line).
    env:
        Environment mapping to read ``NANOGPT_WEBHOOK_URL`` from;
        defaults to ``os.environ``.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    bool
        True on a 2xx response; False when the URL is missing/empty or the
        request fails for any reason (never raises).
    """
    env = os.environ if env is None else env
    url = env.get(WEBHOOK_ENV_VAR)
    if not url:
        return False
    try:
        data = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status: Any = getattr(resp, "status", None)
            return 200 <= int(status) < 300
    except Exception:
        return False
