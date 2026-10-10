"""Tests for FR3b (AR-v0.5.0-hpc-readiness / T004): run management.

Contract under test, spanning three modules:

``utils/notify.py`` (new module):
  * ``WEBHOOK_ENV_VAR == "NANOGPT_WEBHOOK_URL"``.
  * ``notify_webhook(message, env=None, timeout=10.0)``: env defaults to
    ``os.environ``; a missing / empty URL returns False silently (no
    exception, no network).  With a URL it POSTs JSON ``{"text": message}``
    (Content-Type: application/json) via ``urllib.request.urlopen``; a 2xx
    status returns True and any exception returns False without raising.

``utils/monitoring.py`` — new ``MetricsHeartbeatWatchdog(threading.Thread)``:
  * ``check(now)`` parses ``type == "train"`` events from
    ``utils.experiment.load_metrics(metrics_path)`` (ISO-8601 ``ts`` ->
    epoch seconds, invalid lines skipped).  Fewer than 2 events -> None
    (baseline not established).  baseline = mean adjacent interval of the
    last ``baseline_window`` timestamps (intervals <= 0 dropped); alert when
    ``age = now - last_ts`` exceeds ``timeout_factor * baseline``.  On the
    first alert (debounced via ``self.alerted``) it prints to sys.stderr,
    invokes ``on_alert(msg)`` (None or raising callbacks swallowed) and
    calls ``notify_webhook(msg)``, returning the message:
    ``"[watchdog] {group}: training stalled (no metrics for {age:.0f}s, "
    "baseline {baseline:.1f}s)"``.  It never kills the training process.
  * ``run()`` polls ``check(time.time())`` every ``poll_interval`` until the
    stop event is set; ``stop()`` is idempotent; the thread is a daemon and
    exits after start()+stop()+join().

``utils/logging.py`` — ``Logger``:
  * ``__init__`` attaches a ``logging.handlers.RotatingFileHandler`` writing
    to ``{log_dir}/train.log`` (log_dir root, maxBytes=10 MiB,
    backupCount=3, utf-8, asctime in the format).  OSError degrades to a
    console warning.  The handler is exposed as ``self.file_handler``.

``run_component_ablations.py``:
  * ``start_group_watchdog(out_root, name, poll_interval=15.0)`` builds and
    starts a watchdog on ``group_exp_dir(out_root, name)/metrics.jsonl``;
    any failure returns None (watchdog never blocks training).
  * ``run_train_stage`` starts/stops a watchdog around every
    ``subprocess.run`` (resume attempt and degraded rerun each get one
    cycle; a None watchdog skips stop).
  * ``notify_failures(errors, env=None)``: empty errors -> False (nothing
    sent); otherwise a multi-line message (first line carries the group
    count, one ``"- {group} (exit {returncode})"`` line per failure) goes
    through ``notify_webhook`` and its return value is propagated.
  * ``main()`` calls ``notify_failures`` when the train stage recorded
    errors.

The watchdog/alert tests use a fixed ISO timestamp base for determinism and
monkeypatch away the webhook env var so no real network call can happen.
"""

import json
import logging
import logging.handlers
import re
import time
import urllib.error
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import run_component_ablations as rca
import utils.notify as notify_mod
from utils.logging import Logger
from utils.monitoring import MetricsHeartbeatWatchdog
from utils.notify import WEBHOOK_ENV_VAR, notify_webhook

# Fixed deterministic base for synthesized ISO-8601 UTC timestamps.
_TS_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_train_metrics(path, n_events=5, interval_s=10, base=_TS_BASE):
    """Write ``n_events`` train events at ``interval_s`` apart.

    Returns the list of epoch seconds corresponding to the written ``ts``
    values (parsed back with ``datetime.fromisoformat(...).timestamp()``
    exactly like the watchdog must).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts_epoch = []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_events):
            ts = (base + timedelta(seconds=i * interval_s)).isoformat()
            ts_epoch.append(datetime.fromisoformat(ts).timestamp())
            f.write(json.dumps({"step": i, "type": "train", "loss": 1.0, "ts": ts}) + "\n")
    return ts_epoch


class _FakeResponse:
    """Minimal context-manager stand-in for urlopen's return value."""

    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# 1-2. MetricsHeartbeatWatchdog.check: stall alert + debounce / healthy paths
# ---------------------------------------------------------------------------


def test_watchdog_alerts_on_stall_and_debounces(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(WEBHOOK_ENV_VAR, raising=False)
    metrics_path = tmp_path / "metrics.jsonl"
    ts_epoch = _write_train_metrics(metrics_path, n_events=5, interval_s=10)
    last = ts_epoch[-1]

    collected = []
    wd = MetricsHeartbeatWatchdog(str(metrics_path), "g1", on_alert=collected.append)
    assert wd.group_name == "g1"
    assert wd.alerted is False

    # 31s stale vs a 10s baseline: 31 > timeout_factor(3) * 10 -> alert.
    msg = wd.check(now=last + 31)
    assert isinstance(msg, str)
    assert msg == ("[watchdog] g1: training stalled (no metrics for 31s, baseline 10.0s)")
    # on_alert received exactly the same message, exactly once.
    assert collected == [msg]
    assert wd.alerted is True
    # The alert also went to stderr.
    assert "[watchdog] g1" in capsys.readouterr().err

    # Debounce: an even staler check must not re-alert.
    assert wd.check(now=last + 100) is None
    assert collected == [msg]


def test_watchdog_no_alert_when_healthy_or_insufficient(tmp_path, monkeypatch):
    monkeypatch.delenv(WEBHOOK_ENV_VAR, raising=False)

    # (a) Healthy stream: last event 5s ago, well under 3x10s.
    p5 = tmp_path / "metrics.jsonl"
    ts5 = _write_train_metrics(p5, n_events=5, interval_s=10)
    wd = MetricsHeartbeatWatchdog(str(p5), "healthy")
    assert wd.check(now=ts5[-1] + 5) is None
    assert wd.alerted is False

    # (b) Single event: fewer than 2 events -> baseline not established.
    p1 = tmp_path / "single.jsonl"
    ts1 = _write_train_metrics(p1, n_events=1)
    wd1 = MetricsHeartbeatWatchdog(str(p1), "single")
    assert wd1.check(now=ts1[0] + 10_000) is None
    assert wd1.alerted is False

    # (c) Empty file: no events at all -> None.
    pe = tmp_path / "empty.jsonl"
    pe.write_text("", encoding="utf-8")
    wde = MetricsHeartbeatWatchdog(str(pe), "empty")
    assert wde.check(now=time.time()) is None

    # (d) Insufficient data never arms the debounce flag.
    assert wde.alerted is False


# ---------------------------------------------------------------------------
# 3-4. notify_webhook: silent missing-URL + success/failure paths
# ---------------------------------------------------------------------------


def test_webhook_missing_url_silent(monkeypatch):
    assert WEBHOOK_ENV_VAR == "NANOGPT_WEBHOOK_URL"

    # URL absent from the environment -> False, no exception.
    monkeypatch.delenv(WEBHOOK_ENV_VAR, raising=False)
    assert notify_webhook("x") is False

    # URL explicitly empty -> False, no exception.
    assert notify_webhook("x", env={WEBHOOK_ENV_VAR: ""}) is False


def test_webhook_success_and_failure_paths(monkeypatch):
    requests = []

    def fake_urlopen(req, *args, **kwargs):
        requests.append(req)
        return _FakeResponse(200)

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", fake_urlopen)

    ok = notify_webhook("m", env={WEBHOOK_ENV_VAR: "http://x/hook"})
    assert ok is True
    assert len(requests) == 1
    req = requests[0]
    assert req.full_url == "http://x/hook"
    # JSON body {"text": message} with an application/json content type.
    assert json.loads(req.data) == {"text": "m"}
    headers = {k.lower(): v.lower() for k, v in req.headers.items()}
    assert headers.get("content-type") == "application/json"

    # urlopen raising -> False, no exception propagates.
    def failing_urlopen(req, *args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", failing_urlopen)
    assert notify_webhook("m", env={WEBHOOK_ENV_VAR: "http://x/hook"}) is False
    assert len(requests) == 1  # no second request was made


# ---------------------------------------------------------------------------
# 5. Logger: RotatingFileHandler on {log_dir}/train.log
# ---------------------------------------------------------------------------


def test_logger_rotating_file_handler_writes_log(tmp_path):
    # Unique run_name -> fresh underlying logging.Logger per test process.
    run_name = f"rot-{uuid.uuid4().hex[:8]}"
    logger = Logger(
        project_name="t",
        run_name=run_name,
        log_dir=str(tmp_path),
        use_tensorboard=False,
    )

    log_file = tmp_path / "train.log"
    assert log_file.exists()
    assert isinstance(logger.file_handler, logging.handlers.RotatingFileHandler)

    logger._console.warning("hello-rotation")
    logger.file_handler.flush()
    content = log_file.read_text(encoding="utf-8")
    assert "hello-rotation" in content
    # Formatter carries %(asctime)s (default "YYYY-MM-DD HH:MM:SS,mmm").
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)

    logger.close()  # must not raise


# ---------------------------------------------------------------------------
# 6. run_component_ablations.notify_failures
# ---------------------------------------------------------------------------


def test_notify_failures_message_and_skip(monkeypatch):
    calls = []

    def fake_notify(message, *args, **kwargs):
        calls.append(message)
        return True

    monkeypatch.setattr(rca, "notify_webhook", fake_notify, raising=False)

    # Empty errors -> False and nothing sent.
    assert rca.notify_failures([]) is False
    assert calls == []

    errors = [
        {"group": "moe", "returncode": 1, "command": ["x"]},
        {"group": "mha", "returncode": 2, "command": ["y"]},
    ]
    assert rca.notify_failures(errors) is True
    assert len(calls) == 1
    msg = calls[0]
    # First line carries the group count; every group gets its own line.
    assert "2" in msg.splitlines()[0]
    assert "- moe (exit 1)" in msg
    assert "- mha (exit 2)" in msg


# ---------------------------------------------------------------------------
# 7. run_train_stage: watchdog start/stop around subprocess.run
# ---------------------------------------------------------------------------


def test_run_train_stage_starts_and_stops_watchdog(tmp_path, monkeypatch):
    out_root = str(tmp_path)
    name = "baseline"
    # No manifest -> the group is pending; no checkpoint -> single launch.

    class FakeWatchdog:
        def __init__(self):
            self.start_calls = 0
            self.stop_calls = 0

        def start(self):
            self.start_calls += 1

        def stop(self):
            self.stop_calls += 1

    fake = FakeWatchdog()
    monkeypatch.setattr(rca, "start_group_watchdog", lambda *a, **k: fake, raising=False)
    monkeypatch.setattr(
        rca.subprocess,
        "run",
        lambda cmd, *a, **k: SimpleNamespace(returncode=0),
    )

    args = rca.get_args(
        [
            "--stage",
            "train",
            "--groups",
            name,
            "--out_root",
            out_root,
            "--config",
            str(tmp_path / "cfg.yaml"),
        ]
    )
    result = rca.run_train_stage(args)

    assert fake.start_calls >= 1
    assert fake.stop_calls == fake.start_calls  # every start got a stop
    assert name in result["completed"]
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# 8. MetricsHeartbeatWatchdog thread lifecycle
# ---------------------------------------------------------------------------


def test_watchdog_thread_lifecycle(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    wd = MetricsHeartbeatWatchdog(str(metrics_path), "lc", poll_interval=0.05)
    assert wd.daemon is True
    assert wd.alerted is False

    wd.start()
    wd.stop()
    wd.join(timeout=5)
    assert not wd.is_alive()

    wd.stop()  # idempotent: a second stop must not raise
    assert wd.alerted is False  # empty metrics never triggered an alert
