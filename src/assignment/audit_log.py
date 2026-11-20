"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        key = request_id or user_id
        self._open[key] = time.time()
        self.logs.append({
            "request_id": key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": text,
        })

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision and latency on the matching input entry."""
        key = request_id or user_id
        start = self._open.pop(key, None)
        latency_ms = round((time.time() - start) * 1000, 1) if start else None
        fields = {
            "output": (text or "")[:500],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        }
        for entry in reversed(self.logs):
            if entry["request_id"] == key and "output" not in entry:
                entry.update(fields)
                return
        # output without a recorded input — still keep it
        self.logs.append({"request_id": key, "user_id": user_id,
                          "timestamp": utc_now_iso(), **fields})

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
