"""Abuse-prevention and cost guards for the platform interface."""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import defaultdict, deque
from pathlib import Path


class SlidingWindowLimiter:
    """Per-key sliding-window event limiter, safe for concurrent use."""

    def __init__(self, *, max_events: int, window_s: float) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.max_events = max_events
        self.window_s = window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            cutoff = now - self.window_s
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_events:
                return False
            events.append(now)
            return True

    def retry_after_s(self, key: str) -> float:
        with self._lock:
            events = self._events.get(key)
            if not events or len(events) < self.max_events:
                return 0.0
            return max(0.0, events[0] + self.window_s - time.monotonic())


class UsageTracker:
    """File-backed monthly counter for dispatched remote jobs."""

    def __init__(self, state_path: Path, *, monthly_budget: int) -> None:
        if monthly_budget < 1:
            raise ValueError("monthly_budget must be at least 1")
        self.state_path = Path(state_path)
        self.monthly_budget = monthly_budget
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def current_month(self) -> str:
        return time.strftime("%Y-%m", time.gmtime())

    def used(self) -> int:
        with self._lock:
            return int(self._load().get(self.current_month(), 0))

    def try_consume(self) -> bool:
        """Increment this month's counter if under budget."""
        month = self.current_month()
        with self._lock:
            state = self._load()
            used = int(state.get(month, 0))
            if used >= self.monthly_budget:
                return False
            state[month] = used + 1
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state, indent=2))
            return True


def cleanup_expired_jobs(output_root: Path, *, max_age_days: float) -> list[str]:
    """Delete job artifact directories older than the retention window."""
    jobs_root = output_root / "cifar100" / "jobs"
    if not jobs_root.is_dir() or max_age_days <= 0:
        return []
    cutoff = time.time() - max_age_days * 86400
    removed = []
    for entry in jobs_root.iterdir():
        if not entry.is_dir():
            continue
        newest = max(
            (path.stat().st_mtime for path in entry.rglob("*") if path.is_file()),
            default=entry.stat().st_mtime,
        )
        if newest < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return removed
